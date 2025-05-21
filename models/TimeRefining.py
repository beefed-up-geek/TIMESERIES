import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize

class DFTSeriesDecompMulti(nn.Module):
    """Return (season_A/B/C, trend_A/B/C) – tensors in (B, L, C)."""

    def __init__(self, top_k: int = 5):
        super().__init__()
        self.top_k = top_k
        self.levels = 3

    def _keep_top_k(self, x: torch.Tensor) -> torch.Tensor:
        xf = torch.fft.rfft(x, dim=-2)
        mag = xf.abs(); mag[..., 0] = 0
        topk_vals = torch.topk(mag, self.top_k, dim=-2).values
        thresh = topk_vals.min(dim=-2, keepdim=True).values
        xf[mag <= thresh] = 0.0
        return xf

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        residual = x
        seasons, trends = [], []
        for _ in range(self.levels):
            xf = self._keep_top_k(residual)
            seasonal = torch.fft.irfft(xf, n=residual.shape[-2], dim=-2)
            residual = residual - seasonal
            seasons.append(seasonal)
            trends.append(residual)
        return (*seasons, *trends)
    
class ForecastBlock(nn.Module):
    def __init__(self, in_ch: int, d_model: int, pred_len: int, enc_in: int):
        super().__init__()
        self.pred_len, self.enc_in = pred_len, enc_in
        self.conv = nn.Conv1d(in_ch, d_model, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, pred_len * enc_in)
        # nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        # nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L, C) → (B, pred_len, enc_in)
        h = F.relu(self.conv(x.permute(0, 2, 1)))
        h = self.pool(h).squeeze(-1)
        out = self.fc(h).view(-1, self.pred_len, self.enc_in)
        return out
    
class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.loss_weights = (0.25, 0.5, 1.0)
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence

        self.preprocess = series_decomp(configs.moving_avg)
        self.enc_in = configs.enc_in
        # self.use_future_temporal_feature = configs.use_future_temporal_feature

        if self.channel_independence == 1:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)

        self.layer = configs.e_layers

        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )
        
        self.normalize_layer = Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
            
        self.decomp = DFTSeriesDecompMulti()
        in_ch = configs.enc_in * 2
        self.block_A = ForecastBlock(configs.enc_in, configs.d_model, configs.pred_len, configs.enc_in)
        self.block_B = ForecastBlock(configs.enc_in, configs.d_model, configs.pred_len, configs.enc_in)
        self.block_C = ForecastBlock(configs.enc_in, configs.d_model, configs.pred_len, configs.enc_in)
    def _concat_st(self, season: torch.Tensor, trend: torch.Tensor) -> torch.Tensor:
        return torch.cat([season, trend], dim=-1)
    
    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return x_enc, x_mark_enc
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        x_mark_enc = x_mark_sampling_list if x_mark_enc is not None else None

        return x_enc, x_mark_enc


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)
        
        x_list = []
        x_mark_list = []
        
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_list.append(x)
                    x_mark = x_mark.repeat(N, 1, 1)
                    x_mark_list.append(x_mark)
                else:
                    x_list.append(x)
                    x_mark_list.append(x_mark)
                    
        else:
            for i, x in zip(range(len(x_enc)), x_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        enc_out_list = []
        
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)
                enc_out_list.append(enc_out)
                
        
        
        
        
        if x_mark_enc is not None:
            B, T, N = x_enc.size()
            x_enc = self.normalize_layer(x_enc, 'norm')
            # if self.channel_independence == 1:
            #     x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            #     x_mark_enc = x_mark_enc.repeat(N, 1, 1)
        
        else:
            B, T, N = x_enc.size()
            x_enc = self.normalize_layer(x_enc, 'norm')
            # if self.channel_independence == 1:
            #     x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            
        
        
        sA, sB, sC, tA, tB, tC = self.decomp(x_enc)
        
        
        
        # Module A
        pred_A = self.block_A(sA + tA)
        # Module B (predict residual)
        res_B = self.block_B(sB + tB)
        pred_B = pred_A + res_B
        # Module C (predict residual)
        res_C = self.block_C(sC + tC)
        pred_C = pred_B + res_C
        
        pred_A = self.normalize_layer(pred_A, "denorm")
        pred_B = self.normalize_layer(pred_B, "denorm")
        pred_C = self.normalize_layer(pred_C, "denorm")
        
        return pred_C

    

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out
        