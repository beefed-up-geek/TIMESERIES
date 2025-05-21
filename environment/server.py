device# 경로 ./environment/server.py
import socket
import threading
import pickle
import torch
import datetime
import os
import numpy as np
import torch.nn as nn
import importlib.util
import sys
import traceback
import pandas as pd
import argparse
from time import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.load_config import get_experiment_config_and_data, TASK_ENVIRONMENT_MAP
from utils.tools import EarlyStopping
from utils.metrics import metric

from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, SCINet, PAttn, TimeXer, \
    WPMixer, MultiPatchFormer, TimeRefining


model_dict = {
    'TimeRefining': TimeRefining,
    'TimesNet': TimesNet,
    'Autoformer': Autoformer,
    'Transformer': Transformer,
    'Nonstationary_Transformer': Nonstationary_Transformer,
    'DLinear': DLinear,
    'FEDformer': FEDformer,
    'Informer': Informer,
    'LightTS': LightTS,
    'Reformer': Reformer,
    'ETSformer': ETSformer,
    'PatchTST': PatchTST,
    'Pyraformer': Pyraformer,
    'MICN': MICN,
    'Crossformer': Crossformer,
    'FiLM': FiLM,
    'iTransformer': iTransformer,
    'Koopa': Koopa,
    'TiDE': TiDE,
    'FreTS': FreTS,
    'MambaSimple': MambaSimple,
    'TimeMixer': TimeMixer,
    'TSMixer': TSMixer,
    'SegRNN': SegRNN,
    'TemporalFusionTransformer': TemporalFusionTransformer,
    "SCINet": SCINet,
    'PAttn': PAttn,
    'TimeXer': TimeXer,
    'WPMixer': WPMixer,
    'MultiPatchFormer': MultiPatchFormer
}

HOST = '0.0.0.0'
PORT = 7777

active_connections = 0
conn_lock = threading.Lock()

def dynamic_model_loader(model_path, input_length, pred_length, data_dim):
    spec = importlib.util.spec_from_file_location("model", model_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    model_class = model_module.model
    return model_class(input_length, pred_length, data_dim)

def evaluate(model, test_loader, device):
    model.eval()
    total_preds, total_trues = [], []

    with torch.no_grad():
        for batch_x, batch_y, _, _ in test_loader:
            batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)
            preds = model(batch_x)
            preds_np = preds.cpu().numpy()
            trues_np = batch_y[:, -preds.shape[1]:, :].cpu().numpy()
            total_preds.append(preds_np)
            total_trues.append(trues_np)

    total_preds = np.concatenate(total_preds, axis=0)
    total_trues = np.concatenate(total_trues, axis=0)
    final_metrics = metric(total_preds, total_trues)
    return final_metrics

def handle_client(conn, addr, args):
    print(args)
    global active_connections
    with conn_lock:
        active_connections += 1
        print(f"[ACTIVE CONNECTIONS] {active_connections}")

    epoch_losses = []  # 에폭별 loss 저장용 리스트 추가

    try:
        data = b""
        while True:
            packet = conn.recv(4096)
            if not packet: break
            data += packet
        params = pickle.loads(data)

        task, dataset, pred_len = params['task_name'], params['data'], params['pred_len']
        model_file, pretrained = params['checkpoints'], params.get('pretrained')
        epochs = params.get('epochs', 10000)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config, _, train_loader = get_experiment_config_and_data(task, pred_len, dataset, flag='train')
        _, _, test_loader = get_experiment_config_and_data(task, pred_len, dataset, flag='test')
        
        # model_path = os.path.join('./models', model_file if model_file.endswith('.py') else model_file+'.py')
        model = model_dict[args.model].Model(args).float()

        timestamp = datetime.datetime.now().strftime('%m%d%H%M')
        base_filename = f"{timestamp}_{task}_{dataset}_{os.path.splitext(model_file)[0]}"

        if pretrained:
            model.load_state_dict(torch.load(pretrained, map_location=device))
        else:
            criterion = nn.MSELoss()
            optimizer = config['optimizer'](model.parameters(), lr=config['learning_rate'])
            scheduler = config['scheduler'](optimizer, T_max=epochs)
            early_stopping = EarlyStopping(patience=config['patience'], verbose=False, save_mode=False)
            train_steps = len(train_loader)
            
            if args.use_amp:
                scaler = torch.cuda.amp.GradScaler()
            
            for epoch in range(args.train_epochs):
                epoch_loss = 0
                iter_count = 0
                train_loss = []
                
                model.train()
                epoch_time = time.time()
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                    iter_count += 1
                    optimizer.zero_grad()
                    batch_x = batch_x.float().to(device)
                    batch_y = batch_y.float().to(device)
                    batch_x_mark = batch_x_mark.float().to(device)
                    batch_y_mark = batch_y_mark.float().to(device)
                    
                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
                    
                    # encoder - decoder
                    if args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                            f_dim = -1 if args.features == 'MS' else 0
                            outputs = outputs[:, -args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
                            loss = criterion(outputs, batch_y)
                            train_loss.append(loss.item())
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if args.features == 'MS' else 0
                        outputs = outputs[:, -args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                    
                    if (i + 1) % 100 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()
                    
                    # output = model(batch_x)
                    # loss = criterion(output, batch_y[:, -output.shape[1]:, :])
                    if args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    epoch_loss += loss.item()

                epoch_loss /= len(train_loader)
                epoch_losses.append(epoch_loss)  # loss 기록
                scheduler.step()

                epoch_log = {
                    'status': 'training',
                    'epoch': epoch,
                    'total_epochs': epochs,
                    'loss': epoch_loss
                }
                conn.sendall(pickle.dumps(epoch_log))

                early_stopping(epoch_loss, model, './')

                if early_stopping.early_stop:
                    conn.sendall(pickle.dumps({'status': 'early_stopping', 'epoch': epoch}))
                    break

            # 학습 완료 후 최종 모델 저장
            model_save_path = base_filename + ".pth"
            torch.save(model.state_dict(), model_save_path)

        final_metrics = evaluate(model, test_loader, device)

        # 최종 메트릭 DataFrame 생성
        metrics_df = pd.DataFrame([final_metrics], columns=['MAE', 'MSE', 'RMSE', 'MAPE', 'MSPE'])

        # Epoch별 loss 기록 DataFrame 생성
        epoch_df = pd.DataFrame({
            'Epoch': list(range(1, len(epoch_losses) + 1)),
            'Epoch_Loss': epoch_losses
        })

        # 최종 결과와 Epoch별 loss를 분리하여 저장
        csv_save_path = base_filename + ".csv"

        with open(csv_save_path, 'w') as f:
            # 최종 성능 메트릭 저장
            metrics_df.to_csv(f, index=False)
            f.write('\n')  # 빈 줄 추가
            # Epoch별 loss 기록 저장
            epoch_df.to_csv(f, index=False)

        conn.sendall(pickle.dumps({'status': 'finished', 'metrics': final_metrics}))

    except Exception as e:
        tb_str = traceback.format_exc()
        conn.sendall(pickle.dumps({'status': 'error', 'message': str(e), 'traceback': tb_str}))
        print(f"[ERROR] {tb_str}")

    finally:
        conn.close()
        with conn_lock:
            active_connections -= 1
            print(f"[ACTIVE CONNECTIONS] {active_connections}")


if __name__ == "__main__":
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen()
    print(f"[LISTENING] Server listening on port {PORT}...")
    
    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test',
                        help="[ETTh2_96_96, ETTh2_96_192]")
    parser.add_argument('--model', type=str, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False,
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    args = parser.parse_args()
    
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]


    while True:
        conn, addr = server.accept()
        threading.Thread(target=handle_client, args=(conn, addr, args)).start()
