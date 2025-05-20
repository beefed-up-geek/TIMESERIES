#경로 ./models/model_template.py
import torch
import torch.nn as nn

class model(nn.Module):
    def __init__(self, input_length, pred_length, data_dim):
        super(model, self).__init__()
        
        self.input_length = input_length
        self.pred_length = pred_length
        self.data_dim = data_dim

        # 모델에서 사용할 레이어를 정의해주세요.
        # ====EDIT HERE====
        self.layers = nn.Sequential(
            # 예시:
            # nn.Linear(input_length * data_dim, 128),
            # nn.ReLU(),
            # nn.Linear(128, pred_length * data_dim)
        )
        # ==================

    def forward(self, x):
        # forward pass 로직을 작성해주세요.
        # 입력: x.shape = (batch_size, input_length, data_dim)
        # 출력: (batch_size, pred_length, data_dim)

        # ====EDIT HERE====
        # 예시:
        # batch_size = x.size(0)
        # output = self.layers(x.view(batch_size, -1))
        # output = output.view(batch_size, self.pred_length, self.data_dim)
        # return output
        # ==================
        pass
