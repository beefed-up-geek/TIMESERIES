#경로 ./models/simple_mlp.py
import torch
import torch.nn as nn

class model(nn.Module):
    def __init__(self, input_length, pred_length, data_dim, hidden_dim=128):
        super(model, self).__init__()
        self.input_length = input_length
        self.pred_length = pred_length
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_length * data_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pred_length * data_dim)
        )

    def forward(self, x):
        batch_size = x.size(0)
        output = self.model(x)
        output = output.view(batch_size, self.pred_length, self.data_dim)
        return output
