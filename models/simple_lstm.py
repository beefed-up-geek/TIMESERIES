#경로 ./models/simple_lstm.py
import torch
import torch.nn as nn

class model(nn.Module):
    def __init__(self, input_length, pred_length, data_dim, hidden_dim=128, num_layers=2):
        super(model, self).__init__()
        self.input_length = input_length
        self.pred_length = pred_length
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(data_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, data_dim)

    def forward(self, x):
        batch_size = x.size(0)

        # Encoder
        out, (h, c) = self.lstm(x)

        # Decoder (recursive prediction)
        predictions = []
        input_seq = x[:, -1, :]

        for _ in range(self.pred_length):
            input_seq = input_seq.unsqueeze(1)
            out, (h, c) = self.lstm(input_seq, (h, c))
            pred = self.fc(out[:, -1, :])
            predictions.append(pred)
            input_seq = pred

        predictions = torch.stack(predictions, dim=1)
        return predictions
