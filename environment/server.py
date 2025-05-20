# 경로 ./environment/server.py
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.load_config import get_experiment_config_and_data, TASK_ENVIRONMENT_MAP
from utils.tools import EarlyStopping
from utils.metrics import metric

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

def train(model, train_loader, config, epochs, device, conn):
    criterion = nn.MSELoss()
    optimizer = config['optimizer'](model.parameters(), lr=config['learning_rate'])
    scheduler = config['scheduler'](optimizer, T_max=epochs)
    early_stopping = EarlyStopping(patience=config['patience'], verbose=False)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        for batch_x, batch_y, _, _ in train_loader:
            batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            # Fix: output size matching label length + pred length
            loss = criterion(output, batch_y[:, -output.shape[1]:, :])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        scheduler.step()

        # 실시간 로그를 클라이언트로 전송
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

def handle_client(conn, addr):
    global active_connections
    with conn_lock:
        active_connections += 1
        print(f"[ACTIVE CONNECTIONS] {active_connections}")

    try:
        data = b""
        while True:
            packet = conn.recv(4096)
            if not packet: break
            data += packet
        params = pickle.loads(data)

        task, dataset, pred_len = params['task'], params['dataset'], params['pred_len']
        model_file, pretrained = params['model_file'], params.get('pretrained')
        epochs = params.get('epochs', 10000)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config, _, train_loader = get_experiment_config_and_data(task, pred_len, dataset, flag='train')
        _, _, test_loader = get_experiment_config_and_data(task, pred_len, dataset, flag='test')

        seq_len = config['seq_len']
        data_dim = next(iter(train_loader))[0].shape[-1]

        model_path = os.path.join('./models', model_file if model_file.endswith('.py') else model_file+'.py')
        model = dynamic_model_loader(model_path, seq_len, pred_len, data_dim).to(device)

        # 파일명 구성
        timestamp = datetime.datetime.now().strftime('%m%d%H%M')
        base_filename = f"{timestamp}_{task}_{dataset}_{os.path.splitext(model_file)[0]}"

        if pretrained:
            model.load_state_dict(torch.load(pretrained, map_location=device))
        else:
            train(model, train_loader, config, epochs, device, conn)

            # 여기서 EarlyStopping에 저장된 체크포인트를 불러와서 저장
            model.load_state_dict(torch.load('./checkpoint', map_location=device))
            
            # 지정된 이름으로 최종 모델 저장
            model_save_path = base_filename + ".pth"
            torch.save(model.state_dict(), model_save_path)

        # 평가 및 메트릭 저장
        final_metrics = evaluate(model, test_loader, device)

        # 최종 메트릭 CSV 저장
        metrics_df = pd.DataFrame([final_metrics], columns=['MAE', 'MSE', 'RMSE', 'MAPE', 'MSPE'])
        csv_save_path = base_filename + ".csv"
        metrics_df.to_csv(csv_save_path, index=False)

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

    while True:
        conn, addr = server.accept()
        threading.Thread(target=handle_client, args=(conn, addr)).start()
