# client.py
import socket
import pickle

HOST = '127.0.0.1'
PORT = 7777

params = {
    'task': 'multivariate_long_term',
    'dataset': 'solar',
    'pred_len': 96,
    'model_file': 'simple_lstm',
    'epochs': 2,
    'pretrained': None
}

def send_request(params):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.connect((HOST, PORT))
        print("[CLIENT] Connected to the server.")

        client.sendall(pickle.dumps(params))
        client.shutdown(socket.SHUT_WR)
        print("[CLIENT] Request sent to the server.")

        response_data = b""
        while True:
            packet = client.recv(4096)
            if not packet:
                print("[CLIENT] Server connection closed.")
                break
            response_data += packet

            while True:
                try:
                    result, response_data = pickle.loads(response_data), b""
                    if result['status'] == 'training':
                        epoch = result['epoch']
                        total_epochs = result['total_epochs']
                        loss = result['loss']
                        print(f"[TRAINING] Epoch [{epoch}/{total_epochs}] - Loss: {loss:.6f}")
                    elif result['status'] == 'early_stopping':
                        epoch = result['epoch']
                        print(f"[EARLY STOPPING] Triggered at epoch {epoch}.")
                    elif result['status'] == 'finished':
                        metrics = result['metrics']
                        print("[TRAINING COMPLETED] Final metrics (MAE, MSE, RMSE, MAPE, MSPE):", metrics)
                    elif result['status'] == 'error':
                        print("[ERROR]", result['message'])
                        print("[TRACEBACK]", result['traceback'])
                    else:
                        print("[OTHER MESSAGE]", result)
                except (pickle.UnpicklingError, EOFError):
                    break

send_request(params)
