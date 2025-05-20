# Time Series Forecasting 실험 환경

## 프로젝트 구조

프로젝트의 디렉터리 구조는 다음과 같습니다.

```
TIMESERIES/
├── data_provider/
│   ├── __init__.py
│   ├── data_factory.py
│   ├── data_loader.py
│   └── m4.py
├── dataset/
│   ├── electricity/
│   ├── ETT-small/
│   ├── exchange_rate/
│   ├── illness/
│   ├── m4/
│   ├── PEMS/
│   ├── solar/
│   ├── traffic/
│   └── weather/
├── environment/
│   ├── client.py
│   ├── load_config.py
│   └── server.py
├── models/
│   ├── model_template.py
│   ├── simple_lstm.py
│   └── simple_mlp.py
├── utils/
├── README.md
└── requirements.txt
```

## 환경 설정

### 의존성 설치

```bash
pip install -r requirements.txt
```
이후 추가로 본인의 환경에 맞는 torch==2.2.2 버전을 설치해주세요!
### 데이터셋 다운로드

[구글 드라이브 링크](https://drive.google.com/drive/folders/15EHw24-SYDUMglJ54V-va_RpQxFUUcN6?usp=drive_link)에 접속하여 모든 데이터를 다운받은 후, 프로젝트의 `dataset` 폴더에 저장합니다.

## 서버 실행 방법

서버를 실행하려면 다음 명령어를 사용합니다.</br>
TIMESERIES 홈 디렉터리에서 실행해야합니다! </br>
서버는 여러 클라이언트의 요청을 받아 동시에 여러 실험을 진행할 수 있습니다!
```bash
python -m environment.server
```

## 클라이언트 사용 방법

다음의 `client.py`의 `params` 값을 수정하여 실험을 수행할 수 있습니다.

```python
# 경로 ./environment/client.py
import socket
import pickle

HOST = '127.0.0.1'
PORT = 7777

params = {
    'task': 'multivariate_long_term', # task명
    'dataset': 'solar',               # dataset명
    'pred_len': 96,                   # 예측 길이
    'model_file': 'simple_lstm',      # 모델 파일명
    'epochs': 2,                      # 에포크 수 (선택, 기본값 10000)
    'pretrained': None                # 사전 학습된 모델 파일 경로 (선택)
}

def send_request(params):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.connect((HOST, PORT))
        client.sendall(pickle.dumps(params))
        client.shutdown(socket.SHUT_WR)

        response_data = b""
        while True:
            packet = client.recv(4096)
            if not packet: break
            response_data += packet

        result = pickle.loads(response_data)
        print("Server response:", result)

send_request(params)
```

## 사용 가능한 Task 및 Dataset

| Task                      | Dataset                                                                          | Prediction Lengths   | Input Length |
| ------------------------- | -------------------------------------------------------------------------------- | -------------------- | ------------ |
| univariate\_short\_term   | m4                                                                               | \[6, 12, 24, 48]     | 2배 pred\_len |
| few\_shot                 | ETTh1, ETTh2, ETTm1, ETTm2, weather, electricity, traffic                        | \[96, 192, 336, 720] | 512          |
| zero\_shot                | ETTh1, ETTh2, ETTm1, ETTm2                                                       | \[96, 192, 336, 720] | 512          |
| multivariate\_long\_term  | ETTh1, ETTh2, ETTm1, ETTm2, weather, solar, electricity, traffic, exchange\_rate | \[96, 192, 336, 720] | 720          |
| multivariate\_short\_term | PEMS03, PEMS04, PEMS07, PEMS08                                                   | \[12]                | 12           |
| imputation                | ETTh1, ETTh2, ETTm1, ETTm2, weather, electricity                                 | \[0]                 | 1024         |

## 다양한 클라이언트 호출 예시

다음은 다양한 실험 조건을 설정하여 사용할 수 있는 `params` 예시입니다.

```python
{'task': 'multivariate_long_term', 'dataset': 'solar', 'pred_len': 96, 'model_file': 'simple_lstm'}
{'task': 'few_shot', 'dataset': 'weather', 'pred_len': 336, 'model_file': 'simple_mlp', 'epochs': 50}
{'task': 'zero_shot', 'dataset': 'ETTh1', 'pred_len': 720, 'model_file': 'simple_lstm'}
{'task': 'multivariate_short_term', 'dataset': 'PEMS04', 'pred_len': 12, 'model_file': 'simple_mlp'}
{'task': 'imputation', 'dataset': 'electricity', 'pred_len': 0, 'model_file': 'simple_lstm', 'epochs': 100}
{'task': 'univariate_short_term', 'dataset': 'm4', 'pred_len': 48, 'model_file': 'simple_mlp'}
{'task': 'few_shot', 'dataset': 'traffic', 'pred_len': 192, 'model_file': 'simple_lstm'}
{'task': 'multivariate_long_term', 'dataset': 'exchange_rate', 'pred_len': 720, 'model_file': 'simple_mlp'}
{'task': 'zero_shot', 'dataset': 'ETTm2', 'pred_len': 96, 'model_file': 'simple_lstm'}
{'task': 'multivariate_long_term', 'dataset': 'electricity', 'pred_len': 336, 'model_file': 'simple_mlp', 'epochs': 500}

```

## 나만의 모델 만들기

아래의 템플릿을 참고하여 자신만의 모델을 만들 수 있습니다. </br>
EDIT HERE를 수정해서 모델을 만드세요

```python
# 경로 ./models/model_template.py
class model(nn.Module):
    def __init__(self, input_length, pred_length, data_dim):
        super(model, self).__init__()
        
        self.input_length = input_length
        self.pred_length = pred_length
        self.data_dim = data_dim

        # ====EDIT HERE====
        self.layers = nn.Sequential(
        )
        # ==================

    def forward(self, x):
        # ====EDIT HERE====
       
        # ==================
        pass

```
