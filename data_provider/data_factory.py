# 경로: ./data_provider/data_factory.py
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, Dataset_PEMS
from torch.utils.data import DataLoader
import os

data_dict = {
    'ETTh1': (Dataset_ETT_hour, 'ETT-small/ETTh1.csv'),
    'ETTh2': (Dataset_ETT_hour, 'ETT-small/ETTh2.csv'),
    'ETTm1': (Dataset_ETT_minute, 'ETT-small/ETTm1.csv'),
    'ETTm2': (Dataset_ETT_minute, 'ETT-small/ETTm2.csv'),
    'electricity': (Dataset_Custom, 'electricity/electricity.csv'),
    'exchange_rate': (Dataset_Custom, 'exchange_rate/exchange_rate.csv'),
    'illness': (Dataset_Custom, 'illness/national_illness.csv'),
    'solar': (Dataset_Custom, 'solar/solar_AL.csv'),
    'traffic': (Dataset_Custom, 'traffic/traffic.csv'),
    'weather': (Dataset_Custom, 'weather/weather.csv'),
    'm4': (Dataset_M4, 'm4'),  
    'PEMS03': (Dataset_PEMS, 'PEMS/PEMS03.csv'),
    'PEMS04': (Dataset_PEMS, 'PEMS/PEMS04.csv'),
    'PEMS07': (Dataset_PEMS, 'PEMS/PEMS07.csv'),
    'PEMS08': (Dataset_PEMS, 'PEMS/PEMS08.csv'),
}

def data_provider(args, flag):
    Data, data_path = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent

    shuffle_flag = False if flag == 'test' else True
    drop_last = args.data != 'm4'
    batch_size = args.batch_size
    freq = args.freq

    # 모든 컬럼을 target으로 설정 (명시적으로 None을 전달)
    data_set = Data(
        root_path=args.root_path,
        data_path=data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features='M',    # 항상 다변수 (모든 feature 사용)
        target=None,     # 수정된 부분 (명시적으로 None 전달)
        timeenc=timeenc,
        freq=freq,
        percent=percent,
        seasonal_patterns=args.seasonal_patterns
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )

    return data_set, data_loader
