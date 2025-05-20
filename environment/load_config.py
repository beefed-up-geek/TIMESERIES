# 경로 ./environment/load_config.py
import argparse
from data_provider.data_factory import data_provider
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

# Task 별 실험 환경 정보 맵
TASK_ENVIRONMENT_MAP = {
    'univariate_short_term': {
        'datasets': ['m4'],
        'pred_length_candidate': [6, 12, 24, 48],  # 가능한 prediction lengths
        'input_length': lambda pred_len: pred_len * 2,  # prediction length의 2배
        'optimizer': Adam,
        'learning_rate': 1e-4,
        'scheduler': OneCycleLR,
        'batch_size': 32,
        'patience': 10
    },
    'few_shot': {
        'datasets': ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather', 'electricity', 'traffic'],
        'pred_length_candidate': [96, 192, 336, 720],
        'input_length': 512,  
        'optimizer': Adam,
        'learning_rate': 1e-4,
        'scheduler': OneCycleLR,
        'batch_size': 16,
        'patience': 10
    },
    'zero_shot': {
        'datasets': ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'],
        'pred_length_candidate': [96, 192, 336, 720],
        'input_length': 512, 
        'optimizer': Adam,
        'learning_rate': 1e-4,
        'scheduler': OneCycleLR,
        'batch_size': 16,
        'patience': 10
    },
    'multivariate_long_term': {
        'datasets': ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather', 'solar', 'electricity', 'traffic', 'exchange_rate'],
        'pred_length_candidate': [96, 192, 336, 720],
        'input_length': 720,  
        'optimizer': Adam,
        'learning_rate': 1e-3,
        'scheduler': CosineAnnealingLR,
        'batch_size': 512,
        'patience': 10
    },
    'multivariate_short_term': {
        'datasets': ['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08'],
        'pred_length_candidate': [12],
        'input_length': 12,  
        'optimizer': Adam,
        'learning_rate': 1e-3,
        'scheduler': CosineAnnealingLR,
        'batch_size': 512,
        'patience': 10
    },
    'imputation': {
        'datasets': ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather', 'electricity'],
        'pred_length_candidate': [0],  # Imputation 특수 목적
        'input_length': 1024,  # 고정
        'optimizer': Adam,
        'learning_rate': 1e-3,
        'scheduler': CosineAnnealingLR,
        'batch_size': 512,
        'patience': 10
    }
}

def get_experiment_config_and_data(task, pred_len, dataset_name, root_path='./dataset', flag='train'):
    assert task in TASK_ENVIRONMENT_MAP, f"Invalid task: {task}"
    task_info = TASK_ENVIRONMENT_MAP[task]

    assert dataset_name in task_info['datasets'], f"{dataset_name} not in {task} task datasets."
    assert pred_len in task_info['pred_length_candidate'], f"Invalid pred length: {pred_len} for task: {task}"

    seq_len = task_info['input_length'](pred_len) if callable(task_info['input_length']) else task_info['input_length']

    args = argparse.Namespace(
        data=dataset_name,
        root_path=root_path,
        seq_len=seq_len,
        label_len=pred_len // 2,
        pred_len=pred_len,
        features='M',
        target='OT',
        embed='timeF',
        freq='h',
        percent=100,
        seasonal_patterns='Monthly',
        batch_size=task_info['batch_size'],
        num_workers=4
    )

    data_set, data_loader = data_provider(args, flag)

    optimizer_fn = task_info['optimizer']
    learning_rate = task_info['learning_rate']
    scheduler_fn = task_info['scheduler']

    config = {
        'seq_len': seq_len,
        'label_len': pred_len // 2,
        'pred_len': pred_len,
        'optimizer': optimizer_fn,
        'learning_rate': learning_rate,
        'scheduler': scheduler_fn,
        'batch_size': task_info['batch_size'],
        'patience': task_info['patience']
    }

    return config, data_set, data_loader
