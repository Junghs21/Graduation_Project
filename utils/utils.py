import torch
import random
from typing import Any, Dict, List
import argparse
import copy
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np
from collections import defaultdict
from client import *
import math

def average_weights(weights: List[Dict[str, torch.Tensor]], steps) -> Dict[str, torch.Tensor]:
    total_step = sum(steps)
    weights_avg = copy.deepcopy(weights[0])

    for key in weights_avg.keys():
        for i in range(0, len(weights)):
            factor = steps[i] / total_step
            weights_avg[key] = weights_avg[key].detach().cpu() 
            weights_avg[key] += weights[i][key].detach().cpu() * factor

    return weights_avg


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", type=str, default="163.180.117.36")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=50051)

    parser.add_argument("--data_root", type=str, default="../datasets/")
    parser.add_argument("--data", type=str, default="cifar10")
    parser.add_argument("--model_name", type=str, default="cnn")
    parser.add_argument("--aggregator", type=str, default="fedavg")
    parser.add_argument("--shards", type=int, default=2)

    parser.add_argument("--n_clients", type=int, default=14)
    parser.add_argument("--n_class", type=int, default=26)
    parser.add_argument("--frac", type=float, default=0.1)
    parser.add_argument("--target_accuracy", type=float, default=0.995)
    parser.add_argument("--max_cluster", type=int, default=4)

    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--clients_per_rounds", type=int, default=5)
    parser.add_argument("--n_client_epochs", type=int, default=5)
    parser.add_argument("--optim", type=str, default="sgd")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--early_stopping", type=int, default=1)
    parser.add_argument("--update_counter", type=int, default=1)
    parser.add_argument("--non_iid", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--is_testing", type=int, default=0)
    
    # Cluster
    parser.add_argument("--n_cluster", type=int, default=2)

    # FedProx
    parser.add_argument('--mu', type=float, default=0.3, help='proximal term constant')
    
    # GPU
    parser.add_argument('--use_gpu', action='store_true', help='use gpu')
    parser.add_argument('--use_multiple_gpu', action='store_true', help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='number of gpus')
    
    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    return args

def split_dataset_non_iid(dataset, num_clients):
    random.seed(1)
    class_data = defaultdict(list)
    sample_per_client = math.floor(len(dataset) / num_clients)
    
    # 클래스별로 분리
    for idx, (data, label) in enumerate(dataset):
        class_data[label].append(idx)
        
    client_indices = []
    
    for client_id in range(num_clients):
        # 각 클라이언트에 대해 무작위로 지정된 수만큼의 샘플 인덱스를 할당
        class_indices = []
        client_sample_size = sample_per_client
        for label, indices in class_data.items():
            # 남아 있는 인덱스의 일부를 무작위로 선택
            split_size = random.randint(0, min(math.floor(len(indices)), client_sample_size))
            
            if client_id < num_clients - 1:
                datas = indices[:split_size]
                class_data[label] = indices[split_size:]  # 할당된 부분 제외
            else:
                datas = indices[:]  # 마지막 클라이언트는 남은 모든 데이터를 받음
            client_sample_size -= split_size
            class_indices.extend(datas)
            
        client_indices.append(class_indices)
    
    return client_indices

def generate_random_device_data_ratio(num_devices, min_ratio=1e-2):
    random.seed(1)
    """
    각 디바이스에 할당할 랜덤 데이터 비율을 생성하는 함수.
    각 비율은 최소 min_ratio보다 크고, 비율의 합은 1이 됨.
    
    Args:
        num_devices: 디바이스(클라이언트) 수
        min_ratio: 각 디바이스에 할당할 최소 비율 (기본값: 1e-2)
    
    Returns:
        각 디바이스에 할당할 데이터 비율 리스트
    """
    # 최소 비율을 먼저 각 디바이스에 할당
    remaining_ratio = 1 - (num_devices * min_ratio)
    
    if remaining_ratio < 0:
        raise ValueError(f"min_ratio가 너무 큽니다. num_devices가 {num_devices}일 때 min_ratio는 {1/num_devices} 이하로 설정해야 합니다.")
    
    # 남은 비율을 랜덤하게 생성
    random_ratios = np.random.rand(num_devices)
    
    # 랜덤 비율을 남은 비율로 정규화
    random_ratios = random_ratios / random_ratios.sum() * remaining_ratio
    
    # 최소 비율을 더하여 최종 비율 생성
    device_data_ratio = random_ratios + min_ratio
    return device_data_ratio

def create_random_dataset(dataset, num_devices, num_classes=10, device_data_ratio=None):
    """
    Args:
        dataset: 전체 데이터셋 (CIFAR-10)
        num_devices: 디바이스 수 (클라이언트 수)
        num_classes: 클래스 수 (기본값 10, CIFAR-10은 10개 클래스)
        device_data_ratio: 각 디바이스에 할당할 데이터 비율 (리스트, 기본값 None)
        
    Returns:
        각 디바이스에 할당된 데이터 인덱스를 포함한 리스트
    """
                       
    if device_data_ratio is None:
        # 디바이스별 데이터를 균등하게 할당할 기본 비율
        device_data_ratio = [1 / num_devices] * num_devices
    else:
        assert len(device_data_ratio) == num_devices, "디바이스 수와 비율 리스트의 길이가 일치해야 합니다."
        assert abs(sum(device_data_ratio) - 1) < 1e-5, "비율의 합이 1이 되어야 합니다."
    
    targets = np.array(dataset.targets)
    
    # 각 클래스를 인덱싱
    class_indices = [np.where(targets == i)[0] for i in range(num_classes)]
    
    # 각 클래스 내에서 섞음
    for class_idx in class_indices:
        np.random.shuffle(class_idx)
    
    # Non-IID 분배를 위한 준비
    device_data_indices = [[] for _ in range(num_devices)]
    
    # 각 디바이스에 비율에 맞춰 데이터를 할당
    for i in range(num_classes):
        class_data = class_indices[i]
        total_class_data = len(class_data)
        
        # 각 디바이스에 할당할 데이터 양을 비율에 따라 계산
        start_idx = 0
        for device_idx in range(num_devices):
            num_samples = int(total_class_data * device_data_ratio[device_idx])
            end_idx = start_idx + num_samples
            
            # 해당 디바이스에 데이터를 할당
            device_data_indices[device_idx].extend(class_data[start_idx:end_idx])
            start_idx = end_idx
        
        # 남은 데이터가 있으면 임의의 디바이스에 할당
        remaining_data = class_data[start_idx:]
        if len(remaining_data) > 0:
            for device_idx in range(len(remaining_data)):
                device_data_indices[device_idx].append(remaining_data[device_idx])

    return device_data_indices

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

class UpdateCounter:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.update = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'Cluster update counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.update = True
        else:
            self.best_score = score
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            self.val_loss_min = val_loss
            self.counter = 0

