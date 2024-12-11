from model import *
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import torch.nn.functional as F
from grand.federated_master import *
from grand.federated_worker import *

# 입력받은 설정의 클라이언트 생성
class Client:
    def __init__(self, args):
        self.args = args
        self.gpu = args.gpu
        self.n_clients = args.n_clients
        self.host = args.host
        self.epochs = args.n_client_epochs
        self.momentum = args.momentum
        self.lr = args.lr
        self.n_class = args.n_class
        self.frac = args.frac
        self.rounds = args.rounds
        self.clients = self._createClient()
        
    def _setup(self):
        random.seed(1)
        setting = []
        # 지연시간, 통신시간, 배치크기
        for i in range(self.n_clients):
            if self.args.is_testing:
                setting.append([0, 0, self.args.batch_size])
                continue
            setting.append([0.2 * i, i * 0.05, self.args.batch_size])
        return setting
            
    def _printClients(self):
        for client in self.clients:
            delay, transfer, batch, _, _, _, _, idx, iter, _ = client
            print(f"client_id:{idx} | rounds:{iter} | model:{self.args.model_name} | batch:{batch} | optimizer:{self.args.optim} | train delay:{delay} | transfer delay:{transfer}")
    
    def _createClient(self):
        # 지연시간, 통신시간, 배치크기, 모델, 옵티마이저, 로스함수, 클라이언트 아이피, 인덱스, 라운드 횟수, gpu 
        settings = self._setup()
        clients = []
        
        for idx in range(self.n_clients):
            setting = settings[idx]
            if self.args.data == "MNIST" or self.args.data == "FashionMNIST" or self.args.data == "EMNIST":
                model = CNN(1, self.n_class) if self.args.model_name == "cnn" else mobilenet(1, 1, self.n_class)
                setting.extend([model, optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum), nn.CrossEntropyLoss(ignore_index=-1), self.host, idx, self.rounds, idx % self.gpu if self.args.use_multiple_gpu else 1])            
            elif self.args.data == "cifar10" or self.args.data == "cifar100":
                model = CNN(3, self.n_class) if self.args.model_name == "cnn" else mobilenet(1, 3, self.n_class)
                setting.extend([model, optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum), nn.CrossEntropyLoss(ignore_index=-1), self.host, idx, self.rounds, idx % self.gpu if self.args.use_multiple_gpu else 1])
            
            clients.append(setting)
        
        return clients