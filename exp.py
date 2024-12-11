from torchmetrics.classification import MulticlassF1Score
from client import Client
from typing import Any, Dict, List, Optional, Tuple
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.multiprocessing as mp
import torch
import random
from train import *
from model import *
from utils.utils import *
from utils.metrics import *
from utils.clusters import *
import os
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from grand import federated_master, federated_worker

random.seed(1)

class Exp(object):
    def __init__(self, args: Dict[str, Any]):
        self.args = args
        self.Client = Client(args)
        self.gRPCClient = federated_worker.gRPCClient("localhost",  self.args.port)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.patience = self.args.patience
        self.clients_per_rounds = self.args.clients_per_rounds
        self.server = None
        self.epoch = self.args.n_client_epochs
        self.patience = 0
        self.n_clients = self.args.n_clients
        self.n_class = self.args.n_class
        self.mu = self.args.mu
        self.processes = []
        self.device_dataloaders = []
        self.device_subset = []
        self.start_events = []
        self.states = {}
        self.Client._printClients()
        self.serve()
        print(args)
        
        torch.cuda.empty_cache()
        
        transform = transforms.Compose([
            transforms.ToTensor(),  # 이미지를 PyTorch Tensor로 변환 (0~255 값을 0~1로 스케일링)
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # 평균과 표준편차로 정규화
        ])
        
        if self.args.data == "cifar10":
            self.train_dataset = datasets.CIFAR10(root='../datasets/cifar10/', train=True, download=True, transform=transform)
        elif self.args.data == "cifar100":
            self.train_dataset = datasets.CIFAR100(root='../datasets/cifar100/', train=True, download=True,  transform=transform)
        elif self.args.data == "FashionMNIST":
            self.train_dataset = datasets.FashionMNIST(root='../datasets/FashionMNIST/', train=True, download=True,  transform=transforms.ToTensor())
        elif self.args.data == "EMNIST":
            self.train_dataset = datasets.EMNIST(root='../datasets/EMNIST/', split = 'byclass', train=True, download=True,  transform=transforms.ToTensor())
            
    def serve(self):
        mp.set_start_method('spawn', force=True)
        self.conn, conn = mp.Pipe()
        p = mp.Process(target=federated_master.serve, args=(2000 * 1024 * 1024, self.args.port, self.args.rounds, conn))
        self.server = p
        p.start()

    def createClients(self, clients):        
        # 클라이언트별로 프로세스 생성
        for idx in range(self.n_clients):
            start =mp.Event()
            if self.args.aggregator == "fedavg":
                p = mp.Process(target=train_local_client_prox, args=(clients, self.device_dataloaders[idx], self.clients_per_rounds * self.epoch, idx, self.args.port, 0, start))
            elif self.args.aggregator == "fedprox":
                p = mp.Process(target=train_local_client_prox, args=(clients, self.device_dataloaders[idx], self.clients_per_rounds * self.epoch, idx, self.args.port, self.mu, start))
            elif self.args.aggregator == "fednova":
                p = mp.Process(target=train_local_client_nova, args=(clients, self.device_dataloaders[idx], self.clients_per_rounds * self.epoch, idx, self.args.port, start))
            
            self.processes.append(p)
            self.start_events.append(start)
            
        for i, p in enumerate(self.processes):
            print(f"Client {i} created! Data: {len(self.device_dataloaders[i])}")
            p.start()
            
        for i, p in enumerate(self.start_events):
            p.set()
            
        print(f"Training start!")
            
    def updateModel(self, model, conns, devices):
        if devices is None:
            for idx in range(self.n_clients):
                conns[idx].send(model)       
        else:
            for idx in range(self.n_clients):
                if idx in devices:
                    conns[idx].send([model, True])
                else:
                    conns[idx].send([model, False])
    
    
    # 클라이언트 및 서버 실행 및 종료
    def run(self, setting):
        f = open(f"../metrics/results/{self.args.aggregator}/{self.args.aggregator}_{self.args.data}_{self.args.model_name}_{'non_iid' if self.args.non_iid else 'iid'}.csv", "a")
        f.write("round, avg_loss, avg_accuracy, avg_mae, avg_mse, avg_rae, avg_rmse, time\n")
        
        path = os.path.join("./checkpoints", setting)
        if not os.path.exists(path):
            os.makedirs(path)
        
        clients = self.Client._createClient()
        
        mp.set_start_method('spawn', force=True)
        
        if self.args.non_iid == 1:
            device_data_indices = split_dataset_non_iid(self.train_dataset, self.n_clients)
        else:
            ratio = generate_random_device_data_ratio(self.n_clients)
            device_data_indices = create_random_dataset(self.train_dataset, self.n_clients, self.n_class, ratio)
        
        for idx in range(self.n_clients):
            indices = device_data_indices[idx]
            batch_size = clients[idx][2]
            subset = Subset(self.train_dataset, indices)
            dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True)
            self.device_subset.append(subset)
            self.device_dataloaders.append(dataloader)
        
        self.gRPCClient.setup(self.args.data, self.args.n_class, self.args.model_name)
        self.createClients(clients)
        
        # 서버와 클라이언트가 학습이 종료되면 서버에서 파이프로 exp에 요청 -> 클라 종료 -> 서버 종료 순으로 동작.
        s = time.time()
        
        while self.conn.poll() == False: 
            pass
        
        print(f"Train terminated. | time: {time.time() - s}")
    
        for idx in range(self.n_clients):
            self.processes[idx].terminate()
            self.processes[idx].join()
            
        self.server.terminate()
        self.server.join()
        self.conn.close()
        f.close()
        
        torch.cuda.empty_cache()