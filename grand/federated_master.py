import grpc
from concurrent import futures
import grand.federated_pb2_grpc as pb2_grpc
import grand.federated_pb2 as pb2
import pickle
import torch
import torch.nn as nn
import copy
import traceback
from model import CNN, MobileNet
from utils.metrics import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class grpcServiceServicer(pb2_grpc.grpcServiceServicer):
    def __init__(self, round, conn):
        super(grpcServiceServicer, self).__init__()
        self.global_state = None
        self.test_loader = None
        self.model = None
        self.val = None
        self.round = round
        self.current_round = 0
        self.conn = conn
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 이미지를 PyTorch Tensor로 변환 (0~255 값을 0~1로 스케일링)
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # 평균과 표준편차로 정규화
        ])
        self.weight_buffer = {}  # 가중치 수집용 버퍼
        self.clients = {}  # 연결된 클라이언트 관리 
            
    def valSetup(self, request, context):
        data, n_class, model_name = request.data, request.n_class, request.model_name
        try:
            if self.test_loader is None and self.model is None:
                if data == "cifar10":
                    test_dataset = datasets.CIFAR10(root='../datasets/cifar10/', train=False, download=True, transform=self.transform)
                    self.test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True, num_workers=0)
                    self.model = CNN(3, n_class) if model_name == "cnn" else MobileNet(1, 3, n_class)
                elif data == "cifar100":
                    test_dataset = datasets.CIFAR100(root='../datasets/cifar100/', train=False, download=True, transform=self.transform)
                    self.test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True, num_workers=0)
                    self.model = CNN(3, n_class) if model_name == "cnn" else MobileNet(1, 3, n_class)
                elif data == "EMNIST":
                    self.train_dataset = datasets.EMNIST(root='../datasets/EMNIST/', split = 'byclass', train=True, download=True,  transform=transforms.ToTensor())
                    test_dataset = datasets.EMNIST(root='../datasets/EMNIST/', split = 'byclass', train=False, download=True, transform=transforms.ToTensor())
                    self.test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True, num_workers=0)
                    self.model = CNN(1, n_class) if model_name == "cnn" else MobileNet(1, 1, n_class)
                elif data == "FashionMNIST":
                    self.train_dataset = datasets.FashionMNIST(root='../datasets/FashionMNIST/', train=True, download=True,  transform=transforms.ToTensor())
                    test_dataset = datasets.FashionMNIST(root='../datasets/FashionMNIST/', train=False, download=True, transform=transforms.ToTensor())
                    self.test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True, num_workers=0)
                    self.model = CNN(1, n_class) if model_name == "cnn" else MobileNet(1, 1, n_class)
            self.global_state = self.model.state_dict()
                
        except Exception as e:
            print(e)
            context.set_details('Failed to deserialize states')
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.EmptyResponse(message="Error")
        
        return pb2.EmptyResponse(message="success")
        
    def valid(self, val_model):    
        self.model.to(self.device)
        self.model.load_state_dict(val_model)
        
        with torch.no_grad():  # 그라디언트 계산 비활성화
            self.model.eval()
            criterion = nn.CrossEntropyLoss()
            correct = 0
            total = 0
            total_loss = 0.0
            mae, mse, rse, rmse = 0, 0, 0, 0
            
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # 모델 출력 예측
                output = self.model(data)
                loss = criterion(output, target)
                
                a, b, c, d = metric(torch.argmax(output, dim=-1).cpu(), target.cpu())
                mae += a; mse += b; rse += c; rmse += d

                # 손실 집계
                total_loss += loss.item() * data.size(0)  # 배치 크기로 가중합
                
                # 예측값을 통한 정확도 계산
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
        # 평균 손실 및 정확도 계산
        avg_loss = total_loss / total
        accuracy = correct / total
        avg_mae = mae / total        
        avg_mse = mse / total        
        avg_rse = rse / total        
        avg_rmse = rmse / total     

        return {"loss" : avg_loss, "accuracy" : accuracy, "mae" : avg_mae, "mse" : avg_mse, "rse" : avg_rse, "rmse" : avg_rmse}
    
    def aggregate_weights(self, client_states, weights, setting):  # aggregation 메서드 분리
        model_state = copy.deepcopy(client_states[0])
        keys = model_state.keys()

        for key in keys:
            model_state[key] = torch.zeros_like(model_state[key].to(self.device).float())
            for client_state in client_states:
                factor = 1 / len(client_states)
                if setting == "fednova":
                    factor = weights[client_state] / sum(weights.values())
                model_state[key] += client_state[key].to(self.device).float() * factor
        return model_state
    
    def broadcast_global_weights(self):  # 클라이언트에 전송
        for client_id, stub in self.clients.items():
            try:
                request = pb2.GlobalState(state=pickle.dumps(self.global_state))
                stub.getGlobalModel(request)
                print(f"Sent global weights to client {client_id}")
            except Exception as e:
                print(f"Failed to send global weights to client {client_id}: {e}")
        
    def sendState(self, request, context):
        try:
            client_states = pickle.loads(request.state)
            weights = pickle.loads(request.weights)
            setting = request.setting
            client_id = request.drop

            if self.current_round == self.round:
                self.conn.send("train complete.")
                self.conn.close()
                print("train complete.")
                
            elif len(list(self.weight_buffer.keys())) < 3:
                self.weight_buffer[client_id] = client_states  # 버퍼에 가중치 추가
                print(f"Received weights from client {client_id}. Buffer size: {len(self.weight_buffer.keys())}")

                if len(list(self.weight_buffer.keys())) == 3:
                    print("Starting aggregation...")
                    self.global_state = self.aggregate_weights(list(self.weight_buffer.values()), weights, setting)  # aggregation 호출
                    
                    print("Aggregation completed. Broadcasting global weights to clients.")
                    self.broadcast_global_weights()  # 클라이언트로 global weight 전송

                    # 성능 평가
                    self.val = self.valid(self.global_state)
                    self.current_round = self.current_round + 1
                    print(f"============ Round {self.current_round} | Loss {self.val['loss']} | Accuracy {self.val['accuracy']} ============")
                    
                    self.weight_buffer.clear()  # 버퍼 초기화
                    
            response = pb2.GlobalState(
                    state=b"x",
                    loss=0, 
                    accuracy=0,  
                    mae=0,  
                    mse=0,  
                    rse=0,  
                    rmse=0
                )
            return response

        except Exception as e:
            print(traceback.format_exc())
            context.set_details('Failed to process weights sendStates 내부 오류')
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.GlobalState(state=b"Error")
    
    def getGlobalModel(self, request, context):
        val = self.val
        return pb2.GlobalState(state=pickle.dumps(self.global_state),loss=val["loss"],accuracy=val["accuracy"],mae=val["mae"],mse=val["mse"],rse=val["rse"],rmse=val["rmse"])
        
# gRPC server
def serve(MAX_MESSAGE_LENGTH, PORT, round, conn):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                         options=[
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)
    ])
    pb2_grpc.add_grpcServiceServicer_to_server(grpcServiceServicer(round, conn), server)
    server.add_insecure_port(f'[::]:{PORT}')
    server.start()
    print("server start.")
    server.wait_for_termination()