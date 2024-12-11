from torch.utils.data import DataLoader
import torch
import time
import numpy as np
import utils.utils as utils
import copy
from grand.federated_worker import *
def train_local_client_prox(clients, data_loader, epochs, client_id, port, mu, event):
    train_delay, transfer_delay, batch_size, model, optimizer, loss_fn, host, idx, rounds, device_id = clients[client_id]
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.train()
    global_model = copy.deepcopy(model)
    conn = gRPCClient("localhost", port)
    train_round = 0
    
    while event.is_set() == False:
        pass
    
    while True:
        train_round += 1
        for epoch in range(epochs):
            running_loss = 0.0
            
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                pred = model(images)
                loss = loss_fn(pred,labels)
                prox_term = 0
                
                # Proximal term (||w_t - w_global||^2)
                if mu > 0:
                    for param, global_param in zip(model.parameters(), global_model.parameters()):
                        prox_term += ((param.to(device) - global_param.to(device)) ** 2).sum()
                    prox_term = (mu / 2) * prox_term
                    
                loss = loss + prox_term
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
        print(f"Client {client_id} - iter [{train_round}], Loss: {running_loss/len(data_loader)}")
        state = conn.getGlobalModel()
        if state is not None and not torch.equal(global_model.state_dict()["conv1.weight"], state["conv1.weight"].to(device)):
            model.load_state_dict(state)
            global_model = copy.deepcopy(model)
            print(f"Client {client_id} global model received")
        else:
            conn.sendStates(states=model.state_dict(),setting="else",weights=b"1", drop=client_id) # 부모에게 학습된 모델 전송
            

def train_local_client_nova(clients, data_loader, epochs, client_id, port, event):
    train_delay, transfer_delay, batch_size, model, optimizer, loss_fn, host, idx, rounds, device_id = clients[client_id]
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.train()
    global_model = copy.deepcopy(model)
    train_round, tau = 0, 0
    conn = gRPCClient("localhost", port)
    
    while event.is_set() == False:
        pass
    
    while True:
        train_round += 1
        for epoch in range(epochs):
            running_loss = 0.0
            
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                pred = model(images)
                loss = loss_fn(pred,labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                tau += 1
                
        conn.sendStates(states=model.state_dict(), weights=tau,setting="fednova", drop=client_id) # 부모에게 학습된 모델 전송
        print(f"Client {client_id} - iter [{train_round}], Loss: {running_loss/len(data_loader)}, Steps: {tau}") 
        
        state = conn.getGlobalModel()
        if state is not None and not torch.equal(global_model.state_dict()["conv1.weight"], state["conv1.weight"].to(device)):
            model.load_state_dict(state)
            global_model = copy.deepcopy(model)
            print(f"Client {client_id} global model received")
        else:
            conn.sendStates(states=model.state_dict(),setting="else",weights=b"1", drop=client_id) # 부모에게 학습된 모델 전송