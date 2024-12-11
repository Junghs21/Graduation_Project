import torch
from sklearn.metrics import silhouette_score
import numpy as np
import math
import utils.utils as util
from collections import OrderedDict

# 합칠 때 클러스터의 data distribution도 고려할 것
def make_clusters(clusters, states, steps, min_cluster):
    cluster_score_buffer = []
    cluster_idx_buffer = []
    state_buffer = [states]
    sil_scores = []
    
    while len(clusters) >= 2:
        score_arr = calc_scores(clusters, states) # 각 클러스터 간 유사도 점수 계산 
        cluster_score_buffer.append(score_arr)
        buffer = []
        visited = [False for _ in range(len(clusters))]
        
        for idx1, cluster1 in enumerate(clusters):
            if visited[idx1]: continue
            selected_idx, best_score = idx1, -1000
            for idx2, cluster2 in enumerate(clusters):
                if cluster1 == cluster2 or visited[idx2] or visited[idx1]: continue
                
                if best_score < score_arr[idx1][idx2]: # cluster1 & cluster2의 유사도 점수
                    best_score = score_arr[idx1][idx2]
                    selected = cluster2
                    selected_idx = idx2
            
            if selected_idx != idx1:
                visited[idx1] = True
                visited[selected_idx] = True
                buffer.append([cluster1, selected])
                
            elif visited[idx1] == False:
                buffer.append([cluster1])
                visited[idx1] = True
                
        cluster_idx_buffer.append(buffer)
        
        # 선택된 클러스터 aggregation
        new_states = {}
        for idx, cluster in enumerate(buffer):
            if len(cluster) > 1:
                new_states[idx] = util.average_weights([states[cluster[0]], states[cluster[1]]], [1, 1])
                steps[cluster[0]] = 1; steps[cluster[1]] = 1
            else:
                new_states[idx] = states[cluster[0]]
        
        states = new_states
        state_buffer.append(list(states.values()))
        clusters = states.keys()
        
    for pair in zip(cluster_score_buffer, cluster_idx_buffer):
        try:
            score, indices = pair
            if len(score) == 2: break
            label = [0 for _ in range(len(score))]
            
            for i, clust in enumerate(indices):
                for x in clust:
                    label[x] = i
                    
            sil_scores.append(silhouette_score(np.array(score), label, metric="euclidean"))
            
        except Exception as e:
            print(f"Error calculating silhouette score: {e}")
                
    best_idx = torch.argmax(torch.tensor(sil_scores))
    cluster_label = cluster_idx_buffer[best_idx]
    for idx in range(best_idx-1, -1, -1):
        x = []
        search_buffer = cluster_idx_buffer[idx]
        for indices in cluster_label:
            temp = []
            for indice in indices:
                temp.extend(search_buffer[indice])
            x.append(temp)
        cluster_label = x
    
    return state_buffer[best_idx+1], cluster_label, sil_scores[best_idx] 

def calc_scores(devices, states):
    score_arr = [[0 for _ in range(len(devices))] for _ in range(len(devices))]
    for k1, k2 in enumerate(devices):
        for j1, j2 in enumerate(devices):
            if k1 < j1: 
                break
            param1 = states[k2]; param2 = states[j2]
            C, L1 = calculate_similarity_and_distance(param1, param2)
            score = (1 + C) / (1 + L1)
            if k1 == j1: score = torch.tensor(0)
            score_arr[k1][j1] = score.item()
            score_arr[j1][k1] = score.item()
    
    return np.array(score_arr)

# 두 모델의 파라미터 리스트 or 레이블 벡터를 입력으로 받습니다.
def calculate_similarity_and_distance(params1, params2):
    # 파라미터 텐서를 직렬화하여 하나의 벡터로 만듭니다.
    if isinstance(params1, OrderedDict):
        tensors1 = list(params1.values())  # OrderedDict에서 텐서 값 추출
        tensors2 = list(params2.values())
    else:
        tensors1 = torch.tensor(params1, dtype=torch.float32)
        tensors2 = torch.tensor(params2, dtype=torch.float32)
    
    # Flattening
    tensors1 = torch.cat([p.view(-1) for p in tensors1]).cpu()
    tensors2 = torch.cat([p.view(-1) for p in tensors2]).cpu()
    
    dot_product = torch.dot(tensors1, tensors2)
    
    norm_1 = torch.norm(tensors1, p=2)
    norm_2 = torch.norm(tensors2, p=2)
    
    cosine_similarity = dot_product / (norm_1 * norm_2)

    # L1 거리 계산
    vector1_norm = tensors1 / norm_1
    vector2_norm = tensors2 / norm_2
    l1_distance = torch.norm(vector1_norm - vector2_norm, p=1).item()
    return cosine_similarity, l1_distance

def weighted_sample_without_replacement(clusters, entropies, weights, k):
    # 가중치 정규화
    weights = np.array(weights)
    normalized_weights = weights / weights.sum()
    
    # 합이 1이 아닌 경우 마지막 원소를 조정
    if not np.isclose(normalized_weights.sum(), 1):
        normalized_weights[-1] += 1 - normalized_weights.sum()
        
    # 인덱스를 비복원 방식으로 선택
    indices = np.arange(len(clusters))
    selected_indices = np.random.choice(indices, size=k, replace=False, p=normalized_weights)
    
    # 선택된 클러스터 반환
    selected_clusters = [clusters[i] for i in selected_indices]
    selected_entropies = [entropies[i] for i in selected_indices]
    
    if len(selected_clusters) < 2:
        selected_clusters = clusters
        selected_entropies = entropies
        
    return selected_clusters, selected_entropies

def entropy(probabilities):
    return -sum(p * math.log2(p) for p in probabilities if p > 0)

def selectCluster(labels, n_class, clusters, select_len):
    cluster_entropies = calculateEntropy(labels, n_class, clusters)
        
    # 엔트로피에 따라 가중치를 정규화하여 선택 확률 부여
    total_entropy = sum(cluster_entropies)
    selection_probabilities = [entropy / total_entropy for entropy in cluster_entropies]
    # 클러스터 선택
    return weighted_sample_without_replacement(clusters, cluster_entropies, selection_probabilities, select_len)

def calculateEntropy(labels, n_class, clusters):
    cluster_datas ,cluster_entropies = [], []
    for cluster in clusters:
        for device in cluster:
            label_counts = [0 for _ in range(n_class)]
            label_counts += labels[device]
        
        # 확률 계산
        total_labels = sum(label_counts)
        label_probabilities = label_counts / total_labels
        
        # 엔트로피 계산
        cluster_entropy = entropy(label_probabilities)
        cluster_entropies.append(cluster_entropy)
        cluster_datas.append(label_counts)
        
    return cluster_entropies, cluster_datas