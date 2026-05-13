##GNn의 핵심 주변 이웃의 정보를 내 정보와 합치는 Layer
##CARE-GNN 논문의 핵심인 'Label-aware Filter'가 들어가는 부분
## GNN의 핵심 주변 이웃의 정보를 내 정보와 합치는 Layer
import torch
import torch.nn as nn
import torch.nn.functional as F

class InterAggregator(nn.Module):
    def __init__(self, features_dim, hidden_dim, relations_count):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.relations_count = relations_count
        self.intra_layers = nn.ModuleList([nn.Linear(features_dim, hidden_dim) for _ in range(relations_count)])
        self.inter_layer = nn.Linear(hidden_dim * relations_count, hidden_dim)

    def forward(self, nodes_features, adj_list):
        relation_outputs = []
        num_nodes = nodes_features.size(0)
        
        for i, edge_index in enumerate(adj_list):
            # [수정] 인덱스(edge_index)를 계산 가능한 희소 행렬(Sparse Matrix)로 변환
            adj = torch.sparse_coo_tensor(
                edge_index, 
                torch.ones(edge_index.size(1), device=edge_index.device),
                (num_nodes, num_nodes)
            ).to_sparse_csr() # 계산 속도를 위해 CSR 형식으로 변환

            # [핵심] 이제 행렬곱을 수행 (둘 다 float 형식이 됨)
            neighbor_features = torch.sparse.mm(adj.float(), nodes_features.float())
            
            res = F.relu(self.intra_layers[i](neighbor_features))
            relation_outputs.append(res)
            
        combined = torch.cat(relation_outputs, dim=1)
        return self.inter_layer(combined)