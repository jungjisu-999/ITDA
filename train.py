import torch
import torch.nn as nn
import numpy as np
import random
import os
from models import CAREGNN
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_investigation(file_path):
    # 데이터 로드
    data = torch.load(file_path, weights_only=False)
    
    x = data.x.float()
    y = data.y.float() # BCE Loss를 위해 float 변환
    
    # [수정] PyG 데이터를 CARE-GNN용 adj_list로 변환
    edge_index = data.edge_index
    edge_type = data.edge_type
    
    adj_list = []
    for i in range(int(edge_type.max()) + 1):
        adj_list.append(edge_index[:, edge_type == i]) 
    
    # [수정] 데이터셋에 들어있는 마스크 활용 (없으면 에러나서 보완함)
    train_idx = torch.where(data.train_mask)[0]
    test_idx = torch.where(data.test_mask)[0]

    # 모델 초기화
    model = CAREGNN(input_dim=x.size(1), hidden_dim=128, relations_count=len(adj_list))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 사기꾼 검거 가중치 적용
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15.0]))

    # 학습 루프
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        logits = model(x, adj_list)
        # 차원 맞추기 (Squeeze)
        loss = criterion(logits[train_idx].squeeze(), y[train_idx])
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f" Epoch {epoch+1}/200 | Loss: {loss.item():.4f}", end="")

    # 평가
    model.eval()
    with torch.no_grad():
        logits = model(x, adj_list).squeeze()
        test_logits = logits[test_idx]
        test_y = y[test_idx].cpu().numpy()
        
        probs = torch.sigmoid(test_logits).cpu().numpy()
        preds = (probs > 0.5).astype(float)

        metrics = {
            "F1": f1_score(test_y, preds),
            "Macro-F1": f1_score(test_y, preds, average='macro'),
            "Accuracy": accuracy_score(test_y, preds),
            "Precision": precision_score(test_y, preds),
            "Recall": recall_score(test_y, preds),
            "AUC-ROC": roc_auc_score(test_y, probs),
            "PR-AUC": average_precision_score(test_y, probs)
        }
    return metrics

if __name__ == "__main__":
    data_files = [
        'data/graph_dat_rgcn_1.pt',
        'data/graph_dat_rgcn_42.pt',
        'data/graph_dat_rgcn_57.pt',
        'data/graph_dat_rgcn_123.pt',
        'data/graph_dat_rgcn_777.pt'
    ]

    all_seed_results = []
    print(" [CARE-GNN] 5개 시드 릴레이 수사를 시작합니다.")
    
    for f in data_files:
        # 파일이 있는지 확인
        if not os.path.exists(f):
            print(f" 파일 없음: {f}")
            continue
            
        seed_num = int(f.split('_')[-1].split('.')[0])
        set_seed(seed_num)
        
        print(f"\n🔍 현재 수사 파일: {f}")
        res = run_investigation(f)
        all_seed_results.append(res)
        print(f" -> 완료! (F1: {res['F1']:.4f})")

    if all_seed_results:
        print("\n === [최종 수사 보고서 요약] ===")
        for metric in all_seed_results[0].keys():
            values = [r[metric] for r in all_seed_results]
            print(f"{metric:10} : 평균 {np.mean(values):.4f} (±{np.std(values):.4f})")