import torch
import torch.nn as nn
import numpy as np
import random
import os
from models.model import CAREGNN
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score

# 1. 수사 결과 저장용 리스트
all_seed_results = []

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
    print(f"\n로드된 데이터 구조 확인: {data}")
    x = data['x']
    y = data['y']
    adj_list = data['adj_list']
    train_idx = data['train_idx']
    test_idx = data['test_idx']
    
    # 모델 및 하이퍼파라미터 설정 (팀 공통 기준)
    model = CAREGNN(input_dim=x.size(1), hidden_dim=128, relations_count=len(adj_list))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # 사기꾼 검거를 위한 가중치 (기존 베스트였던 15 적용)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15.0]))

    # 학습 시작 (200 Epochs)
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        logits = model(x, adj_list)
        loss = criterion(logits[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()

    # 평가 모드
    model.eval()
    with torch.no_grad():
        logits = model(x, adj_list)
        test_logits = logits[test_idx]
        test_y = y[test_idx].cpu().numpy()
        
        # 지표 계산을 위한 변환
        probs = torch.sigmoid(test_logits).cpu().numpy()
        preds = (test_logits > 0).float().cpu().numpy() # Logit 0이 확률 0.5 지점

        # 결과 산출
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
    # 팀원분이 주신 5개 시드 파일 리스트
    data_files = [
        'data/graph_dat_1.pt',
        'data/graph_dat_42.pt',
        'data/graph_dat_57.pt',
        'data/graph_dat_123.pt',
        'data/graph_dat_777.pt'
    ]

    print("🚀 [CARE-GNN] 5개 시드 릴레이 수사를 시작합니다.")
    
    for f in data_files:
        seed_num = int(f.split('_')[-1].split('.')[0])
        set_seed(seed_num) # 파일 이름에 적힌 시드로 고정
        
        print(f"🔍 현재 수사 파일: {f} ...", end=" ", flush=True)
        res = run_investigation(f)
        all_seed_results.append(res)
        print(f"완료! (F1: {res['F1']:.4f})")

    # 최종 결과 평균 계산
    print("\n📊 === [최종 수사 보고서 요약] ===")
    for metric in all_seed_results[0].keys():
        values = [r[metric] for r in all_seed_results]
        print(f"{metric:10} : 평균 {np.mean(values):.4f} (±{np.std(values):.4f})")