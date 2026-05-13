import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def split_with_ratio(train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    print(f"✂️ 분할 비율 {train_ratio}:{val_ratio}:{test_ratio}로 작업을 시작합니다.")
    
    # 1. 라벨 데이터 로드
    y = np.load('data/final_y.npy')
    indices = np.arange(len(y))
    
    # 2. 1차 분할: Train과 나머지(Val + Test)
    train_idx, temp_idx = train_test_split(
        indices, test_size=(val_ratio + test_ratio), stratify=y, random_state=42
    )
    
    # 3. 2차 분할: Val과 Test
    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=(1 - relative_val_ratio), stratify=y[temp_idx], random_state=42
    )
    
    # 4. 저장
    np.save(f'data/train_idx_{int(train_ratio*10)}.npy', train_idx)
    np.save(f'data/val_idx_{int(train_ratio*10)}.npy', val_idx)
    np.save(f'data/test_idx_{int(train_ratio*10)}.npy', test_idx)
    
    print(f"✅ 저장 완료! (Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)})")

if __name__ == "__main__":
    # 7:1:2 실험용
    split_with_ratio(0.7, 0.1, 0.2)
    # 6:2:2 실험용
    split_with_ratio(0.6, 0.2, 0.2)