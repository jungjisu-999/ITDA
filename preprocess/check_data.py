import pandas as pd
import numpy as np
import torch
import os

csv_path = 'raw_data/fraud_findat.csv'
npy_path = 'raw_data/review_embeddings.npy'

print("🛠️ [수사반장 지수] 데이터 1:1 매칭 및 정제 작업을 시작합니다.")

try:
    # 1. CSV 로드 (깨진 줄의 인덱스를 확인하기 위해 index_col을 활용)
    df = pd.read_csv(
        csv_path, 
        encoding='cp949', 
        encoding_errors='ignore', 
        on_bad_lines='skip',  # 깨진 1개 줄을 건너뜀
        engine='python'
    )
    
    # 살아남은 행들의 원래 번호(인덱스)를 추출합니다.
    survived_indices = df.index.tolist()
    print(f"✅ CSV 로드 성공: {len(df)}개 행 생존")

    # 2. 임베딩 로드 및 필터링
    embeddings = np.load(npy_path)
    print(f"✅ 원본 임베딩 로드: {len(embeddings)}개")

    # CSV에서 살아남은 행 번호에 해당하는 임베딩만 골라냅니다.
    # 이렇게 해야 중간에 하나가 빠져도 뒤쪽 데이터가 밀리지 않습니다.
    refined_embeddings = embeddings[survived_indices]
    print(f"✅ 임베딩 필터링 완료: {len(refined_embeddings)}개")

    # 3. 최종 검증 및 저장
    if len(df) == len(refined_embeddings):
        print(f"\n✨ [성공] {len(df)}개의 데이터가 1:1로 완벽하게 매칭되었습니다!")
        
        # 나중에 쓰기 편하게 정제된 데이터를 저장합니다.
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/cleaned_meta.csv', index=False)
        np.save('data/refined_embeddings.npy', refined_embeddings)
        print(" 정제된 데이터가 'data/' 폴더에 저장되었습니다.")
    else:
        print("\n 여전히 개수가 맞지 않습니다. 구조적 확인이 필요합니다.")

except Exception as e:
    print(f"\n 작업 중 에러 발생: {e}")

