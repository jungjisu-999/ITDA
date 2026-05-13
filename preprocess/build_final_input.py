import pandas as pd
import numpy as np
import os

def merge_all_features():
    print("🔗 [수사반장 지수] 데이터 결합 작업을 시작합니다...")
    
    # 1. 정제된 데이터 로드 (우리가 앞 단계에서 만든 cleaned_meta.csv)
    df = pd.read_csv('data/cleaned_meta.csv')
    embeddings = np.load('data/refined_embeddings.npy')
    
    # 2. 모델 학습에 사용할 '수치형 컬럼' 선택
    # rating(평점)은 아주 중요한 피처입니다. 
    # 만약 팀원이 나중에 추가 변수를 주면 이 리스트에 이름만 더 추가하면 됩니다.
    feature_cols = ['rating'] 
    
    csv_features = df[feature_cols].values
    
    # 3. 결합 (Concatenate)
    # 텍스트 임베딩(384차원) 옆에 평점(1차원)을 붙여서 385차원으로 만듭니다.
    final_x = np.concatenate([embeddings, csv_features], axis=1)
    
    print(f"✅ 최종 피처(X) 행렬 생성 완료: {final_x.shape}") 
    # 예상 결과: (40616, 385)
    
    # 4. 라벨(Y) 및 ID 정보 추출
    y = df['label'].values
    user_ids = df['user_id'].values
    prod_ids = df['prod_id'].values
    
    # 5. 저장 (data 폴더)
    np.save('data/final_x.npy', final_x)
    np.save('data/final_y.npy', y)
    np.save('data/user_ids.npy', user_ids)
    np.save('data/prod_ids.npy', prod_ids)
    
    print("\n✨✨ [작업 완료] ✨✨")
    print("💾 final_x.npy (입력 피처)")
    print("💾 final_y.npy (정답 라벨)")
    print("💾 user_ids.npy / prod_ids.npy (그래프 연결용 ID)")
    print("이제 CARE-GNN 모델에 이 데이터들을 때려 넣기만 하면 됩니다!")

if __name__ == "__main__":
    if os.path.exists('data/cleaned_meta.csv'):
        merge_all_features()
    else:
        print("❌ 'data/cleaned_meta.csv'가 없습니다. 먼저 check_data.py를 실행하세요.")