import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
import os

def build_custom_graph():
    print(" 메모리 최적화 모드로 커스텀 릴레이션 생성을 시작합니다.")

    # 1. 데이터 로드
    df = pd.read_csv('data/cleaned_meta.csv')
    embeddings = np.load('data/refined_embeddings.npy')
    num_nodes = len(df)

    # 릴레이션 생성용 최적화 함수
    def get_adj_optimized(dataframe, group_col):
        # 그룹별로 인덱스를 모음
        group_indices = dataframe.groupby(group_col).indices
        
        adj = sp.dok_matrix((num_nodes, num_nodes), dtype=np.float32)
        
        for name, indices in group_indices.items():
            if len(indices) > 1:
                # 너무 큰 그룹(예: 요일별 수만 개)은 엣지 수를 제한하거나 
                # 메모리 효율을 위해 유효한 연결만 생성
                if len(indices) > 500: # 한 그룹이 너무 크면 샘플링 (메모리 방어)
                    indices = np.random.choice(indices, 500, replace=False)
                
                for i in indices:
                    for j in indices:
                        if i != j:
                            adj[i, j] = 1.0
        return adj.tocoo()

    # --- [Relation 1: 유사도 0.8 이상] ---
    print("1/4: 유사도 기반 관계 생성 중... (청크 단위 계산)")
    # 메모리 방어를 위해 상위 유사도만 추출
    adj_content = sp.lil_matrix((num_nodes, num_nodes))
    for i in range(0, num_nodes, 1000): # 1000개씩 나눠서 계산
        end = min(i + 1000, num_nodes)
        sim_chunk = cosine_similarity(embeddings[i:end], embeddings)
        rows, cols = np.where(sim_chunk >= 0.8)
        for r, c in zip(rows, cols):
            if (i + r) != c:
                adj_content[i + r, c] = 1.0
    adj_content = adj_content.tocoo()

    # --- [Relation 2: 같은 가게 & 극단 별점] ---
    print("2/4: 같은 가게 내 극단 별점 관계 생성 중...")
    extreme_df = df[(df['rating'] == 1.0) | (df['rating'] == 5.0)]
    adj_extreme = get_adj_optimized(extreme_df, 'prod_id')

    # --- [Relation 3: 요일 패턴] ---
    print("3/4: 요일 패턴 관계 생성 중... (샘플링 적용)")
    df['date'] = pd.to_datetime(df['date'])
    df['weekday'] = df['date'].dt.weekday
    adj_pattern = get_adj_optimized(df, 'weekday')

    # --- [Relation 4: R-U-R (같은 유저)] ---
    print("4/4: 동일 작성자 관계 생성 중...")
    adj_rur = get_adj_optimized(df, 'user_id')

    # 2. 저장
    os.makedirs('data', exist_ok=True)
    sp.save_npz('data/adj_content.npz', adj_content)
    sp.save_npz('data/adj_extreme.npz', adj_extreme)
    sp.save_npz('data/adj_pattern.npz', adj_pattern)
    sp.save_npz('data/adj_rur.npz', adj_rur)

    print(f"\n✅ 4개 릴레이션 구축 완료 (메모리 세이프 버전)")
    print(f"🔗 Content: {adj_content.nnz} / Extreme: {adj_extreme.nnz} / Pattern: {adj_pattern.nnz} / RUR: {adj_rur.nnz}")

if __name__ == "__main__":
    build_custom_graph()