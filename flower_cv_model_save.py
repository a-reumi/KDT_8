from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import joblib
import pandas as pd
import utils
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score

#  CSV 로드
df = pd.read_csv(r"C:\Users\kdp\Desktop\KDT7\PROJECT\Project\data\흰색_final.csv")

#  색상 분류 라벨 생성
X_color = df[['H_avg', 'S_avg', 'V_avg', 'H_std', 'S_std', 'V_std']]
kmeans = KMeans(n_clusters=5, random_state=42)
df['color_cluster'] = kmeans.fit_predict(X_color)
cluster_labels = {
    0: '순백',
    1: '붉은기 흰색',
    2: '푸른기 흰색',
    3: '회백색',
    4: '베이지빛 흰색'
}

df['cluster_cluster'] = df['color_cluster'].map(cluster_labels)

#  피처 / 타겟 정의
features = ['R_avg', 'G_avg', 'B_avg', 'H_avg', 'S_avg', 'V_avg',
            'R_std', 'G_std', 'B_std', 'H_std', 'S_std', 'V_std']
X = df[features]
y = df['color_cluster']


# XGBoost

# 1. F1 macro 스코어 설정
f1_macro = make_scorer(f1_score, average='macro')

# 2. 하이퍼파라미터 후보 목록
xgb_param_grid = {
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0]
}

# 3. 모델 정의
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

# 4. GridSearchCV 수행
xgb_grid = GridSearchCV(xgb_model, xgb_param_grid, cv=3, scoring=f1_macro, n_jobs=-1, verbose=1)
xgb_grid.fit(X, y)

# 5. 최적 모델 추출
best_xgb = xgb_grid.best_estimator_

# 6. 교차검증 성능 평가
xgb_f1_scores = cross_val_score(best_xgb, X, y, cv=5, scoring=f1_macro)
print(f" XGBoost F1-score (macro) 평균: {xgb_f1_scores.mean():.4f}")

# 7. 모델 + 피처 컬럼 저장
joblib.dump({
    'model': best_xgb,
    'features': X.columns.tolist()
}, 'xgboost_model.pkl')

print("학습 시 사용된 feature 컬럼:")
print(X.columns.tolist())
print("모델 저장 완료: xgboost_model.pkl")