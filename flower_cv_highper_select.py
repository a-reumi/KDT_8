from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import joblib
import pandas as pd
import utils
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics import make_scorer, f1_score

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

# 1-1. LightGBM

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# 공통 F1-score macro 평가 지표
f1_macro = make_scorer(f1_score, average='macro')

# ========== 1. XGBoost 튜닝 ==========
xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0]
}

xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_grid = GridSearchCV(xgb, xgb_param_grid, cv=3, scoring=f1_macro, n_jobs=-1, verbose=1)
xgb_grid.fit(X, y)
best_xgb = xgb_grid.best_estimator_

# 교차검증 평가
xgb_acc = cross_val_score(best_xgb, X, y, cv=5, scoring='accuracy')
xgb_f1 = cross_val_score(best_xgb, X, y, cv=5, scoring=f1_macro)

# ========== 2. LightGBM 튜닝 ==========
lgbm_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0]
}

lgbm = LGBMClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
lgbm_grid = GridSearchCV(lgbm, lgbm_param_grid, cv=3, scoring=f1_macro, n_jobs=-1, verbose=1)
lgbm_grid.fit(X, y)
best_lgbm = lgbm_grid.best_estimator_

# 교차검증 평가
lgbm_acc = cross_val_score(best_lgbm, X, y, cv=5, scoring='accuracy')
lgbm_f1 = cross_val_score(best_lgbm, X, y, cv=5, scoring=f1_macro)

# ========== 결과 출력 ==========
print("튜닝 후 성능 비교")

print(f" \n XGBoost")
print(f"  최적 파라미터: {xgb_grid.best_params_}")
print(f"  정확도 평균: {xgb_acc.mean():.4f}, 표준편차: {xgb_acc.std():.4f}")
print(f"   F1-score(macro) 평균: {xgb_f1.mean():.4f}, 표준편차: {xgb_f1.std():.4f}")

print(f"\n LightGBM")
print(f" 최적 파라미터: {lgbm_grid.best_params_}")
print(f" 정확도 평균: {lgbm_acc.mean():.4f}, 표준편차: {lgbm_acc.std():.4f}")
print(f" F1-score(macro) 평균: {lgbm_f1.mean():.4f}, 표준편차: {lgbm_f1.std():.4f}")


# # 모델 정의
# lgbm_model = LGBMClassifier(use_label_encoder=False, eval_metric='mlogloss')

# # F1 Macro Scorer 정의
# f1_macro = make_scorer(f1_score, average='macro')

# # 1. F1-macro 기준 교차검증
# f1_scores = cross_val_score(lgbm_model, X, y, cv=5, scoring=f1_macro)

# # 2. 정확도 기준 교차검증
# acc_scores = cross_val_score(lgbm_model, X, y, cv=5)

# # 결과 출력
# print("LightGBM 모델 5-Fold 교차검증 결과")
# print(f"정확도 평균: {acc_scores.mean():.4f}")
# print(f"정확도 표준편차: {acc_scores.std():.4f}")
# print(f"F1-score (macro) 평균: {f1_scores.mean():.4f}")
# print(f"F1-score (macro) 표준편차: {f1_scores.std():.4f}")

# # 1-2. XGBoost

# # 모델 정의
# xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# # F1 Macro Scorer 정의
# f1_macro = make_scorer(f1_score, average='macro')

# # 1. F1-macro 기준 교차검증
# f1_scores = cross_val_score(xgb_model, X, y, cv=5, scoring=f1_macro)

# # 2. 정확도 기준 교차검증
# acc_scores = cross_val_score(xgb_model, X, y, cv=5)

# # 결과 출력
# print("XGBoost 모델 5-Fold 교차검증 결과")
# print(f"정확도 평균: {acc_scores.mean():.4f}")
# print(f"정확도 표준편차: {acc_scores.std():.4f}")
# print(f"F1-score (macro) 평균: {f1_scores.mean():.4f}")
# print(f"F1-score (macro) 표준편차: {f1_scores.std():.4f}")


# lgbm_model.fit(X, y)
# joblib.dump({
#     'model': lgbm_model,
#     'features': features  # 정확한 컬럼 순서 리스트
# }, 'lightgbm_model.pkl')

# xgb_model.fit(X, y)
# joblib.dump({
#     'model': xgb_model,
#     'features': features  # 정확한 컬럼 순서 리스트
# }, 'xgboost_model.pkl')

# print("학습 시 사용된 feature 컬럼:")
# print(X.columns.tolist())

# print("모델 저장 완료 (lightgbm_model.pkl, xgboost_model.pkl)")