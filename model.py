import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

# 1. 데이터 로드
df = pd.read_csv(r"C:\Users\kdp\Desktop\KDT7\PROJECT\Project\data\흰색_final.csv")

X_color = df[['H_avg', 'S_avg', 'V_avg', 'H_std', 'S_std', 'V_std']]
kmeans = KMeans(n_clusters=5, random_state=42)
df['color_cluster'] = kmeans.fit_predict(X_color)


# print(df['color_cluster'].value_counts())

# 3. 특성과 타겟 분리
X = df[['R_avg', 'G_avg', 'B_avg', 
        'R_std', 'G_std', 'B_std', 
        'H_avg', 'S_avg', 'V_avg',
        'H_std', 'S_std', 'V_std']]
y = df['color_cluster']  # 흰색 여부를 타겟으로 설정

# 4. 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 5. SVM용 정규화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. 모델 정의
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42)
}

# 7. 학습 및 평가
print("흰색 vs 기타 분류 결과:\n")
for name, model in models.items():
    if name == "SVM":
        model.fit(X_train_scaled, y_train)
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

    train_score = accuracy_score(y_train, train_pred)
    test_score = accuracy_score(y_test, test_pred)

    print(f"{name}")
    print(f"  - Train Accuracy: {train_score:.4f}")
    print(f"  - Test Accuracy:  {test_score:.4f}\n") 