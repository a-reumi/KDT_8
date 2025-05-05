import os
import cv2
import numpy as np
import pandas as pd
import utils

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

#  CSV 로드
df = pd.read_csv(r"C:\Users\kdp\Desktop\KDT7\PROJECT\Project\data\흰색_final.csv")

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


#  train/test 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#  SVM용 정규화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#  여러 모델 정의

models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42)
}

#  Confusion Matrix 시각화 함수
def plot_confusion(model_name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

#  모델 훈련 및 평가
for name, model in models.items():
    print(f"\n Model: {name}")
    if name == "SVM":
        model.fit(X_train_scaled, y_train)
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

    print("Train Classification Report")
    print(classification_report(y_train, y_pred_train))
    print("Test Classification Report")
    print(classification_report(y_test, y_pred_test))
    
    # plot_confusion(name, y_test, y_pred_test)