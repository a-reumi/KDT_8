from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import joblib
import pandas as pd
import utils
from sklearn.preprocessing import LabelEncoder
import cv2
from sklearn.cluster import KMeans
import numpy as np

import cv2
import numpy as np
import pandas as pd

def extract_avg_std_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("이미지를 불러올 수 없습니다")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (100, 100))  # 선택사항

    # 가운데 50x50 crop (선택)
    h, w, _ = img.shape
    cx, cy = w // 2, h // 2
    crop = img[cy-25:cy+25, cx-25:cx+25]

    pixels = crop.reshape(-1, 3)  # (2500, 3)
    r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]

    # RGB 평균/표준편차
    R_avg, G_avg, B_avg = np.mean(r), np.mean(g), np.mean(b)
    R_std, G_std, B_std = np.std(r), np.std(g), np.std(b)

    # HSV 변환
    hsv_img = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    hsv_pixels = hsv_img.reshape(-1, 3)
    h, s, v = hsv_pixels[:, 0], hsv_pixels[:, 1], hsv_pixels[:, 2]
    H_avg, S_avg, V_avg = np.mean(h), np.mean(s), np.mean(v)
    H_std, S_std, V_std = np.std(h), np.std(s), np.std(v)

    # DataFrame 생성 (컬럼 순서 주의!)
    row = [R_avg, G_avg, B_avg, R_std, G_std, B_std,
           H_avg, S_avg, V_avg, H_std, S_std, V_std]
    
    columns = ['R_avg', 'G_avg', 'B_avg', 'R_std', 'G_std', 'B_std',
               'H_avg', 'S_avg', 'V_avg', 'H_std', 'S_std', 'V_std']

    df_input = pd.DataFrame([row], columns=columns)
    return df_input

# 모델 불러오기
bundle = joblib.load('decision_tree_model.pkl')
model = bundle['model']         # 모델
feature_columns = bundle['features']  # ['R_avg', ..., 'V_std']

# 이미지 경로
img_path = r"C:\Users\KDT-37\Desktop\KDT_7\09_ML_CV\Project\test.jpg"

# 전처리
df_input = extract_avg_std_features(img_path)

# 컬럼 정렬이 다를 경우 정렬
df_input = df_input[feature_columns]

# 예측
pred = model.predict(df_input)[0]
print(f"예측 결과: {pred}")