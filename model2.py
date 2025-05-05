import os
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# 사용자 맞춤 경로 설정
base_dir = r"C:\Users\kdp\Desktop\KDT7\PROJECT\Project\data\white"
output_path = r"C:\Users\kdp\Desktop\KDT7\PROJECT\Project\data\흰색_final.csv"

data_rows = []

# ✅ HSV 평균 + 표준편차 추출 함수 (OpenCV 방식)
def extract_hsv_stats_opencv(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    
    h_mean, s_mean, v_mean = h.mean(), s.mean() * 100 / 255, v.mean() * 100 / 255
    h_std, s_std, v_std = h.std(), s.std() * 100 / 255, v.std() * 100 / 255

    return [h_mean, s_mean, v_mean, h_std, s_std, v_std]

# 전체 이미지 순회
for flower_name in os.listdir(base_dir):
    flower_path = os.path.join(base_dir, flower_name)
    if not os.path.isdir(flower_path):
        continue

    for img_name in os.listdir(flower_path):
        img_path = os.path.join(flower_path, img_name)

        try:
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue

            # 중앙 Crop
            h, w = img.shape[:2]
            x1, y1 = int(w * 0.3), int(h * 0.3)
            x2, y2 = int(w * 0.7), int(h * 0.7)
            cropped_img = img[y1:y2, x1:x2]

            # RGB 변환 및 리사이즈
            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            cropped_img = cv2.resize(cropped_img, (100, 100))
            pixels = cropped_img.reshape(-1, 3)

            # RGB 평균 및 표준편차
            rgb_mean = pixels.mean(axis=0)
            rgb_std = pixels.std(axis=0)

            # ✅ HSV 평균 + 표준편차
            hsv_stats = extract_hsv_stats_opencv(cropped_img)

            # KMeans 대표색 3개 추출
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans.fit(pixels)
            kmeans_colors = kmeans.cluster_centers_.astype(int).flatten()

            # 결과 저장
            row = [flower_name, img_name] + rgb_mean.tolist() + rgb_std.tolist() + hsv_stats + kmeans_colors.tolist()
            data_rows.append(row)

        except Exception as e:
            print(f" 오류 발생: {img_path} → {e}")

# ✅ 컬럼명 정의 (HSV std 포함)
columns = (
    ["label", "filename"] +
    ["R_avg", "G_avg", "B_avg"] +
    ["R_std", "G_std", "B_std"] +
    ["H_avg", "S_avg", "V_avg", "H_std", "S_std", "V_std"] +
    [f"{c}{i+1}" for i in range(3) for c in ["R", "G", "B"]]
)

# DataFrame 저장
df = pd.DataFrame(data_rows, columns=columns)
df.to_csv(output_path, index=False)

output_path

