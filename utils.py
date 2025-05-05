import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def remove_background(image):
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    height, width = image.shape[:2]
    rect = (10, 10, width - 20, height - 20)
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    result = image * mask2[:, :, np.newaxis]
    return result

def extract_dominant_colors(img, k=5):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (100, 100))
    pixels = img.reshape(-1, 3)
    pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]
    if pixels.shape[0] < k:
        return []

    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_.astype(int)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    proportions = counts / counts.sum()

    return list(zip(colors, proportions))

def classify_color_group(rgb_color):
    color = np.uint8([[rgb_color]])
    hsv = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)[0][0]
    h = hsv[0]
    if h < 15 or h >= 165:
        return 'Red'
    elif 15 <= h < 35:
        return 'Orange/Yellow'
    elif 35 <= h < 85:
        return 'Green'
    elif 85 <= h < 135:
        return 'Blue'
    elif 135 <= h < 165:
        return 'Purple'
    else:
        return 'Other'

def plot_colors(colors, proportions, filename):
    plt.figure(figsize=(6, 1))
    start = 0
    for color, prop in zip(colors, proportions):
        plt.fill_between([start, start + prop], 0, 1, color=np.array(color / 255))
        start += prop
    plt.xlim(0, 1)
    plt.axis('off')
    plt.title(filename, fontsize=10)
    plt.tight_layout()
    plt.show()
    
def classify_color_group_from_h(h):
    if h < 15 or h >= 150:
        return 'Red'
    elif 15 <= h < 35:
        return 'Orange/Yellow'
    elif 35 <= h < 85:
        return 'Green'
    elif 85 <= h < 110:
        return 'Blue'
    elif 110 <= h < 151:
        return 'Purple'
    else:
        return 'Other'
    
def compute_color_statistics(dominant_colors):
    rgb_values = np.array([color for color, _ in dominant_colors])
    hsv_values = np.array([rgb_to_hsv(*color) for color, _ in dominant_colors])

    stats = {
        'R_avg': np.mean(rgb_values[:, 0]),
        'G_avg': np.mean(rgb_values[:, 1]),
        'B_avg': np.mean(rgb_values[:, 2]),
        'R_std': np.std(rgb_values[:, 0]),
        'G_std': np.std(rgb_values[:, 1]),
        'B_std': np.std(rgb_values[:, 2]),
        'H_avg': np.mean(hsv_values[:, 0]),
        'S_avg': np.mean(hsv_values[:, 1]),
        'V_avg': np.mean(hsv_values[:, 2]),
        'H_std': np.std(hsv_values[:, 0]),
        'S_std': np.std(hsv_values[:, 1]),
        'V_std': np.std(hsv_values[:, 2]),
    }

    return stats

import cv2
import numpy as np

def rgb_to_hsv(r, g, b):
    color = np.uint8([[[r, g, b]]])  # 1픽셀짜리 RGB 이미지 만들기
    hsv = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)  # OpenCV로 변환
    return hsv[0][0]  # H, S, V 값 반환

def predict_from_image(image_path, model, feature_columns):
    img = cv2.imread(image_path)
    if img is None:
        return "이미지 로드 실패"

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (100, 100))

    # 가운데 50x50 crop
    h, w, _ = img.shape
    cx, cy = w // 2, h // 2
    crop_size = 50
    x1, y1 = cx - crop_size // 2, cy - crop_size // 2
    x2, y2 = cx + crop_size // 2, cy + crop_size // 2
    img_crop = img[y1:y2, x1:x2]

    pixels = img_crop.reshape(-1, 3)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)

    rgb_list = []
    hsv_list = []
    row = []

    for color in colors:
        r, g, b = color
        rgb_list.append([r, g, b])

        rgb_pixel = np.uint8([[[r, g, b]]])
        hsv_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2HSV)[0][0]
        h, s, v = hsv_pixel
        hsv_list.append([h, s, v])

        row += [r, g, b, h, s, v]

    rgb_mean = np.mean(rgb_list, axis=0)
    hsv_mean = np.mean(hsv_list, axis=0)
    rgb_std = np.std(rgb_list, axis=0)
    hsv_std = np.std(hsv_list, axis=0)

    row += rgb_mean.tolist() + hsv_mean.tolist()
    row += rgb_std.tolist() + hsv_std.tolist()

    # DataFrame 생성 (컬럼명 일치 필수!)
    df_input = pd.DataFrame([row], columns=feature_columns)

    # 예측
    pred = model.predict(df_input)[0]
    return pred

def extract_hsv_stats_opencv(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    h_mean, s_mean, v_mean = h.mean(), s.mean() * 100 / 255, v.mean() * 100 / 255
    h_std, s_std, v_std = h.std(), s.std() * 100 / 255, v.std() * 100 / 255

    return [h_mean, s_mean, v_mean, h_std, s_std, v_std]

def extract_avg_std_features_df(image_path):
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지를 불러올 수 없습니다")

    # 가운데 50x50 crop
    h, w = img.shape[:2]
    x1, y1 = int(w * 0.3), int(h * 0.3)
    x2, y2 = int(w * 0.7), int(h * 0.7)
    cropped_img = img[y1:y2, x1:x2]

    # RGB 변환 및 리사이즈
    cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    cropped_img = cv2.resize(cropped_img, (100, 100))
    pixels = cropped_img.reshape(-1, 3)

    # RGB 평균/표준편차
    rgb_mean = pixels.mean(axis=0)
    rgb_std = pixels.std(axis=0)

    # HSV 평균/표준편차
    hsv_stats = extract_hsv_stats_opencv(cropped_img)  # [H_avg, S_avg, V_avg, H_std, S_std, V_std]

    # 컬럼명 정의
    columns = ['R_avg', 'G_avg', 'B_avg', 'R_std', 'G_std', 'B_std',
               'H_avg', 'S_avg', 'V_avg', 'H_std', 'S_std', 'V_std']

    # DataFrame 생성 (1행짜리)
    df = pd.DataFrame([rgb_mean.tolist() + rgb_std.tolist() + hsv_stats], columns=columns)
    
    return df

