import streamlit as st
import joblib
from PIL import Image
import numpy as np
import pandas as pd
import utils
import cv2
import random

color_content_map = {
    0: {
        'emotion': '순수, 평온, 정결',
        'places': [
    "제주도 삼나무숲길의 하얀 수국길",
    "강릉 경포대 순백 매화길",
    "충남 아산 외암민속마을 백목련 거리",
    "경기도 양평 세미원 연꽃정원",
    "서울 종로 창덕궁 후원 흰 매화길",
    "전남 구례 화엄사 흰 벚꽃길",
    "강원도 인제 원대리 자작나무 숲",
    "일본 교토 긴카쿠지(은각사)의 백사 정원",
    "스위스 루체른 눈꽃길",
    "독일 바이에른 알프스 설화 트레일"
],
        'quotes':  [
        "당신은 고요한 빛처럼 잔잔해요.",
        "말하지 않아도 따뜻함이 전해져요.",
        "맑은 공기처럼 마음을 씻어주는 사람이에요.",
        "흰 꽃잎처럼 조용히 피어나는 존재.",
        "당신의 순수함은 누구보다 눈부셔요.",
        "소란 속에서도 평화를 품은 사람.",
        "가장 고운 색은 아무 색도 섞이지 않은 마음이에요.",
        "당신을 보면 마음이 맑아져요.",
        "흰 눈처럼 깨끗한 오늘이 되기를.",
        "차분함 속에 숨은 깊은 아름다움, 당신이에요."
    ],
       'music': [
    "https://www.youtube.com/watch?v=F1B9Fk_SgI0",  # Lauv - I Like Me Better  
    "https://www.youtube.com/watch?v=JkK8g6FMEXE",  # Norah Jones - Don't Know Why  
    "https://www.youtube.com/watch?v=E07s5ZYygMg",  # Coldplay - Sparks  
    "https://www.youtube.com/watch?v=7sXH2Hfz4jA",  # Billie Eilish - come out and play  
    "https://www.youtube.com/watch?v=4Tr0otuiQuU"   # Yiruma - River Flows in You  
]

    },

    1: {
        'emotion': '따뜻함, 사랑스러움, 열정',
        'places': [
    "경남 통영 동피랑 벽화마을",
    "경주 교촌한옥마을 붉은 배롱나무길",
    "전남 담양 메타프로방스 핑크 뮬리밭",
    "서울 북촌 감성 골목",
    "춘천 제이드가든 핑크 수국길",
    "제주 표선 붉은 동백꽃길",
    "프랑스 프로방스 붉은 포도밭",
    "이탈리아 피렌체 올리브 언덕",
    "스페인 안달루시아 석양 거리",
    "일본 오키나와 분홍빛 꽃거리"
]
,
        'quotes': [
    "당신 안에 따뜻한 계절이 있어요.",
    "햇살처럼 마음을 데워주는 사람이에요.",
    "사랑스러운 건 당신의 기본값이에요.",
    "하루가 붉게 물들도록 당신이 피어나요.",
    "마음이 포근해지는 건 당신 때문이에요.",
    "열정은 조용히 피어나는 붉은 꽃 같아요.",
    "당신은 봄보다 부드럽고 따뜻해요.",
    "어떤 말보다 당신의 미소가 깊어요.",
    "오늘도 당신 덕분에 마음이 따뜻해졌어요.",
    "사랑은 말없이 피어나는 감정이에요. 당신처럼."
]
,
        'music': [
    "https://www.youtube.com/watch?v=8xg3vE8Ie_E",  # Taylor Swift - Love Story
    "https://www.youtube.com/watch?v=IcrbM1l_BoI",  # Ed Sheeran - Thinking Out Loud
    "https://www.youtube.com/watch?v=450p7goxZqg",  # Sam Smith - Stay With Me
    "https://www.youtube.com/watch?v=PIh2xe4jnpk",  # James Arthur - Say You Won't Let Go
    "https://www.youtube.com/watch?v=TUVcZfQe-Kw"   # Olivia Rodrigo - drivers license
]

    },

2: {
    'emotion': '청량함, 고요함, 지적임',
    'places': [
        "일본 후라노 라벤더 언덕", "제주 보랏빛 수국길", "영국 히친 라벤더 농장",
        "경북 의성 라벤더밭", "프랑스 프로방스 라벤더 들판", "서울 보라빛 수국정원",
        "스페인 안달루시아 라벤더 마을", "강릉 경포 라벤더 정원", "평창 보랏빛 산책길", "충남 태안 보라빛 해안도로"
    ],
    'quotes': [
    "스위스 루체른 푸른 호수길",
    "강릉 주문진 해변",
    "제주 김녕 해안도로",
    "크로아티아 플리트비체 호수",
    "일본 하코네 온천마을의 푸른 정원",
    "남해 다랭이 마을 푸른 바다길",
    "노르웨이 피오르드 블루 트레일",
    "속초 영금정 해안길",
    "대관령 양떼목장 흐린 날 산책로",
    "서울 북서울 꿈의숲 고요한 메타세쿼이아 길"
]
,
    'music': [
    "https://www.youtube.com/watch?v=J6zwLKVUNiE",  # Cigarettes After Sex - Apocalypse
    "https://www.youtube.com/watch?v=0J2QdDbelmY",  # Muse - Starlight
    "https://www.youtube.com/watch?v=DWcJFNfaw9c",  # HONNE - Day 1 ◑
    "https://www.youtube.com/watch?v=AX8-YzMKZhQ",  # Billie Eilish - ocean eyes
    "https://www.youtube.com/watch?v=EBtBqpi6uBM"   # Tom Misch - Movie
]


},

3: {
    'emotion': '절제, 성찰, 도회적 감성',
    'places': [
    "서울 북촌 한옥마을 겨울 골목",
    "도쿄 롯폰기 모리타워 전망대",
    "파리 오르세 미술관 회색 복도",
    "런던 테이트모던 회색 벽화 앞",
    "부산 영화의전당 건축 산책로",
    "서울 DDP 건축 외벽 산책길",
    "베를린 박물관섬 회색 건물들",
    "순천 국가정원 수변 데크길 흐린 날",
    "브루클린 브릿지 이른 아침",
    "강릉 안목해변 흐린 날의 카페 거리"
]
,
    'quotes': [
    "당신은 조용히 자신을 말하는 사람이에요.",
    "말보다 깊은 마음이 느껴지는 순간이에요.",
    "빛나지 않아도 충분히 아름다운 존재.",
    "차분함은 당신의 가장 큰 매력이에요.",
    "회색은 색이 없는 게 아니라 모든 색을 품은 거예요.",
    "어디에도 휘둘리지 않는 중심 같은 사람.",
    "느릿한 감정 속에 깊은 진심이 있어요.",
    "세상이 흘러도 당신은 흔들림 없어요.",
    "눈에 띄지 않아도 남는 건 결국 진정성이에요.",
    "감정의 색을 숨긴 당신만의 우아함이 있어요."
]
,
    'music': [
    "https://www.youtube.com/watch?v=SBjQ9tuuTJQ",  # James Blake - Retrograde
    "https://www.youtube.com/watch?v=3YxaaGgTQYM",  # Evanescence - My Immortal
    "https://www.youtube.com/watch?v=qeMFqkcPYcg",  # Sia - Breathe Me
    "https://www.youtube.com/watch?v=7gqjzJiO1FY",  # keshi - like i need u
    "https://www.youtube.com/watch?v=I0MT8SwNa_U"   # H.E.R. - Hard Place
]

},

4: {
    'emotion': '편안함, 안정, 따뜻한 여유',
    'places': [
    "그리스 산토리니 바닷가",
    "제주 하늘오름길",
    "스위스 인터라켄 호수",
    "부산 해운대 바다 산책로",
    "시드니 본다이 비치",
    "괌 투몬비치",
    "태국 피피섬",
    "몰디브 바닷길",
    "강릉 안목 해변",
    "울산 대왕암 해안길"
]
,
    'quotes': [
    "당신은 바다처럼 깊고 잔잔해요.",
    "말 없는 위로가 되는 존재, 그게 당신이에요.",
    "고요함은 힘이 있다는 걸 느껴요.",
    "푸른 감정이 오늘을 감싸 안아주네요.",
    "바람보다 잔잔한 마음을 가졌군요.",
    "믿음을 주는 눈빛, 베이지처럼 부드러워요.",
    "당신은 언제나 안정감을 주는 사람이에요.",
    "햇살 아래 펼쳐진 여유로움이 닮았어요.",
    "고요한 순간에 진짜 감정이 흐르죠.",
    "흔들리지 않는 마음, 그 안에 따뜻함이 있어요."
]
,
    'music': [
    "https://www.youtube.com/watch?v=BiQIc7fG9pA",  # Christina Perri - A Thousand Years
    "https://www.youtube.com/watch?v=8xg3vE8Ie_E",  # Taylor Swift - Love Story
    "https://www.youtube.com/watch?v=RgKAFK5djSk",  # Wiz Khalifa - See You Again
    "https://www.youtube.com/watch?v=hLQl3WQQoQ0",  # Adele - Someone Like You
    "https://www.youtube.com/watch?v=AJtDXIazrMo"   # Enrique Iglesias - Bailando
]


}
}

# Streamlit 시작
st.title(" 꽃 이미지 감성 분류기 + 추천 콘텐츠")

uploaded_file = st.file_uploader("이미지를 업로드하세요", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='업로드된 이미지', use_container_width=True)

    # 이미지 저장 후 전처리
    temp_path = "temp_uploaded.jpg"
    image.save(temp_path)

    # 모델 로드
    bundle = joblib.load('xgboost_model.pkl')
    dt_model = bundle['model']
    features = bundle['features']

    # 특징 추출
    df_input = utils.extract_avg_std_features_df(temp_path)
    df_input = df_input[features]

    # 예측
    pred = dt_model.predict(df_input)[0]
    proba = dt_model.predict_proba(df_input)[0]
    class_names = dt_model.classes_

    st.subheader(f"예측된 Color: {pred}")
    st.write("예측 확률 분포:")
    st.bar_chart(pd.DataFrame(proba, index=class_names, columns=['확률']))

    # 예측 결과를 기반으로 콘텐츠 선택
    color_group = pred

    content = color_content_map.get(color_group, {
        'emotion': '감정을 찾을 수 없음',
        'places': ['어딘가의 꽃길'],
        'quotes': ['당신만의 색으로 세상을 물들여요.'],
        'music': [None]
    })

    place = random.choice(content['places'])
    quote = random.choice(content['quotes'])
    music_url = random.choice(content['music'])
    emotion = content['emotion']

    st.markdown(f"**어울리는 장소:** {place}")
    st.markdown(f"**색상이 주는 감정:** {emotion}")
    st.markdown(f"**감성 글귀:** _{quote}_")

 
    if music_url:
        st.markdown(f"[YouTube에서 감성 음악 듣기]({music_url})") 
    else:
        st.info("어울리는 음악이 아직 준비되지 않았어요!")
