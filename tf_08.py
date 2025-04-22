import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

num_tags = 12  
num_words = 10000 # 단어 수, 인덱스 0~9999  
num_departments = 4 # 4개 부서(ex 기술 지원, 회계, 인사, 마케팅 등)

# 입력
title_input = keras.Input(shape=(None,), name="title")  
body_input = keras.Input(shape=(None,), name="body")  
tags_input = keras.Input(shape=(num_tags,), name="tags")  

# 임베딩(Embedding, 텍스트 데이터를 숫자 벡터로 바꿔서 신경망이 이해할 수 있게 만듬)
# 단어 인덱스를 밀집 벡터로 변환, 10000개의 단어를 64차원으로 변환
title_features = layers.Embedding(num_words, 64)(title_input) 
body_features = layers.Embedding(num_words, 64)(body_input)

# 순환 신경망(LSTM, Long Short-Term Memory)
# 장기 기억을 잘하는 신경망
# 기본 RNN은 시간이 지날수록 정보(기억)를 잊어버리는 문제를 해결
# 필요한 정보를 오래 기억, 불필요한 건 잊음
title_features = layers.LSTM(128)(title_features) # 제목 128차원으로 변환
body_features = layers.LSTM(32)(body_features) # 본문 32차원으로 변환

# 모든 입력에서 추출된 특징을 하나의 벡터로 합침
# title(128) + body(32) + tags(12) = 127차원 벡터
x = layers.concatenate([title_features, body_features, tags_input])

# 출력
priority_pred = layers.Dense(1, name="priority")(x) 
# 출력 뉴런이 1, 하나의 숫자를 예측
# 우선 순위 예측
# 글이나 요청이 긴급한지 보통인지 판단하는 용도

department_pred = layers.Dense(num_departments, name="department")(x)
# 부서 분류 예측
# 다중 분류

# 다중 입력 + 다중 출력 모델 생성
model = keras.Model(
    inputs=[title_input, body_input, tags_input],
    outputs=[priority_pred, department_pred],
)

keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)