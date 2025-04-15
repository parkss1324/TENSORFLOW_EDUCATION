import tensorflow as tf
import numpy as np

# 데이터 준비(float 설정으로 학습의 정밀도를 높임)
X = np.array([1, 2, 3, 4], dtype=float) # 독립 변수
Y = np.array([2, 4, 6, 8], dtype=float) # 종속 변수 Y = 2 * X의 관계

# 모델 정의
model = tf.keras.Sequential([ # Sequential은 순차적으로 레이어를 쌓는 가장 단순한 모델 구조
    tf.keras.layers.Dense(units=1, input_shape=[1]) # 완전 연결층 하나만 사용, 출력 뉴런 1개 / 입력 특성이 1 
])

# 컴파일
model.compile(optimizer='sgd', loss='mean_squared_error')
# sgd = 경사하강법
# mean_squared_error = 평균 제곱 오차

# 학습
model.fit(X, Y, epochs=100) # fit() = 실제 데이터를 가지고 모델로 학습, 100번 반복 학습

# 예측
result = model.predict(np.array([[5.0]]))  # 새로운 입력에 대한 출력을 예측
print("예측 결과:", result)