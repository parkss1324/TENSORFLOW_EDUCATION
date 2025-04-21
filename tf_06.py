import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# x_train: 학습용 이미지 60.000개
# y_train: 학습용 레이블(0~9 정답)
# x_test: 테스트용 이미지 10,000개
# y_test: 테스트용 레이블

x_train = x_train.reshape(60000, 784).astype("float32") / 255 # 정규화
x_test = x_test.reshape(10000, 784).astype("float32") / 255

inputs = keras.Input(shape=(784,)) # MNIST 이미지(28x28)를 1D로 펼친(flatten) 상태
x = layers.Dense(64, activation="relu")(inputs) # 첫번째 은닉층, 뉴런 개수 64
x = layers.Dense(64, activation="relu")(x)      # 두번째 은닉층, 뉴런 개수 64
outputs = layers.Dense(10, activation="softmax")(x) # 마지막 출력층, 뉴런 개수 10으로 숫자 0~9 중 분류

model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model") # Keras Functional API 방식으로 모델 정의

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), # 손실 함수
    optimizer=keras.optimizers.RMSprop(),                              # 학습 속도 조절
    metrics=["accuracy"],                                              # 평가 지표: 정확도
)

history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2) 
# 1번에 64개씩 묶어서 학습(배치 학습)
# 전체 데이터 2번 학습
# 전체 학습 데이터를 80%/20%로 나눠서 20%는 검증 용도로 사용

test_scores = model.evaluate(x_test, y_test, verbose=2) # 학습된 모델을 테스트 데이터에 적용해 성능 측정
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

model.save("/Users/parksungsu/Documents/python_opencv/keras_test.keras") 
# 현재까지 학습한 모델을 디스크에 저장, 구성(구조) + 가중치 + 옵티마이저 상태까지 전부 저장됨
# keras_test.keras = 파일 이름 및 형식

del model # 메모리에서 현재 모델 삭제

model = keras.models.load_model("/Users/parksungsu/Documents/python_opencv/keras_test.keras") 
# 디스크에서 저장된 모델을 그대로 복원
# 이전과 똑같은 모델