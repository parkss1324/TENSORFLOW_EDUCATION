import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(784,)) # MNIST 이미지(28x28)를 1D로 펼친(flatten) 상태
x = layers.Dense(64, activation="relu")(inputs) # 첫번째 은닉층, 뉴런 개수 64
x = layers.Dense(64, activation="relu")(x)      # 두번째 은닉층, 뉴런 개수 64
outputs = layers.Dense(10, activation="softmax")(x) # 마지막 출력층, 뉴런 개수 10으로 숫자 0~9 중 분류

model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model") # Keras Functional API 방식으로 모델 정의
model.summary() # 모델 요약 출력, 층의 이름, 출력 형태, 파라미터 개수 등을 보여줌