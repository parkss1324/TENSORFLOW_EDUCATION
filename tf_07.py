import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

encoder_input = keras.Input(shape=(28, 28, 1), name="img")
x = layers.Conv2D(16, 3, activation="relu")(encoder_input) 
# 필터 수 = 16, Filter size = 3, Padding = 0(기본값), Stride = 1(기본값)
# Output size = {(Input size - Filter size + 2 x Padding) / Stride} + 1
# 26 = {(28 - 3 + 2 x 0) / 1} + 1
# shape = (26, 26, 16)

x = layers.Conv2D(32, 3, activation="relu")(x)
# 24 = {(26 - 3 + 2 x 0) / 1} + 1
# shape = (24, 24 ,32)

x = layers.MaxPooling2D(3)(x)
# (24, 24, 32)의 Output Shape를 3x3의 칸으로 나눔
# shape = (8, 8, 32)

x = layers.Conv2D(32, 3, activation="relu")(x)
# 6 = {(8 - 3 + 2 x 0) / 1} + 1
# shape = (6, 6, 32)

x = layers.Conv2D(16, 3, activation="relu")(x)
# 4 = {(6 - 3 + 2 x 0) / 1} + 1
# shape = (4, 4, 16)

encoder_output = layers.GlobalMaxPooling2D()(x)
# 각 채널마다 최대값 하나만 남김
# 4x4의 각 채널(feature map)마다 최대값 1개만 남김 (가로로 16인 벡터가 생김)

encoder = keras.Model(encoder_input, encoder_output, name="encoder")
encoder.summary()

x = layers.Reshape((4, 4, 1))(encoder_output)
# 텐서의 형태를 원하는 대로 바꾸는 층, 데이터 순서를 유지하면서 모양만 바꿈

x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
# CNN의 역방향 버전, 이미지를 크게(업샘플링)할 때 사용
# Output_size = (Input_size - 1) x Stride - 2 x Padding + Kernel_size
# 6 = (4 - 1) x 1 - 0 + 3 

x = layers.Conv2DTranspose(32, 3, activation="relu")(x)
x = layers.UpSampling2D(3)(x)
# 이미지를 단순 확대 
# 3의 값이므로 크기(행렬)를 3개 확대

x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
decoder_output = layers.Conv2DTranspose(1, 3, activation="relu")(x)

autoencoder = keras.Model(encoder_input, decoder_output, name="autoencoder")
autoencoder.summary()