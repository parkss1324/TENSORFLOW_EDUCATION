import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 데이터 준비
X = np.array([1, 2, 3, 4], dtype=float)
Y = np.array([2, 4, 6, 8], dtype=float)

# 모델 정의
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# 컴파일
model.compile(optimizer='sgd', loss='mean_squared_error')

# 학습 (히스토리 저장)
history = model.fit(X, Y, epochs=100, verbose=0)

# 예측
result = model.predict(np.array([[5.0]]))
print("예측 결과:", result)

# 학습 손실 시각화
plt.plot(history.history['loss'])
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
