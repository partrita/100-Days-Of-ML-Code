# 단순 선형 회귀


<p align="center">
  <img src="https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Info-graphs/Day%202.jpg">
</p>


# 1단계: 데이터 전처리
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('studentscores.csv')
X = dataset.iloc[ : ,   : 1 ].values # 독립 변수 (학습 시간)
Y = dataset.iloc[ : , 1 ].values # 종속 변수 (점수)

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 1/4, random_state = 0) # 75% 훈련, 25% 테스트
```

# 2단계: 훈련 세트에 단순 선형 회귀 모델 피팅
 ```python
 from sklearn.linear_model import LinearRegression
 regressor = LinearRegression()
 regressor = regressor.fit(X_train, Y_train) # 훈련 데이터로 모델 학습
 ```
 # 3단계: 결과 예측
 ```python
 Y_pred = regressor.predict(X_test) # 테스트 데이터로 예측
 ```
 
 # 4단계: 시각화
 ## 훈련 결과 시각화
 ```python
 plt.scatter(X_train , Y_train, color = 'red') # 실제 훈련 데이터 점
 plt.plot(X_train , regressor.predict(X_train), color ='blue') # 훈련 데이터에 대한 예측 선
 ```
 ## 테스트 결과 시각화
 ```python
 plt.scatter(X_test , Y_test, color = 'red') # 실제 테스트 데이터 점
 plt.plot(X_test , regressor.predict(X_test), color ='blue') # 테스트 데이터에 대한 예측 선 (X_train에 대한 예측 선과 동일해야 함)
 ```
