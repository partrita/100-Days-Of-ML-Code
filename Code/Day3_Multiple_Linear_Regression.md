# 다중 선형 회귀


<p align="center">
  <img src="https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Info-graphs/Day%203.jpg">
</p>


## 1단계: 데이터 전처리

### 라이브러리 가져오기
```python
import pandas as pd
import numpy as np
```
### 데이터셋 가져오기
```python
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[ : , :-1].values # 독립 변수 (R&D 지출, 관리비, 마케팅 지출, 주)
Y = dataset.iloc[ : ,  4 ].values # 종속 변수 (수익)
```

### 범주형 데이터 인코딩
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[: , 3] = labelencoder.fit_transform(X[ : , 3]) # '주' 열을 숫자로 인코딩
onehotencoder = OneHotEncoder(categorical_features = [3]) # '주' 열을 원-핫 인코딩
X = onehotencoder.fit_transform(X).toarray()
```

### 더미 변수 함정 피하기
```python
X = X[: , 1:] # 첫 번째 더미 변수 열을 제거하여 다중공선성 방지
```

### 데이터셋을 훈련 세트와 테스트 세트로 분할
```python
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0) # 80% 훈련, 20% 테스트
```
## 2단계: 훈련 세트에 다중 선형 회귀 피팅
```python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train) # 훈련 데이터로 모델 학습
```

## 3단계: 테스트 세트 결과 예측
```python
y_pred = regressor.predict(X_test) # 테스트 데이터로 예측
```
