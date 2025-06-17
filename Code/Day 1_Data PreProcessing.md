# 데이터 전처리
<p align="center">
  <img src="https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Info-graphs/Day%201.jpg">
</p>

인포그래픽에 표시된 것처럼 데이터 전처리를 6가지 필수 단계로 나눕니다.
이 예제에 사용된 데이터셋은 [여기](https://github.com/Avik-Jain/100-Days-Of-ML-Code/tree/master/datasets)에서 가져오세요.

## 1단계: 라이브러리 가져오기
```Python
import numpy as np
import pandas as pd
```
## 2단계: 데이터셋 가져오기
```python
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[ : , :-1].values # 종속 변수를 제외한 모든 열 선택
Y = dataset.iloc[ : , 3].values # 종속 변수 열 선택
```
## 3단계: 누락된 데이터 처리
```python
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0) # 누락된 값을 평균으로 대체
imputer = imputer.fit(X[ : , 1:3]) # 숫자형 열에 대해 imputer 학습
X[ : , 1:3] = imputer.transform(X[ : , 1:3]) # 누락된 값 변환
```
## 4단계: 범주형 데이터 인코딩
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0]) # 첫 번째 열(국가)을 숫자로 인코딩
```
### 더미 변수 만들기
```python
onehotencoder = OneHotEncoder(categorical_features = [0]) # 첫 번째 열을 원-핫 인코딩
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y) # 종속 변수(구매 여부)를 숫자로 인코딩
```
## 5단계: 데이터셋을 훈련 세트와 테스트 세트로 분할
```python
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0) # 80% 훈련, 20% 테스트
```

## 6단계: 특징 스케일링
```python
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) # 훈련 세트에 대해 스케일러 학습 및 변환
X_test = sc_X.transform(X_test) # 테스트 세트에 대해 학습된 스케일러로 변환 (fit_transform 아님)
```
### 완료 :smile:
