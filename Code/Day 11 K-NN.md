# K-최근접 이웃 (K-NN)

<p align="center">
  <img src="https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Info-graphs/Day%207.jpg">
</p>

## 데이터셋 | 소셜 네트워크

<p align="center">
  <img src="https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Other%20Docs/data.PNG">
</p> 


## 라이브러리 가져오기
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

## 데이터셋 가져오기
```python
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values
```

## 데이터셋을 훈련 세트와 테스트 세트로 분할
```python
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
```
## 특징 스케일링
```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```
## 훈련 세트에 K-NN 피팅
```python
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2) # n_neighbors: 이웃 수, metric: 거리 측정 방식, p: 민코프스키 거리의 파라미터 (2는 유클리드 거리)
classifier.fit(X_train, y_train)
```
## 테스트 세트 결과 예측
```python
y_pred = classifier.predict(X_test)
```

## 혼동 행렬 만들기
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
```
