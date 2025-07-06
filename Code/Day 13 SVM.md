# 13일차 | 서포트 벡터 머신 (SVM)

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
X_test = sc.fit_transform(X_test) # 참고: 여기서는 fit_transform 대신 transform을 사용해야 합니다. 테스트 세트에서는 훈련 세트에서 학습한 스케일링을 그대로 적용해야 합니다.
```

## 훈련 세트에 SVM 피팅
```python
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0) # kernel: 사용할 커널 함수 ('linear'는 선형 커널), random_state: 난수 시드
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

## 훈련 세트 결과 시각화

```python
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (훈련 세트)')
plt.xlabel('나이')
plt.ylabel('예상 급여')
plt.legend()
plt.show()
```
<p align="center">
  <img src="https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Other%20Docs/ets.png">
</p>

## 테스트 세트 결과 시각화
```python
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (테스트 세트)')
plt.xlabel('나이')
plt.ylabel('예상 급여')
plt.legend()
plt.show()
```
<p align="center">
  <img src="https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Other%20Docs/test.png">
</p>
