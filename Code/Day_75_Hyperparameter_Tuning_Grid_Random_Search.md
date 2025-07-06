# Day 75: 하이퍼파라미터 튜닝 - 그리드 서치, 랜덤 서치 (Hyperparameter Tuning - Grid Search, Random Search)

## 학습 목표
- 모델 파라미터(Model Parameters)와 하이퍼파라미터(Hyperparameters)의 차이 이해
- 하이퍼파라미터 튜닝의 중요성과 목적 학습
- 주요 하이퍼파라미터 튜닝 기법 학습:
    - 수동 탐색 (Manual Search)
    - 그리드 서치 (Grid Search)
    - 랜덤 서치 (Random Search)
- `scikit-learn`을 사용한 그리드 서치 및 랜덤 서치 구현 방법 숙지
- 교차 검증을 하이퍼파라미터 튜닝에 어떻게 활용하는지 이해

## 1. 모델 파라미터 vs 하이퍼파라미터

### 가. 모델 파라미터 (Model Parameters)
- 모델이 학습 과정에서 데이터로부터 **스스로 학습하는 변수**들입니다.
- 사용자가 직접 설정하지 않고, 모델이 학습 데이터를 통해 최적의 값을 찾아냅니다.
- 예:
    - 선형 회귀 모델의 계수(coefficients) 및 절편(intercept).
    - 로지스틱 회귀 모델의 계수.
    - 신경망의 가중치(weights) 및 편향(biases).
    - 결정 트리의 각 노드에서의 분기 조건.

### 나. 하이퍼파라미터 (Hyperparameters)
- 모델의 학습 과정을 제어하거나 모델의 구조를 결정하는 변수들로, **사용자가 직접 설정**해야 합니다.
- 모델이 학습을 시작하기 전에 미리 정의되어야 하며, 학습 과정에서 업데이트되지 않습니다.
- 하이퍼파라미터의 선택에 따라 모델의 성능이 크게 달라질 수 있습니다.
- 예:
    - 로지스틱 회귀, SVM의 규제 강도 `C`.
    - K-최근접 이웃(KNN)의 이웃 수 `k`.
    - 결정 트리, 랜덤 포레스트의 최대 깊이 `max_depth`, 최소 샘플 분할 수 `min_samples_split`.
    - 신경망의 학습률(learning rate), 은닉층의 수, 각 층의 뉴런 수, 활성화 함수, 옵티마이저 종류, 배치 크기, 에포크 수.
    - PCA의 주성분 개수 `n_components`.

## 2. 하이퍼파라미터 튜닝 (Hyperparameter Tuning / Optimization)
- **정의**: 모델의 성능을 최적화하기 위해 가장 좋은 하이퍼파라미터 조합을 찾는 과정입니다.
- **중요성**: 적절한 하이퍼파라미터 설정은 모델의 일반화 성능을 크게 향상시킬 수 있습니다. 잘못된 하이퍼파라미터는 과소적합(Underfitting)이나 과적합(Overfitting)을 유발할 수 있습니다.
- **목표**: 검증 세트(Validation Set) 또는 교차 검증(Cross-Validation)을 통해 가장 좋은 성능을 내는 하이퍼파라미터 조합을 찾습니다.

## 3. 주요 하이퍼파라미터 튜닝 기법

### 가. 수동 탐색 (Manual Search)
- 사용자가 경험이나 직관에 의존하여 하이퍼파라미터 값을 직접 변경해가며 모델 성능을 확인하는 방식입니다.
- 간단한 모델이나 하이퍼파라미터 수가 적을 때는 시도해볼 수 있지만, 비효율적이고 최적의 조합을 찾기 어렵습니다.

### 나. 그리드 서치 (Grid Search)
- **개념**: 사용자가 지정한 하이퍼파라미터 값들의 모든 가능한 조합에 대해 모델 성능을 평가하여 가장 좋은 조합을 찾는 방법입니다.
- **수행 단계**:
    1.  튜닝할 하이퍼파라미터와 탐색할 값들의 목록을 정의합니다. (격자, Grid 생성)
    2.  정의된 모든 하이퍼파라미터 조합에 대해 모델을 학습하고 교차 검증을 통해 성능을 평가합니다.
    3.  가장 높은 교차 검증 성능을 보인 하이퍼파라미터 조합을 최적의 조합으로 선택합니다.
- **장점**:
    - 지정된 범위 내에서 모든 조합을 탐색하므로, 최적의 조합을 찾을 가능성이 비교적 높습니다.
    - 구현이 간단합니다.
- **단점**:
    - 하이퍼파라미터의 종류가 많거나 탐색할 값의 범위가 넓으면 계산 비용이 기하급수적으로 증가합니다 (차원의 저주와 유사).
    - 연속적인 하이퍼파라미터의 경우 이산적인 값들만 탐색합니다.

### 다. 랜덤 서치 (Random Search)
- **개념**: 그리드 서치와 유사하지만, 모든 조합을 시도하는 대신 지정된 횟수만큼 하이퍼파라미터 조합을 무작위로 샘플링하여 평가합니다.
- **수행 단계**:
    1.  튜닝할 하이퍼파라미터와 각 하이퍼파라미터의 탐색 범위(또는 확률 분포)를 정의합니다.
    2.  지정된 횟수(`n_iter`)만큼 하이퍼파라미터 조합을 무작위로 선택하여 모델을 학습하고 교차 검증을 통해 성능을 평가합니다.
    3.  가장 높은 교차 검증 성능을 보인 하이퍼파라미터 조합을 최적의 조합으로 선택합니다.
- **장점**:
    - 그리드 서치보다 계산 효율성이 높습니다. (특히 하이퍼파라미터 공간이 클 때)
    - 모든 하이퍼파라미터가 동일하게 중요하지 않다는 가정 하에, 중요한 하이퍼파라미터에 대해 더 다양한 값을 탐색할 가능성이 있습니다.
    - 제한된 시간 내에 좋은 성능의 조합을 찾을 확률이 높습니다.
- **단점**:
    - 무작위 탐색이므로 최적의 조합을 반드시 찾는다는 보장은 없습니다. (하지만 충분한 반복 횟수를 설정하면 좋은 결과를 얻을 수 있음)

### 라. 베이지안 최적화 (Bayesian Optimization) - 고급 기법
- 이전 탐색 결과를 바탕으로 다음 탐색할 하이퍼파라미터 조합을 더 효율적으로 결정하는 방식입니다. (예: Hyperopt, Optuna 라이브러리)
- 랜덤 서치보다 더 적은 반복으로 좋은 성능을 찾을 수 있는 경우가 많습니다.

## 4. `scikit-learn`을 사용한 그리드 서치 및 랜덤 서치 구현
- `GridSearchCV`와 `RandomizedSearchCV` 클래스를 사용합니다.
- 내부적으로 교차 검증을 수행하여 각 하이퍼파라미터 조합의 성능을 평가합니다.

### 가. 예제 데이터 및 모델 준비
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC # 예제 모델로 SVM 사용
from sklearn.preprocessing import StandardScaler
import numpy as np

# 데이터 로드 및 분할
iris = load_iris()
X, y = iris.data, iris.target

# 데이터 스케일링 (SVM은 스케일링에 민감)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# 기본 모델 (튜닝 전)
base_model = SVC(random_state=42)
base_model.fit(X_train, y_train)
print(f"기본 모델 정확도: {base_model.score(X_test, y_test):.4f}")
```

### 나. 그리드 서치 (GridSearchCV)
```python
# 1. 하이퍼파라미터 그리드 정의
# SVM의 주요 하이퍼파라미터: C (규제 강도), kernel, gamma (커널 계수)
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.1, 1] # 'rbf', 'poly', 'sigmoid' 커널에만 해당
}

# 2. GridSearchCV 객체 생성
# estimator: 사용할 모델
# param_grid: 탐색할 하이퍼파라미터 그리드
# cv: 교차 검증 폴드 수 (또는 교차 검증기 객체)
# scoring: 성능 평가 지표 (예: 'accuracy', 'f1')
# n_jobs: 병렬 처리할 CPU 코어 수 (-1이면 모든 코어 사용)
# verbose: 로그 출력 레벨
grid_search = GridSearchCV(estimator=SVC(random_state=42),
                           param_grid=param_grid,
                           cv=5, # 5-fold cross-validation
                           scoring='accuracy',
                           n_jobs=-1,
                           verbose=1)

# 3. 그리드 서치 수행 (학습 데이터에 대해)
print("\nGridSearchCV 시작...")
grid_search.fit(X_train, y_train)
print("GridSearchCV 완료!")

# 4. 최적 하이퍼파라미터 및 성능 확인
print("\n최적 하이퍼파라미터:", grid_search.best_params_)
print("최적 교차 검증 점수 (Accuracy):", grid_search.best_score_)

# 5. 최적 모델로 테스트 데이터 평가
best_model_grid = grid_search.best_estimator_ # 최적 하이퍼파라미터로 학습된 모델
test_accuracy_grid = best_model_grid.score(X_test, y_test)
print("테스트 세트 정확도 (GridSearch):", test_accuracy_grid)

# (선택) 모든 결과 확인
# results_df = pd.DataFrame(grid_search.cv_results_)
# print("\nGridSearchCV 결과 상세:\n", results_df[['param_C', 'param_kernel', 'param_gamma', 'mean_test_score']].sort_values(by='mean_test_score', ascending=False).head())
```

### 다. 랜덤 서치 (RandomizedSearchCV)
```python
from scipy.stats import expon, uniform # 확률 분포 정의용

# 1. 하이퍼파라미터 탐색 범위 (분포) 정의
param_dist = {
    'C': expon(scale=100), # 지수 분포 (큰 값에 더 많은 확률 부여, 예시) / 또는 [0.1, 1, 10, 100, 1000] 같은 리스트
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': expon(scale=.1), # 또는 ['scale', 'auto', 0.01, 0.1, 1]
    'degree': [2, 3, 4] # 'poly' 커널에만 해당
}

# 2. RandomizedSearchCV 객체 생성
# n_iter: 시도할 하이퍼파라미터 조합의 수
random_search = RandomizedSearchCV(estimator=SVC(random_state=42),
                                   param_distributions=param_dist,
                                   n_iter=50, # 50개의 조합을 무작위로 시도
                                   cv=5,
                                   scoring='accuracy',
                                   n_jobs=-1,
                                   random_state=42, # 재현성을 위해 설정
                                   verbose=1)

# 3. 랜덤 서치 수행
print("\nRandomizedSearchCV 시작...")
random_search.fit(X_train, y_train)
print("RandomizedSearchCV 완료!")

# 4. 최적 하이퍼파라미터 및 성능 확인
print("\n최적 하이퍼파라미터 (RandomSearch):", random_search.best_params_)
print("최적 교차 검증 점수 (Accuracy, RandomSearch):", random_search.best_score_)

# 5. 최적 모델로 테스트 데이터 평가
best_model_random = random_search.best_estimator_
test_accuracy_random = best_model_random.score(X_test, y_test)
print("테스트 세트 정확도 (RandomSearch):", test_accuracy_random)
```

## 5. 하이퍼파라미터 튜닝 시 고려사항
- **탐색 공간 정의**: 하이퍼파라미터의 탐색 범위나 분포를 적절히 설정하는 것이 중요합니다. 너무 넓으면 비효율적이고, 너무 좁으면 최적값을 놓칠 수 있습니다. 도메인 지식이나 이전 경험을 활용할 수 있습니다.
- **계산 비용**: 그리드 서치는 조합 수가 많아지면 매우 오래 걸립니다. 랜덤 서치나 베이지안 최적화는 제한된 예산 내에서 더 효율적일 수 있습니다.
- **교차 검증 폴드 수**: `cv` 값을 적절히 설정해야 합니다. 일반적으로 5 또는 10이 사용됩니다.
- **평가 지표**: `scoring` 파라미터를 문제에 맞는 적절한 평가지표로 설정해야 합니다 (예: 불균형 데이터에서는 'f1' 또는 'roc_auc').
- **데이터 누수 방지**: 하이퍼파라미터 튜닝은 학습 데이터(또는 학습 데이터 내의 검증 폴드)에 대해서만 수행되어야 하며, 최종 평가는 별도의 테스트 세트에서 이루어져야 합니다. `GridSearchCV`와 `RandomizedSearchCV`는 내부적으로 이를 잘 처리합니다.

## 추가 학습 자료
- [Scikit-learn: Tuning the hyper-parameters of an estimator](https://scikit-learn.org/stable/modules/grid_search.html)
- [Hyperparameter Tuning the Random Forest in Python (Towards Data Science)](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74)
- [A Gentle Introduction to Algorithm Tuning (Machine Learning Mastery)](https://machinelearningmastery.com/a-gentle-introduction-to-algorithm-tuning/)

## 다음 학습 내용
- Day 76: 앙상블 방법 - 배깅 (랜덤 포레스트 복습), 부스팅 (AdaBoost, Gradient Boosting) (Ensemble Methods - Bagging (revisit Random Forests), Boosting (AdaBoost, Gradient Boosting))
