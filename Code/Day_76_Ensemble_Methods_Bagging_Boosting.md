# Day 76: 앙상블 방법 - 배깅 (랜덤 포레스트 복습), 부스팅 (AdaBoost, Gradient Boosting) (Ensemble Methods - Bagging, Boosting)

## 학습 목표
- 앙상블 학습(Ensemble Learning)의 개념과 장점 이해
- 주요 앙상블 기법 학습:
    - 배깅 (Bagging) 및 랜덤 포레스트 (Random Forest) 복습
    - 부스팅 (Boosting)의 기본 아이디어
    - 대표적인 부스팅 알고리즘: AdaBoost, Gradient Boosting Machine (GBM)
- 각 앙상블 기법의 작동 원리와 특징 비교

## 1. 앙상블 학습 (Ensemble Learning)
- **정의**: 여러 개의 개별 모델(기본 학습기, Base Learner 또는 약한 학습기, Weak Learner)을 학습시키고, 그 예측들을 결합하여 단일 모델보다 더 강력하고 안정적인 최종 예측을 만들어내는 기법입니다.
- **"다수결의 지혜" 또는 "집단 지성"** 원리에 기반합니다.
- **장점**:
    - **성능 향상**: 단일 모델보다 더 높은 정확도와 일반화 성능을 기대할 수 있습니다.
    - **과적합 감소**: 여러 모델의 예측을 평균내거나 결합함으로써 분산(Variance)을 줄여 과적합을 방지하는 데 도움이 됩니다.
    - **모델 안정성 증가**: 데이터의 작은 변화에 덜 민감한 모델을 만들 수 있습니다.

## 2. 배깅 (Bagging - Bootstrap Aggregating)

### 가. 기본 아이디어 (복습)
- **부트스트랩 샘플링 (Bootstrap Sampling)**: 원본 학습 데이터셋에서 중복을 허용하여 여러 개의 부분 데이터셋(부트스트랩 샘플)을 생성합니다. 각 부트스트랩 샘플은 원본 데이터와 크기가 동일합니다.
- **병렬 학습**: 각 부트스트랩 샘플에 대해 독립적으로 동일한 유형의 기본 학습기를 학습시킵니다.
- **결합 (Aggregating)**:
    - **회귀 문제**: 각 기본 학습기의 예측값을 평균냅니다.
    - **분류 문제**: 각 기본 학습기의 예측 클래스 중 다수결 투표(Majority Voting) 또는 확률 평균을 통해 최종 클래스를 결정합니다.

### 나. 랜덤 포레스트 (Random Forest) (복습)
- 배깅의 대표적인 알고리즘으로, 기본 학습기로 **결정 트리(Decision Tree)**를 사용합니다.
- 일반적인 배깅과의 차이점:
    - 각 트리를 학습할 때, 전체 특성 중 **무작위로 일부 특성만 선택**하여 최적의 분할을 찾습니다. (특성 샘플링)
    - 이를 통해 각 트리들이 서로 다른 특성에 집중하게 되어 트리 간의 상관관계를 줄이고, 모델의 다양성을 높여 일반화 성능을 더욱 향상시킵니다.
- **장점**:
    - 높은 정확도와 좋은 일반화 성능.
    - 과적합에 강한 편.
    - 대용량 데이터에도 잘 작동.
    - 특성 중요도(Feature Importance)를 제공.
- `scikit-learn`의 `RandomForestClassifier`, `RandomForestRegressor` 사용.

## 3. 부스팅 (Boosting)

### 가. 기본 아이디어
- 배깅과 달리, 부스팅은 **순차적으로** 약한 학습기들을 학습시킵니다.
- 각 단계에서 이전 학습기가 잘못 예측한 샘플에 더 큰 가중치를 부여하거나, 이전 학습기의 잔차(Residual)를 다음 학습기가 학습하도록 하여 모델을 점진적으로 개선해 나갑니다.
- 즉, 약한 학습기들을 여러 개 결합하여 강력한 학습기를 만드는 데 초점을 맞춥니다.
- 주로 편향(Bias)을 줄이는 데 효과적입니다.

### 나. 에이다부스트 (AdaBoost - Adaptive Boosting)
- 1995년 Freund와 Schapire에 의해 제안된 초기 부스팅 알고리즘 중 하나입니다.
- **작동 원리**:
    1.  모든 학습 데이터 샘플에 동일한 가중치를 부여하여 시작합니다.
    2.  첫 번째 약한 학습기(예: 간단한 결정 트리 - 스텀프(Stump): 깊이가 1인 트리)를 학습시키고, 예측 오류를 계산합니다.
    3.  **잘못 분류된 샘플에는 가중치를 높이고, 올바르게 분류된 샘플에는 가중치를 낮춥니다.**
    4.  업데이트된 가중치를 사용하여 다음 약한 학습기를 학습시킵니다. 이 학습기는 이전 단계에서 잘못 분류된 샘플에 더 집중하게 됩니다.
    5.  이 과정을 지정된 횟수만큼 반복합니다.
    6.  최종 예측은 각 약한 학습기의 예측을 **가중 합산(Weighted Sum)**하여 결정합니다. 이때, 성능이 좋은(오류율이 낮은) 학습기일수록 더 높은 가중치를 받습니다.
- **특징**:
    - 구현이 비교적 간단합니다.
    - 잘못 분류된 샘플에 집중하여 모델을 개선합니다.
    - 이상치(Outlier)에 민감할 수 있습니다 (잘못 분류된 이상치에 높은 가중치가 부여될 수 있음).
- `scikit-learn`의 `AdaBoostClassifier`, `AdaBoostRegressor` 사용.

### 다. 그래디언트 부스팅 머신 (Gradient Boosting Machine, GBM)
- AdaBoost와 유사하게 이전 학습기의 오류를 보완하는 방식으로 순차적으로 학습기를 추가하지만, **손실 함수(Loss Function)의 그래디언트(Gradient, 기울기)**를 사용하여 오류를 최적화합니다.
- **작동 원리 (회귀 문제 예시)**:
    1.  첫 번째 모델(보통 간단한 모델, 예: 데이터의 평균값)로 초기 예측을 합니다.
    2.  실제값과 예측값 사이의 **잔차(Residual = 실제값 - 예측값)**를 계산합니다. 이 잔차가 현재 모델이 설명하지 못하는 오류를 나타냅니다.
    3.  다음 약한 학습기는 이 **잔차를 예측하도록 학습**합니다. (즉, 이전 모델의 오류를 보정하려고 노력)
    4.  이전 모델의 예측값에 학습된 잔차 예측값(일정 학습률(learning rate)을 곱하여)을 더하여 전체 모델의 예측을 업데이트합니다.
    5.  이 과정을 반복하여 모델을 점진적으로 개선합니다. 각 단계에서 손실 함수를 최소화하는 방향으로 학습이 진행됩니다.
- **주요 하이퍼파라미터**:
    - `n_estimators`: 부스팅 단계의 수 (약한 학습기의 수).
    - `learning_rate`: 각 약한 학습기의 기여도를 조절하는 값 (0과 1 사이). 너무 크면 과적합, 너무 작으면 학습이 느려짐.
    - `max_depth`: 각 약한 학습기(주로 결정 트리)의 최대 깊이.
    - `subsample`: 각 트리를 학습할 때 사용할 학습 데이터의 비율 (Stochastic Gradient Boosting). 과적합 방지에 도움.
- **특징**:
    - 강력한 예측 성능을 보이며, 다양한 문제에 널리 사용됩니다.
    - AdaBoost보다 이상치에 덜 민감할 수 있습니다.
    - 다양한 손실 함수를 사용할 수 있어 유연성이 높습니다.
    - 하이퍼파라미터 튜닝이 중요하며, 과적합에 주의해야 합니다.
- `scikit-learn`의 `GradientBoostingClassifier`, `GradientBoostingRegressor` 사용.

## 4. 배깅 vs 부스팅

| 특징               | 배깅 (Bagging) - 예: 랜덤 포레스트                      | 부스팅 (Boosting) - 예: AdaBoost, GBM                     |
| ------------------ | ------------------------------------------------------- | --------------------------------------------------------- |
| **학습 방식**      | 병렬적 (독립적으로 학습)                                  | 순차적 (이전 모델의 결과에 기반하여 학습)                       |
| **주요 목표**      | 분산 감소 (Variance Reduction), 과적합 방지                | 편향 감소 (Bias Reduction), 모델 성능 점진적 개선             |
| **샘플 가중치**    | 모든 샘플에 동일한 가중치 (부트스트랩 샘플링으로 다양성 확보) | 이전 모델이 잘못 예측한 샘플에 더 높은 가중치 부여 (AdaBoost) |
| **오류 처리**      | -                                                       | 이전 모델의 오류(잔차)를 다음 모델이 학습 (GBM)               |
| **기본 학습기**    | 주로 분산이 큰 모델 (예: 완전히 성장한 결정 트리)           | 주로 편향이 큰 약한 학습기 (예: 얕은 결정 트리 - 스텀프)       |
| **과적합 민감도**  | 상대적으로 덜 민감                                        | 과적합에 민감할 수 있음 (특히 `n_estimators`가 클 때)        |
| **계산 효율성**    | 병렬 처리가 가능하여 학습 속도가 빠를 수 있음               | 순차적 학습으로 인해 병렬 처리가 어려움                       |

## 5. `scikit-learn`을 사용한 부스팅 모델 구현 예시

### 가. AdaBoost 예시
```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier # 약한 학습기로 사용
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 데이터 로드 (유방암 데이터셋 - 이진 분류)
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# AdaBoost 모델 생성 및 학습
# base_estimator: 사용할 약한 학습기 (기본값: DecisionTreeClassifier(max_depth=1))
# n_estimators: 약한 학습기의 수
# learning_rate: 학습률 (기본값: 1.0)
ada_model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1),
                               n_estimators=50,
                               learning_rate=1.0,
                               random_state=42)
ada_model.fit(X_train, y_train)

# 예측 및 평가
y_pred_ada = ada_model.predict(X_test)
accuracy_ada = accuracy_score(y_test, y_pred_ada)
print(f"AdaBoost 정확도: {accuracy_ada:.4f}")
```

### 나. Gradient Boosting Machine (GBM) 예시
```python
from sklearn.ensemble import GradientBoostingClassifier

# Gradient Boosting 모델 생성 및 학습
# n_estimators: 트리의 수
# learning_rate: 학습률
# max_depth: 각 트리의 최대 깊이
# subsample: 각 트리를 학습할 때 사용할 샘플의 비율 (Stochastic Gradient Boosting)
gbm_model = GradientBoostingClassifier(n_estimators=100,
                                     learning_rate=0.1,
                                     max_depth=3,
                                     subsample=0.8, # 80%의 샘플 사용
                                     random_state=42)
gbm_model.fit(X_train, y_train)

# 예측 및 평가
y_pred_gbm = gbm_model.predict(X_test)
accuracy_gbm = accuracy_score(y_test, y_pred_gbm)
print(f"Gradient Boosting 정확도: {accuracy_gbm:.4f}")

# 특성 중요도 확인 (GBM은 특성 중요도 제공)
# import pandas as pd
# feature_importances = pd.Series(gbm_model.feature_importances_, index=cancer.feature_names)
# print("\nGBM 특성 중요도 (상위 5개):\n", feature_importances.sort_values(ascending=False).head())
```

## 추가 학습 자료
- [Ensemble methods (Scikit-learn Documentation)](https://scikit-learn.org/stable/modules/ensemble.html)
- [A Gentle Introduction to Ensemble Learning (Machine Learning Mastery)](https://machinelearningmastery.com/what-is-ensemble-learning/)
- [StatQuest: AdaBoost, Clearly Explained (YouTube)](https://www.youtube.com/watch?v=LsK-xG1cLYA)
- [StatQuest: Gradient Boost Part 1 (of 4): Regression Main Ideas (YouTube)](https://www.youtube.com/watch?v=3CC4N4z3GJc)

## 다음 학습 내용
- Day 77: XGBoost - 소개 및 구현 (XGBoost - Introduction and implementation) - 그래디언트 부스팅의 강력한 확장판.
