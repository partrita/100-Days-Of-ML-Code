# Day 77: XGBoost - 소개 및 구현 (XGBoost - Introduction and implementation)

## 학습 목표
- XGBoost (Extreme Gradient Boosting)의 개념과 등장 배경 이해
- XGBoost가 기존 그래디언트 부스팅 머신(GBM)에 비해 가지는 주요 장점 학습
    - 규제(Regularization)를 통한 과적합 방지
    - 병렬 처리 및 최적화를 통한 빠른 학습 속도
    - 결측치(Missing Values) 자체 처리 기능
    - 교차 검증 내장 기능 등
- XGBoost의 주요 하이퍼파라미터 이해
- 파이썬 XGBoost 라이브러리를 사용한 모델 구현 방법 숙지

## 1. XGBoost (Extreme Gradient Boosting) 소개
- **정의**: 그래디언트 부스팅 알고리즘을 기반으로 성능과 속도를 극대화한 트리 기반의 앙상블 학습 프레임워크입니다.
- Tianqi Chen에 의해 개발되었으며, 캐글(Kaggle)과 같은 데이터 과학 경진대회에서 압도적인 성능을 보여주며 널리 사용되기 시작했습니다.
- 분류(Classification)와 회귀(Regression) 문제 모두에 사용할 수 있습니다.

## 2. XGBoost의 주요 장점 및 특징

### 가. 규제 (Regularization)
- 표준 GBM에는 없는 **L1 규제(Lasso Regression)**와 **L2 규제(Ridge Regression)** 항을 손실 함수에 추가하여 모델의 복잡도를 제어하고 과적합을 방지합니다.
    - L1 규제 (`reg_alpha`): 가중치의 절대값 합에 페널티를 부여하여 일부 가중치를 0으로 만들어 특징 선택 효과를 가집니다.
    - L2 규제 (`reg_lambda`): 가중치의 제곱 합에 페널티를 부여하여 가중치 크기를 줄여 모델을 부드럽게 만듭니다.
- 이는 모델이 너무 복잡해지는 것을 막아 일반화 성능을 향상시킵니다.

### 나. 병렬 처리 및 최적화 (Parallel Processing and Optimization)
- **트리 생성 시 병렬 처리**: 각 노드에서 최적의 분할 지점을 찾는 과정을 병렬로 처리하여 학습 속도를 크게 향상시킵니다. (주의: 트리가 순차적으로 생성되는 부스팅의 특성상 전체 트리를 병렬로 학습하는 것은 아님)
- **캐시 인식 접근 (Cache-aware Access)**: 메모리 접근 패턴을 최적화하여 하드웨어 성능을 최대한 활용합니다.
- **블록 압축 및 샤딩 (Block Compression and Sharding)**: 대용량 데이터를 효율적으로 처리하기 위한 기술.

### 다. 결측치 자체 처리 (Inbuilt Missing Value Treatment)
- 데이터 전처리 단계에서 결측치를 특별히 처리하지 않아도 XGBoost는 학습 과정에서 결측치를 자동으로 처리하는 방법을 학습합니다.
- 각 노드에서 분기할 때 결측치를 가진 샘플을 어느 쪽으로 보낼지(왼쪽 또는 오른쪽 자식 노드)를 학습하여 최적의 방향을 결정합니다.

### 라. 트리 가지치기 (Tree Pruning)
- 일반적인 GBM은 탐욕적(Greedy) 방식으로 트리를 성장시킨 후 가지치기를 하지만, XGBoost는 `max_depth`에 도달할 때까지 트리를 성장시킨 후, 손실 감소 기여도가 음수인 분기(즉, 모델 성능에 도움이 안 되는 분기)를 역방향으로 가지치기합니다 (Post-pruning).
- `gamma` (또는 `min_split_loss`) 파라미터를 사용하여 리프 노드를 추가적으로 분할할 최소 손실 감소 값을 지정하여, 이 값보다 작은 손실 감소를 보이는 분기는 수행하지 않습니다.

### 마. 교차 검증 내장 기능 (Built-in Cross-Validation)
- XGBoost API 내에 교차 검증 기능(`xgb.cv`)이 포함되어 있어, 최적의 부스팅 라운드 수(`n_estimators` 또는 `num_boost_round`)를 찾는 데 유용합니다.
- 학습 과정에서 검증 세트의 성능을 모니터링하고, 성능이 더 이상 향상되지 않으면 조기 종료(Early Stopping)할 수 있습니다.

### 바. 높은 유연성 및 확장성
- 사용자 정의 손실 함수(Custom Objective Function) 및 평가 지표(Evaluation Metric)를 사용할 수 있습니다.
- 다양한 프로그래밍 언어(Python, R, Java, Scala, C++ 등)와 분산 환경(Hadoop, Spark)을 지원합니다.

## 3. XGBoost의 주요 하이퍼파라미터
- XGBoost는 많은 하이퍼파라미터를 가지고 있으며, 적절한 튜닝이 중요합니다.

### 가. 일반 파라미터 (General Parameters)
- `booster` [기본값=gbtree]: 사용할 부스터 모델 유형. `gbtree`(트리 기반 모델), `gblinear`(선형 모델), `dart`(Dropout을 추가한 트리 모델) 중 선택.
- `nthread` [기본값=시스템 최대 스레드 수]: 병렬 처리에 사용할 스레드 수.
- `verbosity` [기본값=1]: 메시지 출력 레벨. 0(Silent), 1(Warning), 2(Info), 3(Debug).

### 나. 부스터 파라미터 (Booster Parameters) - `gbtree` 기준
- `eta` (또는 `learning_rate`) [기본값=0.3]: 학습률. 각 부스팅 단계에서 가중치를 얼마나 줄일지 결정. 낮은 값은 모델을 더 견고하게 만들지만, `num_boost_round`를 늘려야 함. (0.01 ~ 0.2 권장)
- `gamma` (또는 `min_split_loss`) [기본값=0]: 리프 노드를 추가적으로 분할하기 위한 최소 손실 감소 값. 클수록 모델이 보수적(덜 복잡)이 됨.
- `max_depth` [기본값=6]: 각 트리의 최대 깊이. 클수록 모델이 복잡해지고 과적합 가능성이 높아짐. (3 ~ 10 권장)
- `min_child_weight` [기본값=1]: 자식 노드에 필요한 최소한의 관측치 가중치 합. 과적합 방지에 사용.
- `subsample` [기본값=1]: 각 트리를 학습할 때 사용할 학습 데이터 샘플의 비율. (0.5 ~ 1 권장)
- `colsample_bytree` [기본값=1]: 각 트리를 구성할 때 사용할 특성(피처)의 비율. (0.5 ~ 1 권장)
- `colsample_bylevel`, `colsample_bynode`: 더 세부적인 특성 샘플링 비율.
- `lambda` (또는 `reg_lambda`) [기본값=1]: L2 규제 가중치.
- `alpha` (또는 `reg_alpha`) [기본값=0]: L1 규제 가중치.
- `tree_method` [기본값=auto]: 트리 생성 알고리즘. `auto`, `exact`, `approx`, `hist`, `gpu_hist` 등. `hist`가 대용량 데이터에 효율적.

### 다. 학습 과정 파라미터 (Learning Task Parameters)
- `objective` [기본값=reg:squarederror]: 학습 목표 함수.
    - `reg:squarederror`: 회귀 문제 (제곱 오차).
    - `binary:logistic`: 이진 분류 (로지스틱 손실). 예측값은 확률.
    - `multi:softmax`: 다중 클래스 분류 (소프트맥스 손실). 예측값은 클래스. `num_class` 파라미터 필요.
    - `multi:softprob`: 다중 클래스 분류. 예측값은 각 클래스에 대한 확률. `num_class` 파라미터 필요.
- `eval_metric`: 검증 세트에 사용할 평가 지표.
    - 회귀: `rmse`, `mae` 등.
    - 분류: `error` (분류 오류율), `logloss` (로그 손실), `auc` (ROC AUC), `aucpr` (PR AUC), `merror` (다중 클래스 오류율) 등.
- `seed` [기본값=0]: 재현성을 위한 랜덤 시드.

## 4. 파이썬 XGBoost 라이브러리 사용법
- `xgboost` 라이브러리를 설치해야 합니다: `pip install xgboost`

### 가. Scikit-Learn 래퍼 (Wrapper) 사용
- XGBoost는 Scikit-Learn과 호환되는 래퍼 클래스(`XGBClassifier`, `XGBRegressor`)를 제공하여 기존 Scikit-Learn 워크플로우에 쉽게 통합할 수 있습니다.

```python
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 데이터 로드 (유방암 데이터셋 - 이진 분류)
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# XGBClassifier 모델 생성 및 학습 (Scikit-Learn 래퍼)
# 주요 하이퍼파라미터 설정 예시
xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic', # 이진 분류
    n_estimators=100,          # 부스팅 라운드 수 (트리 개수)
    learning_rate=0.1,         # 학습률
    max_depth=3,               # 트리의 최대 깊이
    subsample=0.8,             # 각 트리에 사용할 샘플 비율
    colsample_bytree=0.8,      # 각 트리에 사용할 특성 비율
    gamma=0,                   # 최소 손실 감소
    reg_alpha=0,               # L1 규제
    reg_lambda=1,              # L2 규제
    use_label_encoder=False,   # LabelEncoder 사용 경고 방지 (XGBoost 1.3.0 이상)
    eval_metric='logloss',     # 평가 지표 (학습 중 출력)
    random_state=42
)

# 조기 종료(Early Stopping) 설정하여 학습
# eval_set: 검증 데이터셋. 조기 종료 및 학습 과정 모니터링에 사용
# early_stopping_rounds: 지정된 라운드 동안 성능 향상이 없으면 학습 중단
eval_set = [(X_test, y_test)] # 테스트셋을 검증셋으로 사용 (실제로는 별도 검증셋 권장)
xgb_clf.fit(X_train, y_train,
            early_stopping_rounds=10,
            eval_set=eval_set,
            verbose=True) # verbose=True로 학습 과정 출력

# 예측
y_pred = xgb_clf.predict(X_test)
y_pred_proba = xgb_clf.predict_proba(X_test)[:, 1]

# 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"\nXGBoost 정확도: {accuracy:.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 특성 중요도 시각화
# xgb.plot_importance(xgb_clf)
# plt.show()
```

### 나. XGBoost 고유 API 사용
- XGBoost는 자체적인 데이터 구조(`DMatrix`)와 학습 API를 가지고 있습니다. 대용량 데이터 처리나 세밀한 제어에 더 유리할 수 있습니다.

```python
# 1. DMatrix 생성 (XGBoost 전용 데이터 구조)
dtrain = xgb.DMatrix(data=X_train, label=y_train)
dtest = xgb.DMatrix(data=X_test, label=y_test) # 레이블은 평가에만 사용

# 2. 파라미터 설정 (딕셔너리 형태)
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.1, # learning_rate
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0,
    'lambda': 1, # L2 규제
    'alpha': 0,  # L1 규제
    'seed': 42
}

num_boost_round = 200 # 부스팅 라운드 수

# 3. 모델 학습 (xgb.train)
# watchlist: 학습 과정에서 성능을 모니터링할 데이터셋 리스트
watchlist = [(dtrain, 'train'), (dtest, 'eval')]

bst_model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=num_boost_round,
    evals=watchlist,
    early_stopping_rounds=10, # 조기 종료
    verbose_eval=10 # 10 라운드마다 평가 결과 출력
)

# 4. 예측 (확률값 반환)
pred_probs_bst = bst_model.predict(dtest)
# 확률을 이진 클래스로 변환 (임계값 0.5 기준)
preds_bst = [1 if prob > 0.5 else 0 for prob in pred_probs_bst]

# 5. 평가
accuracy_bst = accuracy_score(y_test, preds_bst)
print(f"\nXGBoost (Native API) 정확도: {accuracy_bst:.4f}")

# 최적의 부스팅 라운드 수 확인 (조기 종료 시)
print(f"최적 부스팅 라운드: {bst_model.best_iteration}")
```

### 다. 교차 검증 (`xgb.cv`)
```python
# xgb.cv를 사용한 교차 검증
# 파라미터는 xgb.train과 유사
# nfold: 폴드 수
# metrics: 평가 지표 리스트
# early_stopping_rounds: 조기 종료
# as_pandas: 결과를 Pandas DataFrame으로 반환

cv_results = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=num_boost_round,
    nfold=5, # 5-fold cross-validation
    metrics={'logloss', 'auc'}, # 여러 지표 사용 가능
    early_stopping_rounds=10,
    seed=42,
    verbose_eval=50,
    as_pandas=True
)

print("\nXGBoost CV 결과 (마지막 5개 라운드):\n", cv_results.tail())
print(f"\n최적 CV LogLoss: {cv_results['test-logloss-mean'].min()} at round {cv_results['test-logloss-mean'].idxmin()}")
print(f"최적 CV AUC: {cv_results['test-auc-mean'].max()} at round {cv_results['test-auc-mean'].idxmax()}")
```

## 5. XGBoost 하이퍼파라미터 튜닝
- `GridSearchCV` 또는 `RandomizedSearchCV` (Scikit-Learn 래퍼 사용 시) 또는 `Hyperopt`, `Optuna`와 같은 베이지안 최적화 도구를 사용하여 튜닝할 수 있습니다.
- 주요 튜닝 대상: `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `gamma`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`.
- 일반적으로 `learning_rate`를 낮추고(`eta` < 0.1), `n_estimators` (또는 `num_boost_round`)를 조기 종료와 함께 크게 설정하는 전략이 많이 사용됩니다.

## 추가 학습 자료
- [XGBoost 공식 문서](https://xgboost.readthedocs.io/en/latest/)
- [A Gentle Introduction to XGBoost for Applied Machine Learning (Machine Learning Mastery)](https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/)
- [XGBoost Algorithm: Long May It Reign! (Towards Data Science)](https://towardsdatascience.com/xgboost-algorithm-long-may-it-reign-9029759fa4a0)

## 다음 학습 내용
- Day 78: LightGBM - 소개 및 구현 (LightGBM - Introduction and implementation) - XGBoost와 유사하지만 더 빠르고 효율적인 또 다른 그래디언트 부스팅 프레임워크.
