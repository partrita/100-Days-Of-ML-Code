# Day 92: 프로젝트를 위한 미세 조정 및 최적화 (Fine-tuning and Optimization for the project)

## 학습 목표
- 어제(Day 91)의 오류 분석 및 모델 개선 아이디어를 바탕으로, 선택된 주요 모델의 성능을 극대화하기 위한 미세 조정 및 최적화 작업 수행.
- 하이퍼파라미터 튜닝 기법(그리드 서치, 랜덤 서치, 베이지안 최적화 등)을 실제로 적용하여 최적의 하이퍼파라미터 조합 탐색.
- 특징 공학 아이디어를 실제로 구현하고, 그 효과를 검증.
- (딥러닝 모델의 경우) 학습률 스케줄링, 조기 종료, 배치 정규화, 드롭아웃 등의 기법을 활용하여 학습 안정성 및 일반화 성능 향상.
- 반복적인 실험과 평가를 통해 최적의 모델 구성 도출.

## 1. 하이퍼파라미터 튜닝 (Hyperparameter Tuning) 심화
- Day 75에서 학습한 내용을 바탕으로, Day 90에서 선택된 유망한 모델(들)에 대해 본격적인 하이퍼파라미터 튜닝을 진행합니다.

### 가. 튜닝 대상 하이퍼파라미터 및 탐색 범위 설정
- **모델별 주요 하이퍼파라미터 숙지**:
    - **로지스틱 회귀**: `C` (규제 강도), `penalty` ('l1', 'l2'), `solver`.
    - **랜덤 포레스트/GBM/XGBoost/LightGBM**: `n_estimators`, `max_depth`, `learning_rate` (부스팅 계열), `min_samples_split`, `min_samples_leaf`, `subsample`, `colsample_bytree`, 규제 파라미터 (`reg_alpha`, `reg_lambda` 등).
    - **SVM**: `C`, `kernel` ('linear', 'rbf', 'poly'), `gamma` (rbf, poly 커널), `degree` (poly 커널).
    - **LSTM/RNN (Keras)**: 임베딩 차원, LSTM/RNN 유닛 수, 드롭아웃 비율, 옵티마이저 종류 및 학습률, 배치 크기.
- **탐색 범위 설정**:
    - 너무 넓으면 계산 비용이 커지고, 너무 좁으면 최적값을 놓칠 수 있습니다.
    - 일반적으로 로그 스케일(Logarithmic Scale)로 탐색하는 것이 효과적인 경우가 많습니다 (예: `C`: [0.01, 0.1, 1, 10, 100]).
    - 이전 실험 결과나 관련 연구/문서를 참고하여 합리적인 범위를 설정합니다.

### 나. 튜닝 도구 선택 및 적용
- **Scikit-learn 모델**:
    - `GridSearchCV`: 모든 조합 탐색. 계산량이 많을 수 있음.
    - `RandomizedSearchCV`: 지정된 횟수만큼 무작위 조합 탐색. 계산 효율적.
    - 교차 검증(`cv`) 폴드 수와 평가지표(`scoring`)를 적절히 설정.
- **Keras (TensorFlow) 딥러닝 모델**:
    - `KerasTuner`: Keras 모델을 위한 하이퍼파라미터 튜닝 라이브러리 (RandomSearch, Hyperband, BayesianOptimization 지원).
    - 또는 `GridSearchCV`/`RandomizedSearchCV`와 Keras 모델을 래핑하는 `KerasClassifier`/`KerasRegressor` (from `scikeras.wrappers` 또는 과거 `tf.keras.wrappers.scikit_learn`)를 함께 사용할 수 있으나, KerasTuner가 더 권장됨.
- **XGBoost/LightGBM**:
    - Scikit-learn 래퍼 사용 시 `GridSearchCV`/`RandomizedSearchCV` 활용.
    - 자체 API 사용 시, 반복문을 통해 직접 탐색하거나 `Optuna`, `Hyperopt`와 같은 베이지안 최적화 도구 사용.
- **베이지안 최적화 도구 (고급)**: `Optuna`, `Hyperopt`, `Scikit-Optimize (skopt)` 등. 이전 탐색 결과를 바탕으로 다음 탐색 지점을 효율적으로 결정.

```python
# 하이퍼파라미터 튜닝 예시 (Random Forest 모델, RandomizedSearchCV 사용)
# (Day 90, 91의 X_train_tfidf, y_train 사용 가정)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint # 정수형 랜덤 값 생성용

# 1. 튜닝할 모델 정의
rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)

# 2. 하이퍼파라미터 탐색 공간 정의
param_dist_rf = {
    'n_estimators': randint(50, 300), # 50에서 299 사이의 정수
    'max_depth': [None] + list(randint(5, 30).rvs(5)), # None 또는 5~29 사이 랜덤 정수 5개
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'max_features': ['sqrt', 'log2', None] + list(np.random.uniform(0.1, 1.0, 3)), # 문자열 또는 0.1~1.0 사이 랜덤 실수 3개
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
}

# 3. RandomizedSearchCV 객체 생성
# n_iter: 시도할 조합 수
# cv: 교차 검증 폴드 수
# scoring: 평가지표
# random_state: 재현성
# verbose: 로그 출력
random_search_rf = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_dist_rf,
    n_iter=20,  # 예시로 20번 시도 (실제로는 더 많이)
    cv=3,       # 예시로 3-폴드 (실제로는 5 또는 StratifiedKFold 객체)
    scoring='f1_macro',
    random_state=42,
    verbose=1,
    n_jobs=-1 # 모든 코어 사용
)

print("\nRandomForest 하이퍼파라미터 튜닝 시작 (RandomizedSearch)...")
# X_train_tfidf, y_train 사용 (Day 90에서 생성된 데이터)
# --- 데이터 준비 (Day 90, 91 코드 참고) ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

data_for_modeling = {
    'processed_text': [
        "movie great wonderful", "terrible film hated", "not bad could better",
        "amazing masterpiece cinema", "okay nothing special write home",
        "fantastic story engaging characters", "boring plot uninspired acting",
        "truly enjoyed this experience", "would not recommend this", "a must see film",
        "very good indeed", "awful and boring", "quite nice actually", "superb film", "meh it was okay"
    ] * 2, # 데이터 양 늘리기
    'sentiment': [1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0] * 2,
}
df_model = pd.DataFrame(data_for_modeling)
X_text_data = df_model['processed_text']
y_target = df_model['sentiment']

X_train_text, _, y_train, _ = train_test_split( # 테스트셋은 여기선 불필요
    X_text_data, y_target, test_size=0.2, random_state=42, stratify=y_target
)
tfidf_vectorizer = TfidfVectorizer(max_features=100)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
# --- 데이터 준비 끝 ---

random_search_rf.fit(X_train_tfidf, y_train)
print("튜닝 완료!")

print("\n최적 하이퍼파라미터 (RandomForest):", random_search_rf.best_params_)
print("최적 교차 검증 F1 (Macro) 점수:", random_search_rf.best_score_)

# 최적 모델 저장 또는 사용
best_rf_model = random_search_rf.best_estimator_

# (KerasTuner 예시는 복잡도가 있어 생략, 필요시 공식 문서 참고)
# https://keras.io/keras_tuner/
```

## 2. 특징 공학 (Feature Engineering) 반복 및 검증
- Day 89에서 생성했거나, Day 91의 오류 분석을 통해 새로 고안한 특징들을 실제 모델 학습에 적용해보고 성능 변화를 관찰합니다.
- **방법**:
    1.  새로운 특징을 추가하거나 기존 특징을 수정한 데이터셋을 준비합니다.
    2.  이전과 동일한 모델(또는 튜닝된 모델)로 학습하고 (교차) 검증 성능을 비교합니다.
    3.  성능이 향상되었다면 해당 특징을 채택하고, 그렇지 않다면 다른 아이디어를 시도하거나 기존 특징을 유지합니다.
- **예시 (어제 아이디어 기반)**:
    - (감성 분석) 문장 내 느낌표(!)나 물음표(?) 개수를 특징으로 추가.
    - (감성 분석) 긍정/부정 사전 단어의 등장 비율을 특징으로 추가.
    - (텍스트) 텍스트 길이를 로그 변환한 특징 사용.
- **주의**: 새로운 특징을 추가할 때마다 모델의 복잡도가 증가하고 과적합의 위험이 생길 수 있으므로, 항상 검증 세트 성능을 기준으로 판단해야 합니다. 특징 선택(Feature Selection) 기법을 병행할 수도 있습니다.

```python
# 특징 공학 효과 검증 예시 (기존 X_train_tfidf에 새로운 특징 추가 가정)
# (실제로는 Day 89에서 만든 특징을 사용)
from scipy.sparse import hstack # 희소 행렬 결합용

# 예시: 텍스트 길이 특징 추가 (이미 계산되어 있다고 가정)
# X_train_text_lengths = np.array([len(text.split()) for text in X_train_text]).reshape(-1, 1)
# (StandardScaler 등으로 스케일링 필요할 수 있음)

# X_train_combined_features = hstack([X_train_tfidf, X_train_text_lengths])

# 이 X_train_combined_features를 사용하여 모델 학습 및 평가, 이전 결과와 비교
# random_search_rf.fit(X_train_combined_features, y_train)
# print("새로운 특징 추가 후 최적 F1 점수:", random_search_rf.best_score_)
```

## 3. 딥러닝 모델 최적화 기법 (해당되는 경우)
- 딥러닝 모델(LSTM, CNN, Transformer 등)을 사용하는 경우, 다음과 같은 기법들을 활용하여 학습을 안정화하고 성능을 향상시킬 수 있습니다.

### 가. 학습률 스케줄링 (Learning Rate Scheduling)
- 학습 과정 동안 학습률을 동적으로 조절하는 기법.
- 초기에는 큰 학습률로 빠르게 수렴하고, 점차 학습률을 줄여 최적점에 더 잘 도달하도록 셔도움.
- 예: `tf.keras.callbacks.LearningRateScheduler`, `tf.keras.optimizers.schedules`.

### 나. 조기 종료 (Early Stopping)
- 검증 세트의 성능이 일정 에포크 동안 향상되지 않으면 학습을 조기에 중단하여 과적합을 방지하고 불필요한 학습 시간 절약.
- `tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)`

### 다. 배치 정규화 (Batch Normalization)
- 각 레이어의 입력 분포를 정규화하여 학습을 안정화하고 속도를 높이며, 그래디언트 소실/폭주 문제를 완화.
- `tf.keras.layers.BatchNormalization`

### 라. 드롭아웃 (Dropout)
- 학습 과정에서 무작위로 일부 뉴런을 비활성화하여 모델이 특정 뉴런에 과도하게 의존하는 것을 방지하고 일반화 성능 향상.
- `tf.keras.layers.Dropout(rate=0.2)`

### 마. 모델 체크포인트 (Model Checkpointing)
- 학습 과정 중 가장 좋은 성능(예: 가장 낮은 검증 손실)을 보인 모델의 가중치를 저장.
- `tf.keras.callbacks.ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)`

### 바. 옵티마이저 선택 및 조정
- `Adam`, `RMSprop`, `SGD` 등 다양한 옵티마이저와 그 파라미터(학습률, 모멘텀 등)를 실험.

## 4. 반복적인 실험 및 결과 기록/분석
- **체계적인 실험 관리**:
    - 각 실험(하이퍼파라미터 조합, 특징 세트, 모델 아키텍처 변경 등)에 대한 설정과 결과를 명확하게 기록합니다. (예: 스프레드시트, MLflow, Weights & Biases와 같은 실험 관리 도구)
    - 어떤 변경이 어떤 결과를 가져왔는지 추적 가능해야 합니다.
- **검증 세트 성능 기반 의사결정**: 모든 미세 조정과 최적화는 학습 데이터 내의 검증 세트(또는 교차 검증 폴드)에 대한 성능을 기준으로 이루어져야 합니다.
- **시간 제약 고려**: 모든 가능한 조합을 시도하는 것은 불가능하므로, 제한된 시간 내에 가장 효과적인 개선을 찾는 데 집중합니다. 우선순위가 높은 아이디어부터 실험합니다.

## 5. 최적 모델 구성 도출
- 여러 번의 반복적인 실험과 평가를 통해, 최종적으로 테스트 세트에서 평가할 **가장 성능이 좋고 안정적인 모델 구성(아키텍처, 특징 세트, 하이퍼파라미터)**을 결정합니다.
- 이 과정에서 "최고의" 단일 모델을 찾을 수도 있고, 여러 좋은 모델을 앙상블하는 전략을 고려할 수도 있습니다.

## 실습 아이디어
- 본인의 캡스톤 프로젝트에서 Day 90에 선정한 주요 모델(들)에 대해, 오늘 배운 하이퍼파라미터 튜닝 기법(RandomizedSearchCV 또는 GridSearchCV)을 실제로 적용해보세요.
    - 튜닝할 하이퍼파라미터와 탐색 범위를 정의하고, 교차 검증을 통해 최적의 조합을 찾습니다.
    - 튜닝 전후의 (교차 검증) 성능을 비교합니다.
- Day 91의 오류 분석에서 얻은 아이디어를 바탕으로, 새로운 특징을 생성하거나 기존 특징을 수정하여 데이터셋을 업데이트하고, 이 데이터셋으로 튜닝된 모델을 다시 학습시켜 성능 변화를 확인해보세요.
- (딥러닝 모델 사용 시) 조기 종료, 드롭아웃 등의 기법을 모델에 추가하고 학습 과정과 성능 변화를 관찰해보세요.
- 각 실험 결과를 간단하게라도 기록하여 어떤 시도가 효과적이었는지 정리합니다.

## 다음 학습 내용
- Day 93: 프로젝트를 위한 간단한 UI 또는 프레젠테이션 구축 (Building a simple UI or presentation for the project) - 최적화된 모델을 사용하여 예측 결과를 보여줄 수 있는 간단한 사용자 인터페이스를 만들거나, 프로젝트 결과를 발표할 프레젠테이션 자료 준비 시작.
