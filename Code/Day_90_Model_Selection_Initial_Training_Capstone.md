# Day 90: 프로젝트를 위한 모델 선정 및 초기 학습 (Model Selection and initial training for the project)

## 학습 목표
- 전처리 및 특징 공학이 완료된 데이터를 사용하여 캡스톤 프로젝트에 적합한 머신러닝/딥러닝 모델 후보군 선정.
- 선택한 모델들에 대해 초기 하이퍼파라미터로 학습을 진행하고, 교차 검증을 통해 기본적인 성능 평가.
- 다양한 모델들의 성능을 비교하여 향후 집중적으로 튜닝하고 개선할 주요 모델 선택.
- 모델 학습 및 평가 파이프라인 구축의 기초 마련.

## 1. 모델 후보군 선정
- 프로젝트의 문제 유형(분류, 회귀, NLP, 시계열 등), 데이터의 특성(크기, 차원, 희소성 등), 사용 가능한 자원(시간, 계산 능력) 등을 고려하여 적절한 모델 후보들을 선택합니다.
- **일반적인 가이드라인**:
    - **분류 문제**:
        - 기본 모델: 로지스틱 회귀(Logistic Regression), K-최근접 이웃(KNN), 나이브 베이즈(Naive Bayes).
        - 트리 기반 모델: 결정 트리(Decision Tree), 랜덤 포레스트(Random Forest), 그래디언트 부스팅 머신(GBM), XGBoost, LightGBM.
        - 서포트 벡터 머신 (SVM).
        - 신경망/딥러닝: 다층 퍼셉트론(MLP), CNN (텍스트/이미지), RNN/LSTM (순차 데이터).
    - **회귀 문제**:
        - 기본 모델: 선형 회귀(Linear Regression), 릿지(Ridge), 라쏘(Lasso), 엘라스틱넷(ElasticNet).
        - 트리 기반 모델: 랜덤 포레스트 회귀, GBM 회귀, XGBoost 회귀, LightGBM 회귀.
        - 서포트 벡터 회귀 (SVR).
        - 신경망/딥러닝.
    - **자연어 처리 (텍스트 분류/회귀)**:
        - 전통적 방법: TF-IDF + (로지스틱 회귀, SVM, 나이브 베이즈, 랜덤 포레스트 등).
        - 딥러닝: Word Embeddings + (LSTM, GRU, 1D CNN, BERT 기반 모델 등).
    - **시계열 예측**:
        - 전통적 방법: ARIMA, SARIMA, 지수평활법.
        - 머신러닝/딥러닝: 과거 시점을 특징으로 변환 후 회귀 모델 적용, RNN/LSTM.
- **초기에는 너무 복잡한 모델보다는 해석 가능하고 빠르게 학습시킬 수 있는 모델부터 시작하는 것이 좋습니다.** (예: 로지스틱 회귀, 간단한 트리 모델)
- 이후 점차 복잡도가 높은 모델(앙상블, 딥러닝)을 추가하여 비교합니다.

**예시 (영화 리뷰 감성 분석 프로젝트 - 이진 분류)**:
- **후보 모델군**:
    1.  TF-IDF + 로지스틱 회귀 (베이스라인)
    2.  TF-IDF + 나이브 베이즈
    3.  TF-IDF + 랜덤 포레스트
    4.  Word2Vec (사전 훈련 또는 직접 학습) + LSTM
    5.  (선택) BERT 기반 사전 훈련 모델 미세 조정 (Fine-tuning) - 고급

## 2. 데이터 준비 및 분할
- Day 88, 89에서 준비한 최종 데이터셋을 사용합니다.
- **학습 데이터(Train Set)와 테스트 데이터(Test Set)로 분할**합니다.
    - `sklearn.model_selection.train_test_split`
    - 테스트 세트는 모델의 최종 성능 평가에만 사용하고, 모델 선택 및 하이퍼파라미터 튜닝 과정에서는 사용하지 않습니다.
- **교차 검증을 위한 준비**: 모델 선택 및 튜닝 과정에서는 학습 데이터를 다시 여러 폴드로 나누어 교차 검증을 수행합니다.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer # 예시용
# from tensorflow.keras.preprocessing.text import Tokenizer # 예시용
# from tensorflow.keras.preprocessing.sequence import pad_sequences # 예시용

# --- 예시 데이터 로드 (Day 89에서 생성된 df_final.csv 가정) ---
# 실제 프로젝트에서는 이전 단계에서 생성/저장한 데이터를 로드합니다.
# df_final = pd.read_csv('capstone_data_processed_featured.csv')
# X_text = df_final['processed_text_or_features'] # 텍스트 또는 특징 컬럼
# y = df_final['sentiment'] # 타겟 변수

# --- 간략화된 예시 데이터 (실제 프로젝트 데이터 사용) ---
# (Day 89의 df와 유사한 형태라고 가정, 텍스트 특징과 수치 특징이 섞여있을 수 있음)
# 여기서는 'processed_text'가 주요 입력이고 'sentiment'가 타겟이라고 가정
data_for_modeling = {
    'processed_text': [
        "movie great wonderful", "terrible film hated", "not bad could better",
        "amazing masterpiece cinema", "okay nothing special write home",
        "fantastic story engaging characters", "boring plot uninspired acting",
        "truly enjoyed this experience", "would not recommend this", "a must see film"
    ],
    'sentiment': [1, 0, 0, 1, 0, 1, 0, 1, 0, 1],
    # ... (다른 특징들이 있다면 함께 포함)
}
df_model = pd.DataFrame(data_for_modeling)
X_text_data = df_model['processed_text']
y_target = df_model['sentiment']
# --- 간략화된 예시 데이터 끝 ---


# 1. 학습/테스트 데이터 분할
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text_data, y_target, test_size=0.2, random_state=42, stratify=y_target
)

print("학습 데이터 크기:", X_train_text.shape, y_train.shape)
print("테스트 데이터 크기:", X_test_text.shape, y_test.shape)

# 2. 텍스트 데이터 벡터화 (예: TF-IDF) - 전통적 모델용
# (실제로는 Day 89에서 생성한 특징을 사용하거나, 여기서 새로 생성)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=100) # 예시로 최대 특징 100개
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)
print("TF-IDF 벡터화된 학습 데이터 크기:", X_train_tfidf.shape)

# 3. (딥러닝 모델용) 텍스트 데이터 정수 인코딩 및 패딩 - LSTM 등 딥러닝 모델용
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# vocab_size = 500 # 어휘 크기
# max_length = 20  # 최대 시퀀스 길이
# oov_tok = "<unk>"

# tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
# tokenizer.fit_on_texts(X_train_text)

# X_train_seq = tokenizer.texts_to_sequences(X_train_text)
# X_test_seq = tokenizer.texts_to_sequences(X_test_text)

# X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
# X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')
# print("패딩된 학습 시퀀스 데이터 크기:", X_train_pad.shape)
```

## 3. 초기 모델 학습 및 평가 (교차 검증 사용)
- 선택한 모델 후보군에 대해 기본 하이퍼파라미터 또는 간단하게 설정한 하이퍼파라미터로 학습을 진행합니다.
- `sklearn.model_selection.cross_val_score` 또는 `cross_validate`를 사용하여 교차 검증을 수행하고, 주요 평가지표(정확도, F1 점수, ROC AUC 등)를 기록합니다.
- 딥러닝 모델의 경우, 교차 검증이 시간과 자원을 많이 소모할 수 있으므로, 학습 데이터의 일부를 검증 세트(Validation Set)로 분리하여 평가하거나, K-폴드 수를 줄여서(예: 3-폴드) 수행할 수 있습니다.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

# 교차 검증 설정
cv_stratified = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) # 예시로 3-폴드

# 모델 후보군 정의
models = {
    "Logistic Regression": LogisticRegression(solver='liblinear', random_state=42),
    "Multinomial Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1) # 간단한 설정
}

results = {} # 모델별 성능 저장

print("\n--- 초기 모델 학습 및 교차 검증 (TF-IDF 기반) ---")
for model_name, model_instance in models.items():
    # cross_val_score는 여러 평가지표를 한 번에 반환하지 않으므로, 필요시 각각 실행
    accuracy_scores = cross_val_score(model_instance, X_train_tfidf, y_train, cv=cv_stratified, scoring='accuracy', n_jobs=-1)
    f1_scores = cross_val_score(model_instance, X_train_tfidf, y_train, cv=cv_stratified, scoring='f1_macro', n_jobs=-1) # macro f1 사용 예시

    results[model_name] = {
        'mean_accuracy': np.mean(accuracy_scores),
        'std_accuracy': np.std(accuracy_scores),
        'mean_f1_macro': np.mean(f1_scores),
        'std_f1_macro': np.std(f1_scores)
    }
    print(f"\n{model_name}:")
    print(f"  평균 정확도: {results[model_name]['mean_accuracy']:.4f} (±{results[model_name]['std_accuracy']:.4f})")
    print(f"  평균 F1 (Macro): {results[model_name]['mean_f1_macro']:.4f} (±{results[model_name]['std_f1_macro']:.4f})")

# 결과 비교
results_df = pd.DataFrame(results).T.sort_values(by='mean_f1_macro', ascending=False)
print("\n\n--- 모델별 교차 검증 성능 비교 (F1 Macro 기준 정렬) ---")
print(results_df)


# --- (선택) 딥러닝 모델 (LSTM) 초기 학습 예시 (Keras) ---
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.metrics import f1_score # Keras 모델 평가용

# def create_lstm_model(vocab_size_dl, embedding_dim_dl, max_length_dl, lstm_units=64, dropout_rate=0.2):
#     model = Sequential([
#         Embedding(input_dim=vocab_size_dl, output_dim=embedding_dim_dl, input_length=max_length_dl),
#         LSTM(units=lstm_units, dropout=dropout_rate, recurrent_dropout=dropout_rate),
#         Dense(units=1, activation='sigmoid')
#     ])
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# # 딥러닝 모델용 데이터 준비 (위에서 주석 처리된 부분 참고하여 X_train_pad, y_train 사용)
# # vocab_size, max_length, embedding_dim 등은 실제 데이터에 맞게 설정 필요
# # embedding_dim = 100

# # K-Fold 교차 검증 (딥러닝 모델은 시간이 오래 걸리므로 주의)
# # 또는 간단히 학습 데이터의 일부를 검증 데이터로 분리하여 평가
# # lstm_f1_scores = []
# # for fold_idx, (train_idx, val_idx) in enumerate(cv_stratified.split(X_train_pad, y_train)):
# #     print(f"\n--- LSTM Fold {fold_idx+1}/{cv_stratified.get_n_splits()} ---")
# #     X_train_fold, X_val_fold = X_train_pad[train_idx], X_train_pad[val_idx]
# #     y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx] # y_train이 Series일 경우 iloc 사용

# #     lstm_model = create_lstm_model(vocab_size, embedding_dim, max_length)
# #     early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# #     history = lstm_model.fit(X_train_fold, y_train_fold,
# #                              epochs=5, # 초기에는 적은 에포크로 테스트
# #                              batch_size=32,
# #                              validation_data=(X_val_fold, y_val_fold),
# #                              callbacks=[early_stopping],
# #                              verbose=1)

# #     y_pred_val = (lstm_model.predict(X_val_fold) > 0.5).astype("int32")
# #     f1_val = f1_score(y_val_fold, y_pred_val, average='macro')
# #     lstm_f1_scores.append(f1_val)
# #     print(f"Fold {fold_idx+1} F1 (Macro): {f1_val:.4f}")

# # if lstm_f1_scores:
# #     print(f"\nLSTM 평균 F1 (Macro): {np.mean(lstm_f1_scores):.4f} (±{np.std(lstm_f1_scores):.4f})")
# #     results['LSTM'] = {
# #         'mean_accuracy': np.nan, # 정확도는 history에서 가져와야 함
# #         'std_accuracy': np.nan,
# #         'mean_f1_macro': np.mean(lstm_f1_scores),
# #         'std_f1_macro': np.std(lstm_f1_scores)
# #     }
# #     results_df = pd.DataFrame(results).T.sort_values(by='mean_f1_macro', ascending=False)
# #     print("\n\n--- 모델별 교차 검증 성능 비교 (LSTM 포함) ---")
# #     print(results_df)
```

## 4. 초기 결과 분석 및 다음 단계 결정
- 각 모델의 교차 검증 성능(평균 및 표준편차)을 비교합니다.
- **고려 사항**:
    - **성능**: 어떤 모델이 가장 좋은 성능을 보이는가? (예: F1 점수, ROC AUC)
    - **안정성**: 교차 검증 폴드 간 성능 표준편차가 작은가? (표준편차가 크면 모델이 데이터 분할에 민감하다는 의미)
    - **학습 시간**: 모델 학습에 소요되는 시간은 현실적인가?
    - **해석 가능성**: 모델의 예측 결과를 얼마나 쉽게 해석할 수 있는가? (필요하다면)
    - **구현 복잡도**: 모델 구현 및 관리가 얼마나 복잡한가?
- 이 결과를 바탕으로, **1~3개 정도의 유망한 모델을 선택**하여 다음 단계인 하이퍼파라미터 튜닝 및 모델 개선을 진행합니다.
- 성능이 매우 낮은 모델은 제외하거나, 특징 공학 또는 데이터 전처리를 다시 검토하여 개선할 여지가 있는지 확인합니다.
- 특정 모델이 특정 평가지표에서는 좋지만 다른 지표에서는 나쁠 수 있으므로, 프로젝트 목표에 맞는 주요 평가지표를 기준으로 판단합니다.

## 5. 파이프라인 구축의 중요성
- 데이터 전처리, 특징 공학, 모델 학습, 평가 과정을 연결하는 파이프라인(`sklearn.pipeline.Pipeline`)을 구축하면 코드를 간결하게 만들고, 데이터 누수를 방지하며, 전체 워크플로우를 효율적으로 관리할 수 있습니다.
- 특히 교차 검증이나 하이퍼파라미터 튜닝 시 파이프라인을 사용하면 각 폴드마다 전처리 과정을 올바르게 적용하는 데 매우 유용합니다.

## 실습 아이디어
- 본인의 캡스톤 프로젝트 데이터에 대해 오늘 배운 내용을 적용해보세요.
    - 프로젝트에 적합한 모델 후보군을 3개 이상 선정합니다.
    - 각 모델에 대해 초기 하이퍼파라미터로 설정하고, (계층별) K-폴드 교차 검증을 수행하여 주요 평가지표(정확도, F1 점수 등)를 기록합니다.
    - (선택) Scikit-learn의 `Pipeline`을 사용하여 전처리(예: TF-IDF 벡터화)와 모델 학습을 하나로 묶어 교차 검증을 수행해보세요.
- 교차 검증 결과를 비교하여, 다음 단계에서 집중적으로 개선할 모델 1~2개를 선정하고 그 이유를 정리해보세요.

## 다음 학습 내용
- Day 91: 프로젝트를 위한 모델 평가 및 반복 (Model Evaluation and Iteration for the project) - 선택된 모델들의 테스트 세트 성능을 상세히 평가하고, 오류 분석 등을 통해 모델 개선 방향을 설정하며 반복적으로 모델을 개선.
