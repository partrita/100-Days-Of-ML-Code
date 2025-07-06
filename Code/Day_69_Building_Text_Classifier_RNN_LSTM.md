# Day 69: RNN/LSTM을 이용한 텍스트 분류기 구축 (Building a text classifier using RNN/LSTM)

## 학습 목표
- `TensorFlow/Keras`를 사용하여 RNN 또는 LSTM 기반의 텍스트 분류 모델을 구축하는 과정 이해
- 텍스트 데이터 전처리: 토큰화, 정수 인코딩, 패딩
- 임베딩 레이어(Embedding Layer)의 역할과 사용법 학습
- RNN/LSTM 레이어를 포함한 모델 구성 및 학습, 평가 방법 숙지

## 1. 텍스트 분류 문제 정의
- 예: 영화 리뷰가 긍정인지 부정인지 분류 (이진 분류)
- 예: 뉴스 기사가 어떤 카테고리(스포츠, 정치, 경제 등)에 속하는지 분류 (다중 클래스 분류)

## 2. 개발 환경 및 라이브러리
- **TensorFlow & Keras**: 딥러닝 모델 구축 및 학습을 위한 주요 프레임워크.
- **NumPy**: 수치 연산.
- **Scikit-learn**: 데이터 분할, 평가 지표 등 (선택 사항).
- **NLTK / KoNLPy**: 텍스트 전처리 (토큰화 등, 선택 사항). Keras의 `Tokenizer`도 사용 가능.

```bash
pip install tensorflow numpy scikit-learn nltk # 또는 konlpy
```

## 3. 구현 단계

### 가. 데이터 준비 및 로드
- 레이블링된 텍스트 데이터셋이 필요합니다. (예: IMDB 영화 리뷰 데이터셋)
- Keras는 IMDB 데이터셋과 같은 일부 텍스트 데이터셋을 내장하고 있어 쉽게 로드할 수 있습니다.

```python
import numpy as np
from tensorflow.keras.datasets import imdb # IMDB 영화 리뷰 데이터셋
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer # 직접 텍스트를 로드할 경우
# from sklearn.model_selection import train_test_split # 직접 텍스트를 로드할 경우

# --- IMDB 데이터셋 로드 예시 ---
num_words = 10000  # 사용할 단어의 최대 개수 (등장 빈도 기준)
(X_train_imdb, y_train_imdb), (X_test_imdb, y_test_imdb) = imdb.load_data(num_words=num_words)

print("훈련용 리뷰 개수: {}".format(len(X_train_imdb)))
print("테스트용 리뷰 개수: {}".format(len(X_test_imdb)))
print("첫 번째 훈련용 리뷰 (정수 시퀀스):", X_train_imdb[0][:10]) # 앞 10개 단어
print("첫 번째 훈련용 리뷰 레이블:", y_train_imdb[0]) # 0: 부정, 1: 긍정

# --- 직접 텍스트 데이터를 로드하고 전처리하는 경우 (예시) ---
# texts = ["This is a great movie.", "I hated this film.", ...] # 실제 텍스트 데이터
# labels = [1, 0, ...] # 긍정/부정 레이블

# # 1. Tokenizer 생성
# tokenizer = Tokenizer(num_words=num_words, oov_token="<unk>") # Out-of-vocabulary 토큰
# tokenizer.fit_on_texts(texts)

# # 2. 텍스트를 정수 시퀀스로 변환
# sequences = tokenizer.texts_to_sequences(texts)
# word_index = tokenizer.word_index
# print("Found %s unique tokens." % len(word_index))
```

### 나. 데이터 전처리

1.  **패딩 (Padding)**:
    - RNN/LSTM 모델은 고정된 길이의 시퀀스를 입력으로 받습니다.
    - 문장(리뷰)의 길이가 각기 다르므로, 최대 길이(maxlen)를 정하고 이보다 짧은 시퀀스는 특정 값(보통 0)으로 채워 길이를 맞춥니다 (패딩). 긴 시퀀스는 잘라냅니다.
    - `pad_sequences` 함수 사용.

    ```python
    maxlen = 200  # 최대 시퀀스 길이 (리뷰 단어 수 제한)

    X_train_padded = pad_sequences(X_train_imdb, maxlen=maxlen, padding='post', truncating='post')
    X_test_padded = pad_sequences(X_test_imdb, maxlen=maxlen, padding='post', truncating='post')

    print("패딩된 첫 번째 훈련용 리뷰 (앞 10개):", X_train_padded[0][:10])
    print("패딩된 훈련 데이터 크기:", X_train_padded.shape) # (샘플 수, maxlen)
    ```

### 다. 모델 구축 (RNN 또는 LSTM 사용)

1.  **임베딩 레이어 (Embedding Layer)**:
    - 정수 인코딩된 단어들을 밀집 벡터(Dense Vector)로 변환합니다.
    - `input_dim`: 어휘의 크기 (num_words).
    - `output_dim`: 임베딩 벡터의 차원 수.
    - `input_length`: 입력 시퀀스의 길이 (maxlen).
    - (선택) 사전 훈련된 단어 임베딩(Word2Vec, GloVe)을 로드하여 사용할 수도 있습니다.

2.  **RNN/LSTM 레이어**:
    - `SimpleRNN` 또는 `LSTM` 레이어를 추가합니다.
    - `units`: RNN/LSTM 셀의 뉴런(유닛) 수. 출력 공간의 차원이기도 합니다.
    - `return_sequences`: `True`이면 각 타임스텝의 은닉 상태를 모두 출력 (다음 RNN/LSTM 레이어가 있을 때), `False`이면 마지막 타임스텝의 은닉 상태만 출력 (주로 분류 레이어 직전에).

3.  **출력 레이어 (Dense Layer)**:
    - 분류를 위한 완전 연결 레이어.
    - 이진 분류: `units=1`, 활성화 함수 `sigmoid`.
    - 다중 클래스 분류: `units=클래스 수`, 활성화 함수 `softmax`.

```python
embedding_dim = 128  # 임베딩 벡터 차원

# --- LSTM 모델 구축 예시 ---
model_lstm = Sequential([
    Embedding(input_dim=num_words, output_dim=embedding_dim, input_length=maxlen),
    LSTM(units=64, dropout=0.2, recurrent_dropout=0.2), # dropout은 과적합 방지에 도움
    Dense(units=1, activation='sigmoid') # 이진 분류
])

model_lstm.compile(optimizer='adam',
                   loss='binary_crossentropy', # 이진 분류 손실 함수
                   metrics=['accuracy'])

model_lstm.summary()

# --- (선택) SimpleRNN 모델 구축 예시 ---
# model_rnn = Sequential([
#     Embedding(input_dim=num_words, output_dim=embedding_dim, input_length=maxlen),
#     SimpleRNN(units=64),
#     Dense(units=1, activation='sigmoid')
# ])
# model_rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model_rnn.summary()

# --- (선택) 양방향 LSTM (Bidirectional LSTM) 모델 구축 예시 ---
# Bidirectional LSTM은 시퀀스를 정방향과 역방향으로 모두 처리하여 문맥 정보를 더 잘 포착할 수 있습니다.
# model_bilstm = Sequential([
#     Embedding(input_dim=num_words, output_dim=embedding_dim, input_length=maxlen),
#     Bidirectional(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2)),
#     Dense(units=1, activation='sigmoid')
# ])
# model_bilstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model_bilstm.summary()
```

### 라. 모델 학습
- `fit` 메소드를 사용하여 모델을 학습시킵니다.
- `epochs`: 전체 데이터셋에 대한 학습 반복 횟수.
- `batch_size`: 한 번의 가중치 업데이트에 사용될 샘플 수.
- `validation_split` 또는 `validation_data`: 검증 데이터셋을 설정하여 학습 중 모델 성능 모니터링.

```python
epochs = 5  # 실제로는 더 많은 epoch 필요할 수 있음
batch_size = 64

print("\nLSTM 모델 학습 시작...")
history_lstm = model_lstm.fit(X_train_padded, y_train_imdb,
                              epochs=epochs,
                              batch_size=batch_size,
                              validation_split=0.2) # 훈련 데이터의 20%를 검증용으로 사용
print("LSTM 모델 학습 완료!")

# (선택) 학습 과정 시각화
import matplotlib.pyplot as plt

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, 'bo-', label='Training acc')
    plt.plot(epochs_range, val_acc, 'ro-', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, 'bo-', label='Training loss')
    plt.plot(epochs_range, val_loss, 'ro-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_history(history_lstm)
```

### 마. 모델 평가
- `evaluate` 메소드를 사용하여 테스트 데이터셋에서 모델의 성능(손실, 정확도 등)을 평가합니다.

```python
print("\nLSTM 모델 평가...")
loss_lstm, accuracy_lstm = model_lstm.evaluate(X_test_padded, y_test_imdb, batch_size=batch_size)
print(f"LSTM 모델 테스트 손실: {loss_lstm:.4f}")
print(f"LSTM 모델 테스트 정확도: {accuracy_lstm:.4f}")
```

### 바. 예측
- 학습된 모델을 사용하여 새로운 텍스트에 대한 예측을 수행할 수 있습니다.
- 예측을 위해서는 새로운 텍스트도 학습 데이터와 동일한 전처리(토큰화, 정수 인코딩, 패딩) 과정을 거쳐야 합니다.

```python
# 예시: 새로운 리뷰 예측 (IMDB 데이터셋의 단어 인덱스를 알아야 함)
# 실제로는 tokenizer.texts_to_sequences 와 pad_sequences를 사용해야 함

# word_index = imdb.get_word_index()
# reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# def decode_review(text_sequence):
#     return ' '.join([reverse_word_index.get(i - 3, '?') for i in text_sequence]) # 0,1,2는 패딩,시작,미등장 토큰

# sample_review_text = "This movie was fantastic! I really loved it and would recommend it to everyone."
# # 1. 토큰화 (실제로는 학습 시 사용한 tokenizer.texts_to_sequences 사용)
# # 아래는 IMDB 데이터셋이 이미 정수 인코딩 되어있다는 가정하에 임의의 시퀀스를 만드는 예시
# sample_sequence = [word_index.get(word.lower(), 2) + 3 for word in sample_review_text.split() if word_index.get(word.lower(), 2) < num_words-3][:maxlen]
# sample_padded = pad_sequences([sample_sequence], maxlen=maxlen, padding='post', truncating='post')

# prediction = model_lstm.predict(sample_padded)
# print(f"\n샘플 리뷰: '{sample_review_text}'")
# print(f"예측된 감성 (0: 부정, 1: 긍정): {'Positive' if prediction[0][0] > 0.5 else 'Negative'} (Raw: {prediction[0][0]:.4f})")
```

## 4. 추가 고려 사항
- **하이퍼파라미터 튜닝**: 임베딩 차원, LSTM 유닛 수, 드롭아웃 비율, 학습률, 배치 크기, 에포크 수 등을 조절하여 성능을 최적화할 수 있습니다.
- **과적합 방지**: 드롭아웃(Dropout), 규제(Regularization), 조기 종료(Early Stopping) 등을 사용할 수 있습니다.
- **모델 복잡도**: 모델이 너무 복잡하면 과적합되기 쉽고, 너무 단순하면 충분한 성능을 내지 못할 수 있습니다.
- **사전 훈련된 임베딩 사용**: `GloVe`, `FastText`, `Word2Vec` 등 사전 훈련된 단어 임베딩을 사용하면 적은 데이터로도 좋은 성능을 얻는 데 도움이 될 수 있습니다. (Embedding 레이어의 `weights` 파라미터 사용)
- **어텐션 메커니즘 (Attention Mechanism)**: 긴 시퀀스에서 중요한 부분에 더 집중하여 성능을 향상시킬 수 있는 기법. (더 고급 주제)
- **트랜스포머 모델 (Transformer Models)**: BERT, GPT와 같은 트랜스포머 기반 모델들은 현재 많은 NLP 작업에서 SOTA(State-of-the-art) 성능을 보입니다.

## 추가 학습 자료
- [TensorFlow Text classification tutorial](https://www.tensorflow.org/tutorials/text/text_classification_rnn)
- [Keras IMDB LSTM 예제](https://keras.io/examples/nlp/imdb_lstm/)
- [딥 러닝을 이용한 자연어 처리 입문 - IMDB 리뷰 감성 분류하기(LSTM)](https://wikidocs.net/22933)

## 다음 학습 내용
- Day 70: 강화 학습 및 NLP 개념 복습 (Review of RL and NLP concepts)
