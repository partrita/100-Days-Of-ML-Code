# Day 66: 간단한 감성 분석기 구축 (Building a simple Sentiment Analyzer)

## 학습 목표
- 감성 사전을 이용하여 간단한 규칙 기반 감성 분석기 구현
- 머신러닝 기반 감성 분석기 (TF-IDF + Logistic Regression) 구현
- `scikit-learn`을 활용한 텍스트 분류 모델 학습 및 평가

## 1. 감성 사전 기반 감성 분석기 구현

### 가. 사용할 감성 사전
- 여기서는 간단하게 직접 작은 규모의 긍정/부정 단어 사전을 정의하여 사용합니다.
- 실제로는 AFINN, SentiWordNet, KNU 한국어 감성사전 등을 활용할 수 있습니다.

### 나. 구현 단계
1.  **간단한 감성 사전 정의**: 긍정 단어와 부정 단어 리스트를 만듭니다.
2.  **텍스트 입력 및 전처리**: 분석할 텍스트를 입력받고, 기본적인 전처리(토큰화, 소문자화 등)를 수행합니다.
3.  **감성 점수 계산**:
    *   텍스트 내 각 토큰이 긍정 사전에 있으면 +1, 부정 사전에 있으면 -1을 부여합니다.
    *   (선택적) 부정어 처리: "not"과 같은 부정어가 긍정 단어 앞에 오면 점수를 반전시킵니다.
4.  **최종 감성 판단**: 계산된 총 점수를 기준으로 긍정, 부정, 중립을 판단합니다.

### 다. 파이썬 코드 예시 (간단한 사전 기반)

```python
# 1. 간단한 감성 사전 정의
positive_words = ['good', 'great', 'awesome', 'happy', 'love', 'excellent', 'nice', 'wonderful', 'best']
negative_words = ['bad', 'terrible', 'awful', 'sad', 'hate', 'poor', 'worst', 'horrible']

# 간단한 부정어 리스트
negation_words = ['not', 'no', 'never']

def simple_lexicon_sentiment(text):
    text = text.lower()
    words = text.split() # 간단한 공백 기준 토큰화

    score = 0
    negation_active = False

    for i, word in enumerate(words):
        # 부정어 처리 (다음 단어에 영향)
        if word in negation_words:
            negation_active = True
            continue

        current_word_score = 0
        if word in positive_words:
            current_word_score = 1
        elif word in negative_words:
            current_word_score = -1

        if negation_active:
            current_word_score *= -1
            negation_active = False # 부정어 효과는 한 단어에만 적용 (간단화)

        score += current_word_score

    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"

# 테스트
review1 = "This movie is very good and I love it"
review2 = "The food was terrible and the service was poor"
review3 = "It's not bad, actually quite nice" # 부정어 처리 테스트
review4 = "This is a book." # 중립 테스트

print(f"'{review1}' -> Sentiment: {simple_lexicon_sentiment(review1)}")
print(f"'{review2}' -> Sentiment: {simple_lexicon_sentiment(review2)}")
print(f"'{review3}' -> Sentiment: {simple_lexicon_sentiment(review3)}")
print(f"'{review4}' -> Sentiment: {simple_lexicon_sentiment(review4)}")
```

### 라. 한계점
- 위 예시는 매우 단순하며, 실제 감성 분석에는 부족한 점이 많습니다.
    - 문맥, 비유, 신조어 처리 불가
    - 감성 강도 표현 미흡
    - 제한된 어휘

## 2. 머신러닝 기반 감성 분석기 구현

### 가. 데이터 준비
- 레이블링된 데이터셋이 필요합니다 (예: 영화 리뷰와 해당 리뷰의 긍정/부정 레이블).
- 여기서는 간단한 예제 데이터를 직접 만듭니다. 실제로는 IMDB 영화 리뷰 데이터셋, 네이버 영화 리뷰 데이터셋 등을 사용할 수 있습니다.

```python
# 예제 데이터 (텍스트, 레이블) - 레이블: 1 (긍정), 0 (부정)
train_data = [
    ("This is a great movie, I loved it!", 1),
    ("The plot was amazing and the actors were brilliant.", 1),
    ("What a fantastic film, truly inspiring.", 1),
    ("I enjoyed every moment of this masterpiece.", 1),
    ("Absolutely wonderful, a must-see for everyone.", 1),
    ("This movie was terrible, a complete waste of time.", 0),
    ("I hated it, the acting was awful.", 0),
    ("A boring and predictable storyline.", 0),
    ("The worst film I have ever seen.", 0),
    ("I do not recommend this movie to anyone.", 0),
    ("It was okay, not great but not bad either.", 1), # 중립적인 것을 긍정으로 가정 (간단화)
    ("The movie is neither good nor bad.", 0) # 중립적인 것을 부정으로 가정 (간단화)
]

train_texts = [data[0] for data in train_data]
train_labels = [data[1] for data in train_data]
```

### 나. 특징 추출 (TF-IDF)
- `scikit-learn`의 `TfidfVectorizer`를 사용합니다.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', # 불용어 제거 (영어)
                             max_features=1000)   # 최대 특징 수 제한 (어휘 크기)

# 학습 데이터에 대해 TF-IDF 행렬 생성
X_train_tfidf = vectorizer.fit_transform(train_texts)

print("TF-IDF 행렬 크기:", X_train_tfidf.shape)
# print("어휘 일부:", vectorizer.get_feature_names_out()[:20])
```

### 다. 모델 학습 (로지스틱 회귀)
- `scikit-learn`의 `LogisticRegression`을 사용합니다.

```python
from sklearn.linear_model import LogisticRegression

# 로지스틱 회귀 모델 초기화 및 학습
model_lr = LogisticRegression()
model_lr.fit(X_train_tfidf, train_labels)

print("로지스틱 회귀 모델 학습 완료!")
```

### 라. 모델 평가 및 예측
- 간단한 테스트 데이터를 사용하여 예측을 수행합니다.
- 실제로는 별도의 테스트 데이터셋을 사용하고, 정확도, 정밀도, 재현율, F1 점수 등으로 평가해야 합니다.

```python
# 테스트 데이터
test_data = [
    "I really liked this film, it was fantastic!", # 예상: 긍정 (1)
    "A truly awful experience, I would not watch it again.", # 예상: 부정 (0)
    "The movie was not good at all.", # 예상: 부정 (0)
    "It was an average movie." # 예상: ? (데이터 및 모델에 따라 다름)
]

# 테스트 데이터에 대해 TF-IDF 변환 (학습 시 사용한 vectorizer 사용)
X_test_tfidf = vectorizer.transform(test_data)

# 예측
predictions = model_lr.predict(X_test_tfidf)
predicted_labels = ["Positive" if p == 1 else "Negative" for p in predictions]

for text, label in zip(test_data, predicted_labels):
    print(f"'{text}' -> Predicted Sentiment: {label}")

# (선택) 예측 확률 확인
# probabilities = model_lr.predict_proba(X_test_tfidf)
# for text, prob in zip(test_data, probabilities):
#     print(f"'{text}' -> Probabilities (Neg, Pos): {prob}")
```

### 마. 전체 코드 흐름 (머신러닝 기반)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split # 데이터 분할용
from sklearn.metrics import accuracy_score # 평가용

# 1. 데이터 준비 (더 많은 데이터가 필요함)
texts = [
    "This is a great movie, I loved it!", "The plot was amazing and the actors were brilliant.",
    "What a fantastic film, truly inspiring.", "I enjoyed every moment of this masterpiece.",
    "Absolutely wonderful, a must-see for everyone.", "This movie was terrible, a complete waste of time.",
    "I hated it, the acting was awful.", "A boring and predictable storyline.",
    "The worst film I have ever seen.", "I do not recommend this movie to anyone.",
    "It was okay, not great but not bad either.", "The movie is neither good nor bad.",
    "An excellent story with powerful performances.", "Just a so-so movie, nothing special.",
    "I was deeply disappointed by this film.", "Two thumbs up for this amazing flick!"
]
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1] # 1: Positive, 0: Negative

# 2. 데이터 분할 (학습용, 테스트용)
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.25, random_state=42)

# 3. 특징 추출 (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test) # 테스트 데이터는 학습된 vectorizer로 transform만 수행

# 4. 모델 학습 (로지스틱 회귀)
model_lr = LogisticRegression()
model_lr.fit(X_train_tfidf, y_train)

# 5. 모델 평가
y_pred = model_lr.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n모델 정확도 (Accuracy on Test Set): {accuracy:.4f}")

# 6. 새로운 텍스트 예측
new_reviews = [
    "This is the best movie I have seen in years!",
    "What a waste of money and time."
]
new_reviews_tfidf = vectorizer.transform(new_reviews)
new_predictions = model_lr.predict(new_reviews_tfidf)
new_predicted_labels = ["Positive" if p == 1 else "Negative" for p in new_predictions]

for review, label in zip(new_reviews, new_predicted_labels):
    print(f"New Review: '{review}' -> Predicted Sentiment: {label}")
```

## 3. 고려 사항 및 개선 방향
- **데이터**: 더 많고 다양한 학습 데이터가 필요합니다.
- **전처리**: 더 정교한 전처리(표제어 추출, 특수문자 처리 등)를 적용할 수 있습니다.
- **특징 공학**: N-gram, 단어 임베딩(Word2Vec, FastText) 등을 특징으로 사용할 수 있습니다.
- **모델 선택**: Naive Bayes, SVM, RandomForest 등 다른 분류 모델을 시도해볼 수 있습니다.
- **하이퍼파라미터 튜닝**: `GridSearchCV` 등을 사용하여 모델의 하이퍼파라미터를 최적화할 수 있습니다.
- **딥러닝 모델**: LSTM, CNN, BERT와 같은 딥러닝 모델을 사용하면 더 높은 성능을 기대할 수 있습니다 (더 많은 데이터와 계산 자원 필요).
- **한국어**: 한국어의 경우 KoNLPy를 사용한 형태소 분석 및 적절한 토큰화가 필수적입니다.

## 추가 학습 자료
- [Scikit-learn Tutorial: Working With Text Data](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
- [IMDB 영화 리뷰 감성 분석 (캐글 노트북 예시 다수)](https://www.kaggle.com/c/word2vec-nlp-tutorial/notebooks) (Word2Vec 사용 예시도 포함)

## 다음 학습 내용
- Day 67: 순환 신경망 (RNN) 소개 (Introduction to Recurrent Neural Networks (RNNs))
