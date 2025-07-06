# Day 62: Bag-of-Words 및 TF-IDF

## 학습 목표
- 텍스트 데이터를 수치 벡터로 변환하는 기법 학습
- Bag-of-Words (BoW) 모델의 개념과 구현 방법 이해
- TF-IDF (Term Frequency-Inverse Document Frequency)의 개념과 계산 방법, 중요성 이해

## 1. 텍스트 표현 (Text Representation)의 필요성
- 머신러닝 알고리즘은 대부분 숫자 입력을 가정합니다.
- 따라서, 텍스트 데이터를 분석하고 모델링하기 위해서는 텍스트를 수치적인 형태로 변환하는 과정이 필요합니다. 이를 텍스트 표현 또는 특징 벡터화(Feature Vectorization)라고 합니다.

## 2. Bag-of-Words (BoW)

### 가. 개념
- 가장 간단하면서도 널리 사용되는 텍스트 표현 방법 중 하나입니다.
- 문서를 단어들의 "가방"으로 간주합니다. 즉, 단어의 순서나 문맥은 무시하고, 문서 내 각 단어의 출현 빈도수(Term Frequency)에만 집중합니다.
- 각 문서는 고유한 단어들의 집합(어휘, Vocabulary)을 기준으로, 해당 단어가 문서에 몇 번 등장했는지를 나타내는 벡터로 표현됩니다.

### 나. BoW 구축 과정
1.  **토큰화 (Tokenization)**: 각 문서를 단어(토큰) 단위로 분리합니다.
2.  **어휘 구축 (Vocabulary Building)**: 전체 문서 집합(Corpus)에 등장하는 모든 고유한 단어들의 집합을 만듭니다. 이 어휘집이 벡터의 차원이 됩니다.
3.  **벡터화 (Vectorization)**: 각 문서를 어휘집의 단어 순서대로 정렬된 벡터로 표현합니다. 벡터의 각 요소는 해당 단어가 해당 문서에 등장한 횟수(빈도수)를 나타냅니다.

### 다. 예시
- 문서 1: "John likes to watch movies. Mary likes movies too."
- 문서 2: "John also likes to watch football games."

1.  **토큰화 및 정제 (소문자화, 구두점 제거 등 가정)**
    - 문서 1 토큰: `["john", "likes", "to", "watch", "movies", "mary", "likes", "movies", "too"]`
    - 문서 2 토큰: `["john", "also", "likes", "to", "watch", "football", "games"]`

2.  **어휘 구축**
    - 전체 어휘: `{"john", "likes", "to", "watch", "movies", "mary", "too", "also", "football", "games"}`
    - (정렬된 어휘) -> `["also", "football", "games", "john", "likes", "mary", "movies", "to", "too", "watch"]` (10차원 벡터)

3.  **벡터화 (단어 빈도수 기준)**
    - 문서 1 벡터: `[0, 0, 0, 1, 2, 1, 2, 1, 1, 1]`
        - (also:0, football:0, games:0, john:1, likes:2, mary:1, movies:2, to:1, too:1, watch:1)
    - 문서 2 벡터: `[1, 1, 1, 1, 1, 0, 0, 1, 0, 1]`
        - (also:1, football:1, games:1, john:1, likes:1, mary:0, movies:0, to:1, too:0, watch:1)

### 라. 장점
- 구현이 간단하고 이해하기 쉽습니다.
- 텍스트 분류, 정보 검색 등 다양한 NLP 작업에서 기본적으로 사용될 수 있습니다.

### 마. 단점
- **단어 순서 무시**: 문맥 정보를 잃어버려 의미 파악에 한계가 있습니다. (예: "I hate you" vs "You hate I"는 동일한 BoW 벡터를 가질 수 있음)
- **희소성 문제 (Sparsity)**: 어휘집의 크기가 매우 커지면 대부분의 값이 0인 희소 벡터(Sparse Vector)가 생성되어 계산 비효율성 및 성능 저하를 유발할 수 있습니다.
- **불용어 문제**: "the", "a"와 같이 자주 등장하지만 의미는 적은 단어들이 높은 빈도수를 가질 수 있습니다. (TF-IDF로 일부 보완 가능)
- **단어의 의미적 유사성 반영 불가**: "car"와 "automobile"은 다른 단어로 취급됩니다.

### 바. 파이썬 구현
- `scikit-learn`의 `CountVectorizer` 사용

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    'John likes to watch movies. Mary likes movies too.',
    'John also likes to watch football games.'
]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names_out())
# ['also', 'football', 'games', 'john', 'likes', 'mary', 'movies', 'to', 'too', 'watch']
print(X.toarray())
# [[0 1 1 1 2 1 2 1 1]  <- 오타 수정: 실제 출력은 위 예시와 같아야 함
#  [1 1 1 1 1 0 0 1 0 1]]
```

## 3. TF-IDF (Term Frequency - Inverse Document Frequency)

### 가. 개념
- BoW의 단점을 보완하기 위해 등장한 가중치 부여 방식입니다.
- 단순히 단어의 빈도수만 고려하는 것이 아니라, 특정 단어가 한 문서에 얼마나 자주 등장하는지(TF)와 함께, 그 단어가 전체 문서 집합에서 얼마나 희귀한지(IDF)를 고려하여 단어의 중요도를 계산합니다.
- 즉, 한 문서 내에서는 자주 등장하지만 다른 여러 문서에서는 잘 등장하지 않는 단어일수록 중요도가 높다고 판단합니다.

### 나. TF (Term Frequency, 단어 빈도)
- 특정 단어가 특정 문서 내에 얼마나 자주 등장하는지를 나타내는 값입니다.
- 계산 방법은 다양하지만, 가장 기본적인 방법은 해당 단어의 등장 횟수입니다.
- `TF(t, d)` = (문서 d에서 단어 t의 등장 횟수) / (문서 d의 전체 단어 수) (정규화된 TF)
- 또는 단순히 `TF(t, d)` = (문서 d에서 단어 t의 등장 횟수)

### 다. IDF (Inverse Document Frequency, 역문서 빈도)
- 특정 단어가 전체 문서 집합에서 얼마나 드물게 등장하는지를 나타내는 값입니다.
- 이 값이 클수록 해당 단어는 특정 문서의 주제를 잘 나타내는 희귀한 단어일 가능성이 높습니다.
- `IDF(t, D)` = log ( (전체 문서 수) / (단어 t를 포함하는 문서 수 + 1) )
    - 분모에 1을 더하는 것은 특정 단어가 모든 문서에 등장하지 않아 0으로 나누어지는 것을 방지하기 위함입니다 (Smoothing).
    - 로그를 취하는 이유는 문서 수의 차이가 극심할 때 값의 스케일을 줄여주기 위함입니다.

### 라. TF-IDF 계산
- `TF-IDF(t, d, D) = TF(t, d) * IDF(t, D)`
- TF-IDF 값은 특정 문서 d 내에서 특정 단어 t의 중요도를 나타냅니다.
- 이 값을 사용하여 BoW 벡터를 구성하면, 단순히 빈도수만 사용했을 때보다 단어의 중요도를 더 잘 반영하는 텍스트 표현을 얻을 수 있습니다.

### 마. 장점
- BoW에 비해 단어의 중요도를 더 잘 반영할 수 있습니다.
- 모든 문서에 자주 등장하는 불용어의 영향력을 줄일 수 있습니다 (IDF 값이 낮아짐).

### 바. 단점
- 여전히 단어의 순서나 문맥 정보를 반영하지 못합니다.
- 단어의 의미적 유사성은 고려하지 못합니다.

### 사. 파이썬 구현
- `scikit-learn`의 `TfidfVectorizer` 사용

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    'John likes to watch movies. Mary likes movies too.',
    'John also likes to watch football games.'
]
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(corpus)

print(tfidf_vectorizer.get_feature_names_out())
print(X_tfidf.toarray())
```

## 추가 학습 자료
- [Scikit-learn: Text feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [딥 러닝을 이용한 자연어 처리 입문 - TF-IDF](https://wikidocs.net/31698)

## 다음 학습 내용
- Day 63: 단어 임베딩 - Word2Vec, GloVe (개념) (Word Embeddings - Word2Vec, GloVe (Conceptual))
