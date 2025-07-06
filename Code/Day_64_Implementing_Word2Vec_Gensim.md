# Day 64: Word2Vec 구현 (Implementing Word2Vec (e.g., using Gensim))

## 학습 목표
- 파이썬 라이브러리 `Gensim`을 사용하여 Word2Vec 모델을 학습시키는 방법 이해
- Word2Vec 모델의 주요 하이퍼파라미터 설정 및 의미 파악
- 학습된 임베딩 벡터를 확인하고 단어 간 유사도를 측정하는 방법 학습

## 1. Gensim 라이브러리 소개
- `Gensim`은 토픽 모델링(Topic Modeling)과 자연어 처리(NLP)를 위한 강력하고 효율적인 파이썬 라이브러리입니다.
- Word2Vec, FastText, Doc2Vec 등 다양한 단어 및 문서 임베딩 알고리즘을 쉽게 구현하고 사용할 수 있도록 지원합니다.
- 대용량 텍스트 데이터 처리에 최적화되어 있습니다.

### Gensim 설치
```bash
pip install gensim
```
필요에 따라 `nltk` (토큰화 등 전처리)도 함께 설치합니다.
```bash
pip install nltk
```

## 2. Word2Vec 학습 과정 (Gensim 사용)

### 가. 데이터 준비 및 전처리
- Word2Vec 모델을 학습시키기 위해서는 토큰화된 문장들의 리스트가 필요합니다.
- 각 문장은 단어(토큰)들의 리스트 형태여야 합니다.
- 예: `sentences = [['this', 'is', 'the', 'first', 'sentence'], ['this', 'is', 'the', 'second', 'sentence']]`

```python
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
# nltk.download('punkt') # 처음 실행 시 필요할 수 있음

# 예제 텍스트 데이터 (실제로는 더 큰 코퍼스를 사용)
corpus_text = """
Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence.
It is concerned with the interactions between computers and human language.
In particular, how to program computers to process and analyze large amounts of natural language data.
Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural language generation.
"""

# 1. 문장 분리
raw_sentences = sent_tokenize(corpus_text)
# print(raw_sentences)

# 2. 각 문장을 단어 리스트로 토큰화 및 소문자화 (간단한 전처리)
sentences = []
for raw_sentence in raw_sentences:
    tokens = [word.lower() for word in word_tokenize(raw_sentence) if word.isalpha()] # 알파벳만 남기고 소문자화
    if tokens: # 빈 리스트가 아닐 경우에만 추가
        sentences.append(tokens)

print("전처리된 문장 리스트:")
for s in sentences:
    print(s)
# 출력 예시:
# ['natural', 'language', 'processing', 'nlp', 'is', 'a', 'subfield', 'of', 'linguistics', 'computer', 'science', 'and', 'artificial', 'intelligence']
# ['it', 'is', 'concerned', 'with', 'the', 'interactions', 'between', 'computers', 'and', 'human', 'language']
# ...
```

### 나. Word2Vec 모델 학습
- `gensim.models.Word2Vec` 클래스를 사용합니다.

```python
from gensim.models import Word2Vec

# Word2Vec 모델 학습
# 주요 하이퍼파라미터:
# - sentences: 학습에 사용할 토큰화된 문장 리스트
# - vector_size: 임베딩 벡터의 차원 (예: 100, 200, 300)
# - window: 컨텍스트 윈도우 크기 (중심 단어 기준 앞뒤로 고려할 단어 수)
# - min_count: 모델 학습에 포함할 단어의 최소 등장 빈도 (이 값보다 적게 등장한 단어는 무시)
# - workers: 학습에 사용할 CPU 코어 수 (병렬 처리)
# - sg: 학습 알고리즘 선택 (0: CBOW, 1: Skip-gram)
# - hs: (0: Negative Sampling, 1: Hierarchical Softmax) - Negative Sampling이 주로 사용됨
# - negative: Negative Sampling 시 사용할 '노이즈 단어' 수 (보통 5-20)
# - epochs: 전체 데이터셋에 대한 학습 반복 횟수

model = Word2Vec(sentences=sentences,
                 vector_size=100,  # 임베딩 벡터 차원
                 window=5,         # 컨텍스트 윈도우 크기
                 min_count=1,      # 최소 단어 빈도
                 workers=4,        # CPU 코어 수
                 sg=0,             # 0: CBOW, 1: Skip-gram (여기서는 CBOW 사용)
                 epochs=10)        # 학습 반복 횟수

print("Word2Vec 모델 학습 완료!")
```

### 다. 학습된 모델 활용
- **어휘 확인**: `model.wv.key_to_index` (단어와 인덱스 매핑), `model.wv.index_to_key` (인덱스와 단어 매핑)
- **특정 단어의 임베딩 벡터 확인**: `model.wv['단어']`
- **단어 간 유사도 계산**: `model.wv.similarity('단어1', '단어2')`
- **가장 유사한 단어 찾기**: `model.wv.most_similar('단어', topn=5)`
- **긍정/부정 단어를 사용한 유추**: `model.wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)` (결과: queen)

```python
# 어휘 확인 (일부)
# print("어휘:", list(model.wv.key_to_index.keys())[:10])
# print("첫 번째 단어:", model.wv.index_to_key[0])

# 특정 단어의 임베딩 벡터 확인
try:
    vector_nlp = model.wv['nlp']
    print("\n'nlp'의 임베딩 벡터 (일부):", vector_nlp[:5])
    print("'nlp'의 임베딩 벡터 차원:", len(vector_nlp))
except KeyError:
    print("\n'nlp' 단어가 어휘에 없습니다.")


# 단어 간 유사도 계산
try:
    similarity_score = model.wv.similarity('language', 'natural')
    print(f"\n'language'와 'natural'의 유사도: {similarity_score:.4f}")
except KeyError as e:
    print(f"\n유사도 계산 오류: 단어 '{e.args[0]}'가 어휘에 없습니다.")

try:
    similarity_score_diff = model.wv.similarity('computer', 'human')
    print(f"'computer'와 'human'의 유사도: {similarity_score_diff:.4f}")
except KeyError as e:
    print(f"유사도 계산 오류: 단어 '{e.args[0]}'가 어휘에 없습니다.")


# 가장 유사한 단어 찾기
try:
    similar_words = model.wv.most_similar('processing', topn=3)
    print("\n'processing'과 가장 유사한 단어들:", similar_words)
except KeyError:
    print("\n'processing' 단어가 어휘에 없습니다.")

# 유추 (데이터가 작아 의미있는 결과가 안 나올 수 있음)
try:
    # 예시: 'nlp' + 'language' - 'computer' 와 유사한 단어? (의미는 없을 수 있음)
    analogy = model.wv.most_similar(positive=['nlp', 'language'], negative=['computer'], topn=1)
    print("\n유추 ('nlp' + 'language' - 'computer'):", analogy)
except KeyError as e:
    print(f"\n유추 오류: 단어 '{e.args[0]}'가 어휘에 없습니다.")
```

### 라. 모델 저장 및 로드
- 학습된 모델은 저장해두고 나중에 다시 불러와 사용할 수 있습니다.

```python
# 모델 저장
model.save("word2vec_gensim.model")
print("\n모델 저장 완료: word2vec_gensim.model")

# 모델 로드
# loaded_model = Word2Vec.load("word2vec_gensim.model")
# print("\n모델 로드 완료!")
# vector_nlp_loaded = loaded_model.wv['nlp']
# print("'nlp'의 로드된 임베딩 벡터 (일부):", vector_nlp_loaded[:5])
```

## 3. 주요 하이퍼파라미터 설명

-   `vector_size`: 임베딩 벡터의 차원 수. 일반적으로 50~300 사이의 값을 사용. 차원이 클수록 더 많은 정보를 담을 수 있지만, 학습 데이터가 충분하지 않으면 과적합될 수 있고 계산량이 증가.
-   `window`: 중심 단어를 기준으로 앞뒤로 몇 개의 단어까지를 문맥(context)으로 간주할지 결정. Skip-gram에서는 중심 단어로부터 예측할 주변 단어의 최대 거리. CBOW에서는 중심 단어를 예측하기 위해 사용될 주변 단어들의 범위. 보통 2~10 사이의 값을 사용.
-   `min_count`: 학습에 사용할 단어의 최소 등장 빈도. 이 값보다 적게 등장한 단어는 어휘에서 제외. 희귀 단어를 무시하여 노이즈를 줄이고 학습 속도를 높일 수 있음. 기본값은 5.
-   `sg`: `0`이면 CBOW 모델을 사용하고, `1`이면 Skip-gram 모델을 사용. Skip-gram이 일반적으로 더 좋은 성능을 보이지만 학습 시간이 오래 걸림.
-   `workers`: 학습 시 사용할 CPU 스레드 수. 멀티코어 환경에서 학습 속도를 높일 수 있음.
-   `epochs`: 전체 학습 데이터셋에 대한 반복 학습 횟수. 너무 작으면 충분히 학습되지 않고, 너무 크면 과적합되거나 학습 시간이 오래 걸림. Gensim 4.0.0부터 `iter` 대신 `epochs` 사용.
-   `hs` (Hierarchical Softmax): `1`이면 계층적 소프트맥스를 사용하고, `0`이면 사용하지 않음. `negative` 파라미터와 함께 사용되며, 둘 중 하나만 선택해야 함.
-   `negative` (Negative Sampling): `hs=0`일 때 사용. 0보다 큰 값을 설정하면 네거티브 샘플링을 사용. 얼마나 많은 "노이즈 단어"를 네거티브 샘플로 사용할지 지정. Skip-gram의 경우 5-20, CBOW의 경우 2-5 정도의 값이 권장됨.
-   `alpha`: 초기 학습률(learning rate).
-   `min_alpha`: 학습률이 선형적으로 감소하여 도달하는 최소 학습률.

## 4. 사전 훈련된 Word2Vec 모델 (Pre-trained Word2Vec Models)
- 대규모 말뭉치(예: Google News, Wikipedia)로 미리 학습된 Word2Vec 모델을 다운로드하여 사용할 수도 있습니다.
- 이러한 모델은 이미 풍부한 의미 정보를 담고 있어, 특정 작업에 대한 전이 학습(Transfer Learning)에 유용합니다.
- Gensim의 `gensim.downloader` 모듈을 통해 다양한 사전 훈련된 모델을 쉽게 로드할 수 있습니다.

```python
# import gensim.downloader as api

# 사전 훈련된 모델 목록 보기
# print(list(api.info()['models'].keys()))

# 예: 'word2vec-google-news-300' 모델 로드 (시간이 오래 걸리고 용량이 큼)
# word2vec_google_model = api.load('word2vec-google-news-300')
# print("Google News Word2Vec 모델 로드 완료!")
# similarity_king_queen = word2vec_google_model.similarity('king', 'queen')
# print(f"'king'과 'queen'의 유사도 (Google News): {similarity_king_queen:.4f}")
```

## 추가 학습 자료
- [Gensim Word2Vec Tutorial](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html)
- [딥 러닝을 이용한 자연어 처리 입문 - Gensim의 Word2Vec](https://wikidocs.net/50739)

## 다음 학습 내용
- Day 65: 감성 분석 - 기본 기법 (Sentiment Analysis - Basic Techniques)
