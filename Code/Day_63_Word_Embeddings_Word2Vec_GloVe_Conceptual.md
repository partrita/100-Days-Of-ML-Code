# Day 63: 단어 임베딩 - Word2Vec, GloVe (개념) (Word Embeddings - Word2Vec, GloVe (Conceptual))

## 학습 목표
- 기존 텍스트 표현 방식(BoW, TF-IDF)의 한계점 재확인
- 단어 임베딩(Word Embedding)의 개념과 필요성 이해
- 분산 표현(Distributed Representation)의 의미 학습
- 대표적인 단어 임베딩 모델인 Word2Vec과 GloVe의 기본 아이디어 및 특징 비교

## 1. 기존 텍스트 표현 방식의 한계
- **BoW, TF-IDF**:
    - **희소 표현 (Sparse Representation)**: 어휘집 크기가 커지면 벡터가 매우 희소해지고 차원이 커집니다.
    - **단어 의미 반영 부족**: 단어 간의 의미적 유사성이나 관계를 파악하기 어렵습니다. (예: "강아지"와 "개"는 서로 다른 단어로 취급)
    - **문맥 정보 손실**: 단어의 순서나 주변 단어와의 관계를 고려하지 않습니다.

## 2. 단어 임베딩 (Word Embedding)이란?
- 단어를 고정된 크기의 **저차원 실수 벡터(Dense Vector)**로 표현하는 기법입니다.
- 각 단어는 벡터 공간의 한 점으로 매핑되며, 이 벡터를 **임베딩 벡터(Embedding Vector)**라고 합니다.
- 핵심 아이디어: **"비슷한 의미를 가진 단어는 벡터 공간에서 서로 가까이 위치한다."**
- 단어의 의미를 벡터에 함축적으로 담아내려고 시도합니다.

### 분산 표현 (Distributed Representation)
- 단어 임베딩은 단어의 의미를 여러 차원에 분산하여 표현합니다.
- 벡터의 각 차원이 특정 의미적 또는 문법적 특징을 나타내는 것은 아니지만, 전체적으로 단어의 의미를 포착합니다.
- 예: "king" - "man" + "woman" ≈ "queen" 과 같은 단어 간의 의미론적 관계를 벡터 연산으로 표현 가능.

### 단어 임베딩의 장점
- **차원 축소**: 고차원의 희소 벡터 대신 저차원의 밀집 벡터를 사용하므로 계산 효율성이 높습니다.
- **의미적 유사도 반영**: 의미가 유사한 단어들은 벡터 공간에서 가까운 거리에 위치하게 되어 단어 간 관계를 파악할 수 있습니다.
- **일반화 성능 향상**: 모델이 학습 데이터에 없는 단어와 유사한 의미의 단어를 처리하는 데 도움을 줄 수 있습니다.
- **다양한 NLP 작업 성능 향상**: 텍스트 분류, 기계 번역, 감성 분석 등에서 성능 향상을 가져옵니다.

## 3. Word2Vec (Word to Vector)

### 가. 기본 아이디어
- 2013년 Google의 Mikolov 등이 제안한 신경망 기반의 단어 임베딩 모델입니다.
- **"주변 단어가 비슷하면 해당 단어의 의미도 비슷할 것이다"** 라는 분포 가설(Distributional Hypothesis)에 기반합니다.
- 특정 단어 주변에 나타나는 단어들을 예측하거나, 주변 단어들로부터 특정 단어를 예측하는 과정에서 단어의 임베딩 벡터를 학습합니다.

### 나. 주요 모델 구조
1.  **CBOW (Continuous Bag-of-Words)**:
    - 주변 단어들(Context Words)이 주어졌을 때, 중심 단어(Center Word)를 예측하는 모델입니다.
    - 여러 주변 단어의 임베딩 벡터를 사용하여 중심 단어의 임베딩 벡터를 예측합니다.
    - 작은 데이터셋에서 성능이 좋고 학습 속도가 빠릅니다.

    ![CBOW Model](https://wikidocs.net/images/page/22660/cbow_image.PNG)
    *(이미지 출처: Wikidocs)*

2.  **Skip-gram**:
    - 중심 단어(Center Word)가 주어졌을 때, 주변 단어들(Context Words)을 예측하는 모델입니다.
    - 하나의 중심 단어 임베딩 벡터를 사용하여 여러 주변 단어의 임베딩 벡터를 예측합니다.
    - 일반적으로 CBOW보다 성능이 우수하며, 특히 등장 빈도가 낮은 단어에 대해 더 잘 학습합니다.
    - 학습 시간이 CBOW보다 오래 걸립니다.

    ![Skip-gram Model](https://wikidocs.net/images/page/22660/skipgram_image.PNG)
    *(이미지 출처: Wikidocs)*

### 다. 학습 방식 (간략히)
- 입력층, 은닉층(투사층, Projection Layer), 출력층으로 구성된 얕은 신경망(Shallow Neural Network)을 사용합니다.
- 은닉층의 가중치가 바로 단어의 임베딩 벡터가 됩니다.
- 실제로는 계산 효율성을 위해 Negative Sampling이나 Hierarchical Softmax와 같은 최적화 기법을 사용합니다.

## 4. GloVe (Global Vectors for Word Representation)

### 가. 기본 아이디어
- 2014년 Stanford 대학에서 제안한 단어 임베딩 모델입니다.
- Word2Vec과 달리, 전체 말뭉치(Corpus)의 **단어 동시 등장 빈도 행렬(Word Co-occurrence Matrix)** 정보를 직접 사용하여 단어 임베딩을 학습합니다.
- "단어 간의 동시 등장 빈도 비율이 의미적 관계를 나타낸다"는 직관에 기반합니다.
- 예: "ice"는 "solid"와 자주 함께 등장하지만 "gas"와는 덜 등장하고, "steam"은 "gas"와 자주 함께 등장하지만 "solid"와는 덜 등장합니다. 이러한 비율 관계를 학습합니다.

### 나. 학습 방식 (간략히)
1.  **단어 동시 등장 빈도 행렬 구축**:
    - 전체 말뭉치에서 특정 윈도우 크기 내에 두 단어가 함께 등장한 횟수를 계산하여 행렬을 만듭니다.
    - X<sub>ij</sub>: 단어 i의 문맥에 단어 j가 등장한 횟수.

2.  **손실 함수 설계**:
    - 임베딩된 단어 벡터 간의 내적(Dot Product)이 두 단어의 동시 등장 빈도 로그값과 유사해지도록 학습합니다.
    - w<sub>i</sub><sup>T</sup>w̃<sub>j</sub> + b<sub>i</sub> + b̃<sub>j</sub> ≈ log(X<sub>ij</sub>)
        - w<sub>i</sub>: 중심 단어 i의 임베딩 벡터
        - w̃<sub>j</sub>: 주변 단어 j의 임베딩 벡터 (별도의 임베딩 행렬 사용)
        - b<sub>i</sub>, b̃<sub>j</sub>: 각 단어의 편향(bias) 항
    - 가중치 함수 f(X<sub>ij</sub>)를 사용하여 등장 빈도가 낮은 단어에 과도하게 가중치가 부여되거나, 매우 높은 단어에 과도하게 가중치가 부여되는 것을 방지합니다.

### 다. 특징
- **전역적 통계 정보 활용**: 말뭉치 전체의 통계 정보를 직접적으로 활용하여 학습합니다.
- **학습 속도**: Word2Vec보다 학습 속도가 빠를 수 있습니다.
- **성능**: 종종 Word2Vec과 유사하거나 더 나은 성능을 보이며, 특히 작은 데이터셋에서 강점을 가질 수 있습니다.

## 5. Word2Vec vs GloVe

| 특징             | Word2Vec (Skip-gram)                                  | GloVe                                                     |
| ---------------- | ----------------------------------------------------- | --------------------------------------------------------- |
| **기본 아이디어**  | 지역적 문맥 정보 (Local context window)               | 전역적 동시 등장 통계 (Global co-occurrence counts)         |
| **학습 방식**    | 주변 단어 예측 (Predictive)                             | 동시 등장 행렬 분해 (Count-based / Matrix Factorization)    |
| **장점**         | 다양한 크기의 데이터셋에서 잘 작동, 의미론적 관계 포착 우수 | 학습 속도가 빠를 수 있음, 전역적 통계 정보 명시적 활용        |
| **단점**         | 말뭉치 전체 통계 정보를 간접적으로만 활용, 학습 시간 소요   | 큰 메모리 필요 (동시 등장 행렬), 하이퍼파라미터 튜닝 민감 가능 |

- 실제로는 두 모델 모두 좋은 성능을 보이며, 문제나 데이터셋의 특성에 따라 선택적으로 사용되거나, 사전 훈련된 임베딩(Pre-trained Embeddings)을 활용하는 경우가 많습니다.

## 추가 학습 자료
- [Word2Vec Tutorial - The Skip-Gram Model (Chris McCormick)](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
- [GloVe: Global Vectors for Word Representation (Project Page)](https://nlp.stanford.edu/projects/glove/)
- [딥 러닝을 이용한 자연어 처리 입문 - Word2Vec, GloVe](https://wikidocs.net/22660)

## 다음 학습 내용
- Day 64: Word2Vec 구현 (Implementing Word2Vec (e.g., using Gensim))
