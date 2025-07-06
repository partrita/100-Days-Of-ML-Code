# Day 70: 강화 학습 및 NLP 개념 복습 (Review of RL and NLP concepts)

## 학습 목표
- 지난 15일간 학습한 강화 학습(RL) 및 자연어 처리(NLP)의 핵심 개념들을 되짚어보고 정리합니다.
- 각 기술의 주요 용어, 알고리즘, 응용 분야를 상기합니다.
- 향후 더 심도 있는 학습을 위한 기반을 다집니다.

## 1. 강화 학습 (Reinforcement Learning, RL) 복습 (Days 55-59)

### 가. 핵심 개념
- **에이전트(Agent)**: 환경과 상호작용하며 학습하는 주체.
- **환경(Environment)**: 에이전트가 행동하는 외부 세계.
- **상태(State, S)**: 특정 시점에서 환경에 대한 관찰.
- **행동(Action, A)**: 에이전트가 상태에서 취할 수 있는 결정.
- **보상(Reward, R)**: 행동에 대한 환경의 피드백. 에이전트는 누적 보상을 최대화하려 함.
- **정책(Policy, π)**: 상태에 따른 행동 선택 전략. π(a|s).
- **가치 함수(Value Function)**:
    - **상태 가치 함수 (V<sup>π</sup>(s))**: 상태 s의 장기적인 가치.
    - **행동 가치 함수 (Q<sup>π</sup>(s, a))**: 상태 s에서 행동 a를 했을 때의 장기적인 가치.
- **모델(Model)**: 환경의 작동 방식 (상태 전이 확률, 보상 함수).
    - **모델 기반 RL**: 모델을 알거나 학습하는 경우.
    - **모델 프리 RL**: 모델 없이 경험으로부터 직접 학습하는 경우.
- **마르코프 결정 과정 (MDP, Markov Decision Process)**: RL 문제를 수학적으로 정의하는 프레임워크 (S, A, P, R, γ).
    - **마르코프 속성**: 미래는 현재에만 의존.
    - **감가율 (Discount Factor, γ)**: 미래 보상의 현재 가치 반영.
- **벨만 방정식 (Bellman Equation)**: 가치 함수 간의 재귀적 관계 정의.

### 나. 주요 알고리즘 및 기법
- **Q-러닝 (Q-Learning)**:
    - 대표적인 모델 프리, 오프 폴리시 알고리즘.
    - Q-테이블을 사용하여 최적 행동 가치 함수 Q*를 학습.
    - 업데이트 규칙: Q(S,A) ← Q(S,A) + α[R + γ max<sub>a'</sub>Q(S',a') - Q(S,A)]
    - **ε-탐욕 정책**: 탐험(Exploration)과 활용(Exploitation)의 균형.
- **심층 Q-네트워크 (Deep Q-Network, DQN)**:
    - Q-테이블 대신 심층 신경망을 사용하여 Q-함수를 근사 (Q(s, a; θ)).
    - 고차원 상태 공간(예: 이미지) 처리 가능, 일반화 성능 향상.
    - **경험 재현 (Experience Replay)**: 샘플 간 상관관계 감소, 데이터 효율성 증대.
    - **타겟 네트워크 분리 (Separate Target Network)**: 학습 안정성 향상.

### 다. 응용 분야
- 게임 AI (아타리 게임, 바둑 등)
- 로봇 제어
- 자율 주행
- 추천 시스템
- 자원 관리 최적화

## 2. 자연어 처리 (Natural Language Processing, NLP) 복습 (Days 60-69)

### 가. 핵심 개념
- **자연어 처리(NLP)**: 컴퓨터가 인간의 언어를 이해, 해석, 생성하도록 하는 기술.
- **말뭉치(Corpus)**: 분석을 위한 대량의 텍스트 데이터 집합.
- **토큰화(Tokenization)**: 텍스트를 의미 있는 단위(토큰)로 분리 (단어, 형태소 등).
- **정제(Cleaning) 및 정규화(Normalization)**: 불필요한 문자 제거, 대소문자 통일 등.
- **불용어(Stopword)**: 분석에 큰 의미 없는 단어 (예: a, the, 은, 는).
- **어간 추출(Stemming)**: 단어의 어미를 제거하여 어간을 추출.
- **표제어 추출(Lemmatization)**: 단어의 기본형(사전형)을 추출.
- **Bag-of-Words (BoW)**: 단어 순서 무시, 단어 빈도 기반 텍스트 표현.
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: 단어 빈도와 역문서 빈도를 사용하여 단어 중요도 가중치 부여.
- **단어 임베딩(Word Embedding)**: 단어를 저차원 밀집 벡터로 표현. 단어 간 의미 유사성 포착.
    - **분산 표현(Distributed Representation)**.
    - **Word2Vec (CBOW, Skip-gram)**: 신경망 기반, 주변 단어와의 관계를 통해 학습.
    - **GloVe (Global Vectors)**: 단어 동시 등장 빈도 행렬 기반 학습.
- **감성 분석(Sentiment Analysis)**: 텍스트의 주관적 의견(긍정/부정/중립) 분석.
    - **감성 사전(Sentiment Lexicon)** 기반 접근.
    - **머신러닝/딥러닝** 기반 접근.
- **순환 신경망 (RNN, Recurrent Neural Network)**: 순차 데이터 처리에 특화된 신경망. 이전 타임스텝의 정보를 현재에 활용.
    - **장기 의존성 문제(Long-Term Dependency Problem)**: 기울기 소실/폭주로 긴 시퀀스 학습 어려움.
- **LSTM (Long Short-Term Memory)**: RNN의 장기 의존성 문제 해결. 셀 상태와 게이트(망각, 입력, 출력) 사용.
- **GRU (Gated Recurrent Unit)**: LSTM보다 단순화된 구조, 유사한 성능.

### 나. 주요 기법 및 라이브러리
- **텍스트 전처리**: NLTK, KoNLPy (한국어), spaCy.
- **BoW/TF-IDF**: Scikit-learn (`CountVectorizer`, `TfidfVectorizer`).
- **Word2Vec/GloVe**: Gensim, TensorFlow/Keras `Embedding` layer.
- **감성 분석**: VADER, Scikit-learn (분류 모델), TensorFlow/Keras (딥러닝 모델).
- **RNN/LSTM 구현**: TensorFlow/Keras (`SimpleRNN`, `LSTM`, `GRU`, `Bidirectional` layers).
    - **임베딩 레이어**: 정수 인코딩된 단어를 임베딩 벡터로 변환.
    - **패딩(Padding)**: 시퀀스 길이를 동일하게 맞춤.

### 다. 응용 분야
- 기계 번역
- 정보 검색, 검색 엔진
- 텍스트 분류 (스팸 필터, 주제 분류)
- 챗봇, 질의응답 시스템
- 텍스트 요약
- 음성 인식 및 합성
- 개체명 인식

## 3. 향후 학습 방향 제안
- **강화 학습 심화**:
    - 정책 경사(Policy Gradient) 방법 (REINFORCE, A2C, A3C)
    - Actor-Critic 방법
    - 심층 결정론적 정책 경사 (DDPG)
    - 실제 환경 적용 및 프로젝트 (OpenAI Gym, PyBullet 등)
- **자연어 처리 심화**:
    - **어텐션 메커니즘(Attention Mechanism)**: Seq2Seq 모델 성능 향상.
    - **트랜스포머(Transformer)**: BERT, GPT 등 SOTA 모델의 기반.
    - **사전 훈련된 언어 모델 (Pre-trained Language Models)** 활용: 전이 학습.
    - 다양한 NLP Task 심층 학습: 기계 번역, 질의응답, 텍스트 생성 등.
    - 한국어 NLP 특화 처리.
- **두 분야의 결합**:
    - 대화형 AI (RL을 이용한 대화 정책 학습)
    - 자연어 설명을 이해하는 로봇 (NLP + RL)

## 자기 점검 질문
1. MDP의 5가지 구성 요소는 무엇이며, 각 요소는 무엇을 의미하는가?
2. Q-러닝 업데이트 규칙을 설명하고, 각 항의 의미를 설명할 수 있는가?
3. DQN이 전통적인 Q-러닝에 비해 가지는 장점과 이를 가능하게 하는 핵심 기법은 무엇인가?
4. 텍스트 전처리 과정에서 토큰화, 어간 추출, 표제어 추출의 차이점은 무엇인가?
5. TF-IDF는 어떻게 계산되며, 어떤 단어에 높은 가중치를 부여하는가?
6. 단어 임베딩이란 무엇이며, BoW와 비교했을 때 어떤 장점이 있는가? Word2Vec과 GloVe의 기본 아이디어 차이는?
7. RNN의 기본 구조와 장기 의존성 문제란 무엇인가?
8. LSTM은 장기 의존성 문제를 해결하기 위해 어떤 메커니즘을 사용하는가? (셀 상태, 3가지 게이트)
9. 텍스트 분류를 위해 RNN/LSTM 모델을 구축할 때 필요한 주요 레이어들은 무엇인가? (Embedding, RNN/LSTM, Dense)

이러한 질문들에 스스로 답해보면서 지난 학습 내용을 점검하고, 부족한 부분은 다시 한번 해당 날짜의 학습 내용을 참고하여 복습하는 것이 좋습니다.

## 다음 학습 내용 예고 (Advanced Topics and Model Deployment)
- Day 71: 주성분 분석 (PCA) - 이론 및 구현 (Principal Component Analysis (PCA) - Theory and implementation)
- Day 72: 선형 판별 분석 (LDA) (Linear Discriminant Analysis (LDA))
- ... 그리고 모델 평가, 앙상블, 시계열 분석, 모델 배포 등으로 이어집니다.
