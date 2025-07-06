# Day 67: 순환 신경망 (RNN) 소개 (Introduction to Recurrent Neural Networks (RNNs))

## 학습 목표
- 순차 데이터(Sequential Data)의 특징과 기존 신경망의 한계점 이해
- 순환 신경망(RNN)의 기본 구조와 작동 원리 학습
- RNN의 "메모리" 역할과 순환적 연결의 의미 파악
- RNN의 다양한 유형(one-to-many, many-to-one, many-to-many) 소개
- RNN의 주요 응용 분야 및 한계점(장기 의존성 문제) 인식

## 1. 순차 데이터 (Sequential Data)
- **정의**: 시간의 흐름이나 순서에 따라 나타나는 데이터. 각 데이터 포인트는 이전 데이터 포인트와 독립적이지 않고 연관성을 가집니다.
- **예시**:
    - **자연어**: 문장 내 단어들의 순서, 문서 내 문장들의 순서.
    - **음성**: 시간에 따른 음파 신호.
    - **시계열 데이터**: 주가, 날씨 변화, 센서 데이터 등.
    - **DNA 염기서열**.
    - **동영상**: 프레임들의 연속.

### 기존 신경망(DNN, CNN)의 한계
- **DNN (Deep Neural Network) / Feedforward Neural Network**:
    - 입력 데이터의 순서를 고려하지 않습니다. 각 입력은 독립적으로 처리됩니다.
    - 고정된 크기의 입력을 가정합니다. (예: BoW 벡터, 고정 크기 이미지)
    - 순차 데이터의 시간적 의존성을 모델링하기 어렵습니다.
- **CNN (Convolutional Neural Network)**:
    - 주로 이미지와 같이 공간적 구조를 가진 데이터 처리에 강점을 보입니다.
    - 필터를 통해 지역적 특징을 추출하지만, 전체 시퀀스의 장기적인 의존성을 파악하는 데는 한계가 있을 수 있습니다. (1D CNN으로 시퀀스 처리가 가능하지만, RNN과는 다른 방식)

## 2. 순환 신경망 (Recurrent Neural Network, RNN)
- **정의**: 순차 데이터 처리에 특화된 인공 신경망의 한 종류입니다.
- **핵심 아이디어**: 네트워크 내부에 **순환적인 연결(Recurrent Connection)**을 포함하여, 이전 타임스텝(time step)의 정보를 현재 타임스텝의 계산에 활용합니다. 이를 통해 "메모리"와 유사한 역할을 수행하여 시퀀스 내의 정보를 기억하고 전달할 수 있습니다.

### RNN의 기본 구조
- 각 타임스텝 `t`에서 RNN 셀은 두 가지 입력을 받습니다:
    1.  **현재 타임스텝의 입력 (x<sub>t</sub>)**
    2.  **이전 타임스텝의 은닉 상태 (hidden state, h<sub>t-1</sub>)**
- 이 두 입력을 사용하여 현재 타임스텝의 **은닉 상태 (h<sub>t</sub>)**를 계산하고, 필요에 따라 **출력 (y<sub>t</sub>)**을 생성합니다.
- 은닉 상태 `h_t`는 다음 타임스텝으로 전달되어 과거의 정보를 요약하고 전달하는 역할을 합니다.

![RNN Unrolled Structure](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png)
*(이미지 출처: Christopher Olah's blog)*

- **수식 표현**:
    - **은닉 상태 계산**: h<sub>t</sub> = tanh(W<sub>hh</sub> * h<sub>t-1</sub> + W<sub>xh</sub> * x<sub>t</sub> + b<sub>h</sub>)
        - W<sub>hh</sub>: 이전 은닉 상태에서 현재 은닉 상태로의 가중치 행렬
        - W<sub>xh</sub>: 현재 입력에서 현재 은닉 상태로의 가중치 행렬
        - b<sub>h</sub>: 은닉 상태의 편향 벡터
        - tanh: 활성화 함수 (주로 하이퍼볼릭 탄젠트 사용)
    - **출력 계산 (선택 사항)**: y<sub>t</sub> = W<sub>hy</sub> * h<sub>t</sub> + b<sub>y</sub>
        - W<sub>hy</sub>: 현재 은닉 상태에서 출력으로의 가중치 행렬
        - b<sub>y</sub>: 출력의 편향 벡터
- **가중치 공유 (Weight Sharing)**: RNN은 모든 타임스텝에서 동일한 가중치 행렬(W<sub>hh</sub>, W<sub>xh</sub>, W<sub>hy</sub>)과 편향(b<sub>h</sub>, b<sub>y</sub>)을 사용합니다. 이는 모델의 파라미터 수를 줄이고, 다양한 길이의 시퀀스에 대해 일반화할 수 있게 합니다.

### RNN의 "메모리"
- 은닉 상태 `h_t`는 과거 타임스텝들의 정보를 요약하여 저장하고 있는 것으로 해석될 수 있습니다.
- 이 "메모리"를 통해 RNN은 시퀀스 내의 의존성을 학습할 수 있습니다.

## 3. RNN의 다양한 구조 (유형)
입력 시퀀스와 출력 시퀀스의 길이에 따라 다양한 형태의 RNN 구조가 가능합니다.

![RNN Architectures](https://i.stack.imgur.com/L4K0B.png)
*(이미지 출처: Stack Overflow, Andrej Karpathy)*

-   **One-to-One (바닐라 신경망)**: 하나의 입력, 하나의 출력. (엄밀히는 RNN이 아님)
    *   예: 이미지 분류 (고정 크기 입력)
-   **One-to-Many**: 하나의 입력을 받아 여러 개의 출력을 순차적으로 생성.
    *   예: 이미지 캡셔닝 (이미지 입력 -> 단어 시퀀스 출력)
-   **Many-to-One**: 여러 개의 입력을 순차적으로 받아 하나의 출력을 생성.
    *   예: 감성 분석 (단어 시퀀스 입력 -> 긍정/부정 레이블 출력), 텍스트 분류.
-   **Many-to-Many (동일 길이)**: 입력 시퀀스와 동일한 길이의 출력 시퀀스를 생성. 각 타임스텝마다 출력이 나옴.
    *   예: 품사 태깅 (단어 시퀀스 입력 -> 품사 태그 시퀀스 출력), 개체명 인식.
-   **Many-to-Many (다른 길이, Sequence-to-Sequence)**: 입력 시퀀스와 다른 길이의 출력 시퀀스를 생성. 인코더-디코더 구조에서 주로 사용.
    *   예: 기계 번역 (소스 언어 문장 입력 -> 타겟 언어 문장 출력), 챗봇.

## 4. RNN의 주요 응용 분야
- **자연어 처리 (NLP)**:
    - 언어 모델링 (Language Modeling)
    - 기계 번역 (Machine Translation)
    - 텍스트 생성 (Text Generation)
    - 감성 분석 (Sentiment Analysis)
    - 질의응답 (Question Answering)
    - 품사 태깅 (Part-of-Speech Tagging)
- **음성 인식 (Speech Recognition)**
- **시계열 예측 (Time Series Prediction)**
- **이미지 캡셔닝 (Image Captioning)**
- **비디오 분석 (Video Analysis)**

## 5. RNN의 한계점: 장기 의존성 문제 (Long-Term Dependency Problem)
- RNN은 이론적으로는 과거의 정보를 모두 기억할 수 있지만, 실제로는 시퀀스가 길어질수록 **앞쪽의 정보가 뒤쪽으로 제대로 전달되지 못하는 문제**가 발생합니다.
- 이는 역전파 과정에서 기울기가 너무 작아지거나(Vanishing Gradient) 너무 커지는(Exploding Gradient) 현상 때문에 발생합니다.
    - **기울기 소실 (Vanishing Gradient)**: 시퀀스가 길어질수록 앞쪽 타임스텝으로 전달되는 기울기가 점차 작아져서, 앞쪽 가중치들이 거의 업데이트되지 않아 장기적인 의존성을 학습하기 어려워집니다.
    - **기울기 폭주 (Exploding Gradient)**: 기울기가 매우 커져서 학습이 불안정해지고 발산할 수 있습니다. (기울기 클리핑으로 어느 정도 완화 가능)
- 이 문제로 인해 기본적인 RNN(Vanilla RNN)은 비교적 짧은 시퀀스에서만 효과적입니다.

## 추가 학습 자료
- [Understanding LSTMs (Christopher Olah's Blog)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) - (LSTM 설명이지만 RNN 기본도 잘 다룸)
- [The Unreasonable Effectiveness of Recurrent Neural Networks (Andrej Karpathy's Blog)](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [딥 러닝을 이용한 자연어 처리 입문 - 순환 신경망 (RNN)](https://wikidocs.net/22886)

## 다음 학습 내용
- Day 68: LSTM (Long Short-Term Memory) 네트워크 (Long Short-Term Memory (LSTM) Networks) - RNN의 장기 의존성 문제를 해결하기 위한 주요 대안.
