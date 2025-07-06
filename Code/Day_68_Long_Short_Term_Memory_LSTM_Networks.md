# Day 68: LSTM (Long Short-Term Memory) 네트워크 (Long Short-Term Memory (LSTM) Networks)

## 학습 목표
- RNN의 장기 의존성 문제(Vanishing/Exploding Gradient) 복습
- LSTM의 등장 배경과 핵심 아이디어 이해
- LSTM 셀(Cell)의 내부 구조와 주요 구성 요소(셀 상태, 게이트) 학습:
    - 망각 게이트 (Forget Gate)
    - 입력 게이트 (Input Gate)
    - 출력 게이트 (Output Gate)
- LSTM이 장기 의존성 문제를 어떻게 완화하는지 이해

## 1. RNN의 장기 의존성 문제 복습
- 바닐라 RNN은 시퀀스가 길어질수록 앞쪽의 정보가 뒤쪽으로 잘 전달되지 못하는 **장기 의존성 문제**를 가집니다.
- 이는 역전파 시 기울기가 점차 사라지거나(Vanishing Gradient) 폭주하여(Exploding Gradient) 발생하며, 이로 인해 먼 과거의 정보를 효과적으로 학습하기 어렵습니다.

## 2. LSTM (Long Short-Term Memory) 소개
- Hochreiter & Schmidhuber (1997)에 의해 제안된 RNN의 특별한 종류로, 장기 의존성 문제를 해결하기 위해 설계되었습니다.
- 핵심 아이디어: **셀 상태(Cell State)**라는 별도의 정보 흐름 경로를 두고, **게이트(Gate)**라는 메커니즘을 통해 이 셀 상태에 정보를 추가하거나 제거하는 것을 제어합니다.
- 이를 통해 중요한 정보는 오래 유지하고, 불필요한 정보는 잊어버릴 수 있도록 하여 장기적인 의존성을 효과적으로 학습할 수 있습니다.

## 3. LSTM 셀의 구조
- LSTM 셀은 바닐라 RNN 셀보다 복잡한 구조를 가집니다.
- 주요 구성 요소:
    - **셀 상태 (Cell State, C<sub>t</sub>)**: LSTM의 핵심. 정보가 큰 변화 없이 셀 전체를 관통하여 흐르는 컨베이어 벨트와 유사. 게이트를 통해 정보가 추가되거나 제거될 수 있음. 장기 기억을 담당.
    - **은닉 상태 (Hidden State, h<sub>t</sub>)**: 이전 타임스텝의 출력 및 현재 타임스텝의 단기 기억. 다음 셀로 전달됨.
    - **게이트 (Gates)**: 시그모이드(Sigmoid) 함수와 요소별 곱셈(Pointwise Multiplication) 연산으로 구성. 정보의 흐름을 제어.
        - 시그모이드 함수는 0과 1 사이의 값을 출력하여, 각 정보 조각을 얼마나 통과시킬지 결정 (0: 통과시키지 않음, 1: 모두 통과시킴).

![LSTM Cell Structure](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)
*(이미지 출처: Christopher Olah's blog)*

### 가. 망각 게이트 (Forget Gate, f<sub>t</sub>)
- **역할**: 이전 셀 상태(C<sub>t-1</sub>)에서 어떤 정보를 버릴지(잊어버릴지) 결정합니다.
- **입력**: 이전 은닉 상태(h<sub>t-1</sub>)와 현재 입력(x<sub>t</sub>).
- **계산**: f<sub>t</sub> = σ(W<sub>f</sub> * [h<sub>t-1</sub>, x<sub>t</sub>] + b<sub>f</sub>)
    - σ: 시그모이드 함수
    - W<sub>f</sub>: 망각 게이트의 가중치 행렬
    - b<sub>f</sub>: 망각 게이트의 편향 벡터
    - [h<sub>t-1</sub>, x<sub>t</sub>]: h<sub>t-1</sub>과 x<sub>t</sub>를 연결(concatenate)한 벡터.
- **출력**: 0과 1 사이의 값을 가지는 벡터. 각 요소는 이전 셀 상태의 해당 정보를 얼마나 유지할지(1에 가까울수록 유지, 0에 가까울수록 삭제)를 나타냅니다.

### 나. 입력 게이트 (Input Gate, i<sub>t</sub>) 및 새로운 후보값 (Candidate Values, C̃<sub>t</sub>)
- **역할**: 현재 입력에서 어떤 새로운 정보를 셀 상태에 저장할지 결정합니다.
- 두 부분으로 나뉩니다:
    1.  **입력 게이트 (i<sub>t</sub>)**: 어떤 값을 업데이트할지 결정.
        - **계산**: i<sub>t</sub> = σ(W<sub>i</sub> * [h<sub>t-1</sub>, x<sub>t</sub>] + b<sub>i</sub>)
    2.  **새로운 후보값 (C̃<sub>t</sub>)**: 셀 상태에 추가될 수 있는 새로운 후보 값들의 벡터를 생성. (tanh 함수 사용)
        - **계산**: C̃<sub>t</sub> = tanh(W<sub>C</sub> * [h<sub>t-1</sub>, x<sub>t</sub>] + b<sub>C</sub>)

### 다. 셀 상태 업데이트 (Updating the Cell State)
- **역할**: 이전 셀 상태를 업데이트하여 새로운 셀 상태(C<sub>t</sub>)를 만듭니다.
- **계산**: C<sub>t</sub> = f<sub>t</sub> * C<sub>t-1</sub> + i<sub>t</sub> * C̃<sub>t</sub>
    - **f<sub>t</sub> * C<sub>t-1</sub>**: 이전 셀 상태에서 망각 게이트를 통해 선택적으로 정보를 버립니다.
    - **i<sub>t</sub> * C̃<sub>t</sub>**: 입력 게이트를 통해 선택된 새로운 후보값을 스케일링합니다.
    - 이 두 부분을 더하여 새로운 셀 상태를 만듭니다.

### 라. 출력 게이트 (Output Gate, o<sub>t</sub>) 및 은닉 상태 업데이트
- **역할**: 업데이트된 셀 상태(C<sub>t</sub>)를 기반으로 어떤 정보를 현재 타임스텝의 은닉 상태(h<sub>t</sub>, 즉 출력)로 내보낼지 결정합니다.
- **계산**:
    1.  **출력 게이트 (o<sub>t</sub>)**: 셀 상태의 어느 부분을 출력할지 결정.
        - o<sub>t</sub> = σ(W<sub>o</sub> * [h<sub>t-1</sub>, x<sub>t</sub>] + b<sub>o</sub>)
    2.  **은닉 상태 (h<sub>t</sub>)**:
        - h<sub>t</sub> = o<sub>t</sub> * tanh(C<sub>t</sub>)
        - 셀 상태 C<sub>t</sub>를 tanh 함수에 통과시켜 -1과 1 사이의 값으로 만들고, 출력 게이트 o<sub>t</sub>와 곱하여 최종 은닉 상태(출력)를 결정합니다.

## 4. LSTM이 장기 의존성 문제를 완화하는 방식
- **셀 상태의 분리**: 셀 상태는 상대적으로 간단한 선형 연산(곱셈과 덧셈)을 통해 정보가 흐르므로, 기울기가 여러 층을 거치면서 소실되거나 폭주하는 문제가 줄어듭니다. 정보가 비교적 잘 보존되어 전달될 수 있습니다.
- **게이트 메커니즘**:
    - **망각 게이트**: 불필요한 정보를 명시적으로 제거하여 셀 상태가 과도하게 복잡해지는 것을 방지합니다.
    - **입력 게이트**: 중요한 새로운 정보만 선택적으로 셀 상태에 추가합니다.
    - 이를 통해 LSTM은 시퀀스에서 중요한 정보는 장기간 기억하고, 덜 중요한 정보는 잊어버리는 학습을 할 수 있습니다.
- 결과적으로, LSTM은 바닐라 RNN보다 훨씬 긴 시퀀스에 대해서도 효과적으로 학습할 수 있으며, 장기 의존성 문제를 크게 완화합니다.

## 5. GRU (Gated Recurrent Unit)
- LSTM과 유사한 성능을 내면서 구조가 더 간단한 모델로, Cho 등이 2014년에 제안했습니다.
- 망각 게이트와 입력 게이트를 **업데이트 게이트(Update Gate)**로 통합하고, 셀 상태와 은닉 상태를 하나로 합쳤습니다.
- **리셋 게이트(Reset Gate)**를 사용하여 과거 정보를 얼마나 무시할지 결정합니다.
- LSTM보다 파라미터 수가 적어 계산 효율성이 높고, 데이터가 적을 때 과적합 방지에 유리할 수 있습니다.
- 성능은 문제에 따라 LSTM과 비슷하거나 약간 낮을 수 있습니다.

## 추가 학습 자료
- [Understanding LSTMs (Christopher Olah's Blog)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) - (LSTM 이해를 위한 최고의 자료 중 하나)
- [Illustrated Guide to LSTMs and GRUs: A step by step explanation (Michael Phi)](https://towardsdatascience.com/illustrated-guide-to-lstms-and-grus-a-step-by-step-explanation-44e9eb85bf21)
- [딥 러닝을 이용한 자연어 처리 입문 - LSTM](https://wikidocs.net/22888)

## 다음 학습 내용
- Day 69: RNN/LSTM을 이용한 텍스트 분류기 구축 (Building a text classifier using RNN/LSTM)
