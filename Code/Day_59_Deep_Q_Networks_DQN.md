# Day 59: 심층 Q-네트워크 (DQN) - 개념 및 장점 (Deep Q-Networks (DQN) - Concept and Advantages)

## 학습 목표
- 전통적인 Q-러닝의 한계점 이해
- 심층 Q-네트워크(DQN)의 기본 아이디어와 구성 요소 학습
- DQN이 Q-테이블 대신 신경망을 사용하는 이유와 장점 이해
- DQN의 주요 기법(경험 재현, 타겟 네트워크) 소개

## 전통적인 Q-러닝의 한계점

1.  **상태 공간의 저주 (Curse of Dimensionality)**:
    *   Q-테이블은 상태와 행동의 모든 가능한 조합에 대한 Q 값을 저장해야 합니다.
    *   상태 공간이나 행동 공간이 매우 커지면 (예: 연속적인 상태 공간, 고차원 이미지 입력), Q-테이블을 유지하고 업데이트하는 데 필요한 메모리와 계산량이 기하급수적으로 증가하여 현실적으로 불가능해집니다.
    *   예: 아타리 게임의 경우, 화면 픽셀 하나하나를 상태로 간주하면 상태 공간이 엄청나게 커집니다.

2.  **일반화 능력 부족**:
    *   Q-테이블은 각 상태-행동 쌍에 대한 Q 값을 개별적으로 학습합니다.
    *   따라서, 비슷한 상태들에 대한 정보를 일반화하여 활용하기 어렵습니다. 방문해보지 않은 상태에 대해서는 Q 값을 추정할 수 없습니다.

## 심층 Q-네트워크 (Deep Q-Network, DQN)
- 2013년 DeepMind에서 제안하고, 2015년 Nature에 발표된 알고리즘으로, 심층 신경망(Deep Neural Network)을 사용하여 Q-함수를 근사(approximate)합니다.
- Q-테이블 대신 신경망 `Q(s, a; θ)`를 사용하여, 상태 `s`를 입력으로 받아 각 행동 `a`에 대한 Q 값을 출력하거나, 상태 `s`와 행동 `a`를 입력으로 받아 해당 Q 값을 출력합니다. (일반적으로 전자를 많이 사용)
    - `θ`는 신경망의 가중치(weights)를 나타냅니다.
- 이를 통해 Q-러닝을 고차원의 상태 공간(예: 이미지 입력)을 가진 문제에 적용할 수 있게 되었습니다.

### DQN의 기본 아이디어
- 신경망을 사용하여 Q-함수를 근사: `Q(s, a; θ) ≈ Q*(s, a)`
- 손실 함수(Loss Function)를 정의하고, 경사 하강법(Gradient Descent)을 사용하여 신경망의 가중치 `θ`를 업데이트합니다.
- 손실 함수는 Q-러닝의 업데이트 규칙에서 TD 오차와 유사하게 정의됩니다:
  L(θ) = E<sub>(s,a,r,s')~D</sub> [ ( r + γ max<sub>a'</sub>Q(s', a'; θ<sup>-</sup>) - Q(s, a; θ) )<sup>2</sup> ]
    - `θ`: 현재 Q-네트워크의 가중치
    - `θ<sup>-</sup>`: 타겟 Q-네트워크의 가중치 (아래 설명 참조)
    - `D`: 경험 재현 메모리 (아래 설명 참조)
    - `r + γ max<sub>a'</sub>Q(s', a'; θ<sup>-</sup>)`: TD 타겟 (목표값)
    - `Q(s, a; θ)`: 현재 Q-네트워크의 예측값

### DQN의 장점
1.  **고차원 입력 처리**: 이미지를 직접 입력으로 받아 처리하는 등 복잡하고 큰 상태 공간을 다룰 수 있습니다. (예: Convolutional Neural Networks, CNN 사용)
2.  **일반화 성능**: 신경망은 학습된 데이터로부터 패턴을 학습하여, 이전에 방문하지 않은 유사한 상태에 대해서도 Q 값을 추정할 수 있는 일반화 능력을 가집니다.
3.  **메모리 효율성**: 거대한 Q-테이블을 저장하는 대신, 상대적으로 작은 크기의 신경망 가중치만 저장하면 됩니다.

## DQN의 주요 기법

### 1. 경험 재현 (Experience Replay)
- 에이전트가 환경과 상호작용하며 얻는 경험 샘플 `(s_t, a_t, r_{t+1}, s_{t+1})`을 리플레이 메모리(Replay Memory) `D`에 저장합니다.
- 신경망 학습 시, 리플레이 메모리에서 미니배치(mini-batch)를 무작위로 샘플링하여 사용합니다.
- **장점**:
    - **데이터 효율성 향상**: 하나의 경험 샘플이 여러 번의 학습에 사용될 수 있습니다.
    - **샘플 간의 상관관계 감소**: 순차적으로 들어오는 데이터는 시간적 상관관계가 높아 학습을 불안정하게 만들 수 있습니다. 무작위 샘플링은 이러한 상관관계를 줄여 학습 안정성을 높입니다.
    - **학습 안정성**: 특정 패턴의 데이터가 연속적으로 들어오는 것을 방지하여 학습이 특정 방향으로 치우치는 것을 막아줍니다.

### 2. 타겟 네트워크 분리 (Separate Target Network)
- Q-러닝 업데이트 시 TD 타겟 `y_i = r + γ max<sub>a'</sub>Q(s', a'; θ)`을 계산할 때, 현재 Q-값을 예측하는 네트워크와 동일한 네트워크를 사용하면 학습이 불안정해질 수 있습니다.
    - Q-값을 업데이트하면 TD 타겟도 함께 변하게 되어, 학습 목표가 계속 흔들리는 문제가 발생합니다 (Moving Target Problem).
- **해결책**: 두 개의 신경망을 사용합니다.
    - **메인 네트워크 (Main Network, Q-Network)**: `Q(s, a; θ)`. 주로 학습(가중치 업데이트)이 이루어지는 네트워크.
    - **타겟 네트워크 (Target Network)**: `Q(s', a'; θ<sup>-</sup>)`. TD 타겟을 계산하는 데 사용되는 네트워크.
        - 타겟 네트워크의 가중치 `θ<sup>-</sup>`는 주기적으로 메인 네트워크의 가중치 `θ`로 복사되어 업데이트됩니다 (예: 매 C 스텝마다).
        - 타겟 네트워크는 학습 과정에서 고정되어 있어 TD 타겟값을 안정적으로 유지시켜 학습 안정성을 높입니다.

## DQN 알고리즘 개요
1. 리플레이 메모리 D를 특정 크기로 초기화.
2. 행동-가치 함수 Q를 무작위 가중치 θ로 초기화 (메인 네트워크).
3. 타겟 행동-가치 함수 Q̂를 가중치 θ<sup>-</sup> = θ로 초기화 (타겟 네트워크).
4. 각 에피소드에 대해:
    a. 초기 상태 s<sub>1</sub>을 관찰.
    b. 에피소드가 끝날 때까지 (t=1 부터 T까지):
        i. ε-탐욕 정책에 따라 행동 a<sub>t</sub>를 선택 (Q(s<sub>t</sub>, · ; θ) 사용).
        ii. 행동 a<sub>t</sub>를 수행하고, 보상 r<sub>t+1</sub>과 다음 상태 s<sub>t+1</sub>을 관찰.
        iii. 경험 (s<sub>t</sub>, a<sub>t</sub>, r<sub>t+1</sub>, s<sub>t+1</sub>)을 D에 저장.
        iv. D에서 미니배치 (s<sub>j</sub>, a<sub>j</sub>, r<sub>j+1</sub>, s<sub>j+1</sub>)를 무작위로 샘플링.
        v. 타겟 y<sub>j</sub> 설정:
            - 만약 s<sub>j+1</sub>이 터미널 상태이면, y<sub>j</sub> = r<sub>j+1</sub>.
            - 그렇지 않으면, y<sub>j</sub> = r<sub>j+1</sub> + γ max<sub>a'</sub>Q̂(s<sub>j+1</sub>, a'; θ<sup>-</sup>).
        vi. 손실 (y<sub>j</sub> - Q(s<sub>j</sub>, a<sub>j</sub>; θ))<sup>2</sup> 에 대해 경사 하강법을 수행하여 메인 네트워크 가중치 θ 업데이트.
        vii. 매 C 스텝마다 타겟 네트워크 가중치 업데이트: θ<sup>-</sup> ← θ.

## 추가 학습 자료
- [DeepMind Nature Paper (2015) - Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
- [Playing Atari with Deep Reinforcement Learning (NIPS 2013 Workshop Paper)](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
- [DQN Explained - Let's Code DQN (YouTube by SimpleAI)](https://www.youtube.com/watch?v= নারদ7377&list=PLZbbT5o_s2xobby-M9D9sRehVjjYmS3ob&index=2)

## 다음 학습 내용
- Day 60: 자연어 처리 소개 (Introduction to Natural Language Processing)
