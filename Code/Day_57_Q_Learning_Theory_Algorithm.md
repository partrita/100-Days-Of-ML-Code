# Day 57: Q-러닝 - 이론 및 알고리즘 (Q-Learning - Theory and Algorithm)

## 학습 목표
- Q-러닝의 개념과 중요성 이해
- Q-러닝이 모델 프리(Model-Free) 및 오프 폴리시(Off-Policy) 알고리즘인 이유 이해
- Q-러닝 업데이트 규칙 학습

## 핵심 개념

### 1. Q-러닝 (Q-Learning)
- 대표적인 모델 프리(Model-Free), 오프 폴리시(Off-Policy) 강화 학습 알고리즘입니다.
- 환경의 모델(상태 전이 확률, 보상 함수)을 알지 못해도 최적의 행동 가치 함수(Optimal Action-Value Function, Q*)를 학습할 수 있습니다.
- Q*를 학습함으로써 에이전트는 각 상태에서 어떤 행동을 취해야 누적 보상을 최대로 얻을 수 있는지 알게 됩니다.

### 2. 모델 프리 (Model-Free)
- 환경의 모델 P(s'|s, a) 및 R(s, a, s')을 직접적으로 사용하거나 추정하지 않습니다.
- 대신, 에이전트가 환경과 직접 상호작용하며 얻는 경험(샘플)로부터 가치 함수를 학습합니다. (예: (s, a, r, s') 튜플)

### 3. 오프 폴리시 (Off-Policy)
- 에이전트가 행동을 선택하는 데 사용하는 정책(행동 정책, Behavior Policy)과 평가하고 개선하려는 정책(타겟 정책, Target Policy)이 다를 수 있습니다.
- Q-러닝에서는 타겟 정책은 항상 탐욕적 정책(Greedy Policy)으로, 현재 학습된 Q 값에 대해 가장 높은 Q 값을 주는 행동을 선택합니다.
- 반면, 행동 정책은 탐험(Exploration)을 위해 ε-탐욕적 정책(ε-Greedy Policy) 등을 사용할 수 있습니다. 이를 통해 다양한 상태-행동 쌍을 경험하고 더 나은 Q 값을 학습할 기회를 얻습니다.

### 4. Q-러닝 업데이트 규칙 (Q-Learning Update Rule)
- Q-러닝은 시간차 학습(Temporal Difference Learning, TD Learning)의 한 형태입니다.
- 현재의 Q 값 추정치와 실제 관찰된 보상 및 다음 상태의 Q 값(TD 타겟) 간의 차이(TD 오차)를 이용하여 Q 값을 업데이트합니다.
- Q(S<sub>t</sub>, A<sub>t</sub>) ← Q(S<sub>t</sub>, A<sub>t</sub>) + α [R<sub>t+1</sub> + γ max<sub>a</sub>Q(S<sub>t+1</sub>, a) - Q(S<sub>t</sub>, A<sub>t</sub>)]
    - **S<sub>t</sub>**: 현재 상태
    - **A<sub>t</sub>**: 현재 상태에서 취한 행동
    - **R<sub>t+1</sub>**: 행동 A<sub>t</sub>에 대한 보상
    - **S<sub>t+1</sub>**: 행동 A<sub>t</sub> 이후의 다음 상태
    - **α (Learning Rate, 학습률)**: 0과 1 사이의 값으로, 새로운 정보를 얼마나 반영할지 결정합니다.
    - **γ (Discount Factor, 감가율)**: 미래 보상의 현재 가치를 나타냅니다.
    - **max<sub>a</sub>Q(S<sub>t+1</sub>, a)**: 다음 상태 S<sub>t+1</sub>에서 가능한 모든 행동 a에 대한 Q 값 중 최댓값. 이것이 오프 폴리시 특성을 나타내는 부분으로, 다음 행동을 실제로 어떤 정책으로 선택했든 상관없이 최적 정책(탐욕 정책)을 가정하고 업데이트합니다.
    - **[R<sub>t+1</sub> + γ max<sub>a</sub>Q(S<sub>t+1</sub>, a)]**: TD 타겟. 현재 추정하는 Q(S<sub>t</sub>, A<sub>t</sub>)의 목표값입니다.
    - **[R<sub>t+1</sub> + γ max<sub>a</sub>Q(S<sub>t+1</sub>, a) - Q(S<sub>t</sub>, A<sub>t</sub>)]**: TD 오차(TD Error).

### 5. Q-러닝 알고리즘 (Q-Learning Algorithm)
1. 모든 상태-행동 쌍 (s, a)에 대해 Q(s, a)를 임의의 값으로 초기화 (보통 0 또는 작은 무작위 값). 터미널 상태의 Q 값은 0으로 초기화.
2. 각 에피소드에 대해 다음을 반복:
    a. 초기 상태 S를 관찰.
    b. 에피소드가 끝날 때까지 다음을 반복 (S가 터미널 상태가 아닐 동안):
        i. 현재 상태 S에서 행동 정책(예: ε-탐욕 정책)에 따라 행동 A를 선택.
        ii. 행동 A를 수행하고, 보상 R과 다음 상태 S'를 관찰.
        iii. Q(S, A) ← Q(S, A) + α [R + γ max<sub>a'</sub>Q(S', a') - Q(S, A)] 를 사용하여 Q 값을 업데이트.
        iv. S ← S' (상태를 다음 상태로 업데이트).

### 6. 수렴 조건
- 학습률 α가 적절히 감소하고, 모든 상태-행동 쌍을 무한히 많이 방문하면 Q(s, a)는 최적 행동 가치 함수 Q*(s, a)로 수렴하는 것이 보장됩니다.

## 추가 학습 자료
- [Q-Learning - GeeksforGeeks](https://www.geeksforgeeks.org/q-learning-in-reinforcement-learning/)
- [Simple Q-Learning by Moustafa Alzantot (YouTube)](https://www.youtube.com/watch?v=aRxrMUyXv4s)

## 다음 학습 내용
- Day 58: Q-러닝 - 간단한 구현 (Q-Learning - Simple Implementation)
