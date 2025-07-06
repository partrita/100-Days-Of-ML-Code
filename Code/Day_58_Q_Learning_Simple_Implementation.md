# Day 58: Q-러닝 - 간단한 구현 (Q-Learning - Simple Implementation)

## 학습 목표
- 간단한 환경(예: 그리드 월드)에서 Q-러닝 알고리즘을 파이썬으로 구현
- Q-테이블의 개념과 활용 이해
- ε-탐욕 정책(Epsilon-Greedy Policy) 구현

## 예제 환경: 간단한 그리드 월드 (Simple Grid World)
- 1xN 또는 MxN 형태의 격자 세계.
- 에이전트는 각 셀(상태)에서 상, 하, 좌, 우 등으로 이동(행동)할 수 있습니다.
- 특정 셀에는 장애물이나 목표 지점이 있을 수 있으며, 이에 따라 보상이 주어집니다.
- 예: 1x5 그리드 월드
  - `[S, F, F, F, G]`
  - S: 시작 지점
  - F: 일반 필드 (이동 시 작은 음수 보상 또는 0 보상)
  - G: 목표 지점 (도달 시 큰 양수 보상, 에피소드 종료)
  - 벽을 벗어나려는 행동은 제자리걸음으로 처리하고 음수 보상을 줄 수 있습니다.

## Q-테이블 (Q-Table)
- Q-러닝에서 Q(s, a) 값을 저장하는 테이블 (보통 2차원 배열 또는 딕셔너리).
- 행은 상태(States), 열은 행동(Actions)을 나타냅니다.
- `Q_table[state][action]`은 해당 상태에서 해당 행동을 취했을 때의 Q 값을 의미합니다.
- 환경이 단순하고 상태와 행동 공간이 작을 때 유용합니다.

## ε-탐욕 정책 (Epsilon-Greedy Policy)
- 탐험(Exploration)과 활용(Exploitation) 사이의 균형을 맞추기 위한 정책.
- **활용**: 현재까지 학습된 Q 값 중 가장 높은 값을 주는 행동을 선택 (Greedy action).
- **탐험**: 무작위로 행동을 선택하여 새로운 경험을 쌓고, 더 나은 정책을 발견할 가능성을 높임.
- **알고리즘**:
    1. 0과 1 사이의 무작위 수 `rand`를 생성합니다.
    2. `rand < ε` 이면 (확률 ε로):
        - 가능한 행동 중 하나를 무작위로 선택 (탐험).
    3. 그렇지 않으면 (`rand ≥ ε` 이면, 확률 1-ε로):
        - 현재 상태에서 Q 값이 가장 높은 행동을 선택 (활용).
- ε 값은 보통 학습 초반에는 높게 설정하여 탐험을 장려하고, 학습이 진행됨에 따라 점차 낮추어 활용에 집중하도록 합니다 (ε-decay).

## 파이썬 구현 개요 (슈도코드 스타일)

```python
import numpy as np

# 환경 설정
# 예: 1x5 그리드 월드
# 상태: 0, 1, 2, 3, 4 (S, F, F, F, G)
# 행동: 0 (왼쪽), 1 (오른쪽)
# 보상: 목표 도달 +1, 그 외 0 (간단화)

num_states = 5
num_actions = 2

# Q-테이블 초기화
q_table = np.zeros((num_states, num_actions))

# 하이퍼파라미터
learning_rate = 0.1  # 학습률 (α)
discount_factor = 0.9 # 감가율 (γ)
epsilon = 1.0         # 초기 epsilon 값
max_epsilon = 1.0     # 최대 epsilon 값
min_epsilon = 0.01    # 최소 epsilon 값
epsilon_decay_rate = 0.001 # epsilon 감쇠율

num_episodes = 1000

# 보상 정의 (간단화)
# rewards = { (state, action, next_state): reward_value }
# 또는 함수로 정의: get_reward(state, action, next_state)
# 여기서는 간단하게 목표 상태 도달 시 +1
goal_state = 4

# Q-러닝 알고리즘
for episode in range(num_episodes):
    state = 0 # 시작 상태 (S)
    done = False

    while not done:
        # Epsilon-greedy 행동 선택
        exploration_exploitation_tradeoff = np.random.uniform(0, 1)

        if exploration_exploitation_tradeoff < epsilon:
            action = np.random.choice(num_actions) # 탐험: 무작위 행동 선택
        else:
            action = np.argmax(q_table[state, :])  # 활용: Q값이 가장 높은 행동 선택

        # 행동 수행 및 다음 상태, 보상 관찰 (환경과의 상호작용)
        # 이 부분은 환경 모델에 따라 달라짐
        if action == 0: # 왼쪽
            next_state = max(0, state - 1)
        else: # 오른쪽
            next_state = min(num_states - 1, state + 1)

        reward = 0
        if next_state == goal_state:
            reward = 1
            done = True

        # Q-테이블 업데이트
        q_table[state, action] = q_table[state, action] + learning_rate * \
            (reward + discount_factor * np.max(q_table[next_state, :]) - q_table[state, action])

        state = next_state

    # Epsilon 감쇠
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay_rate * episode)

print("학습된 Q-테이블:")
print(q_table)

# 학습된 정책 테스트
state = 0
path = [state]
while state != goal_state:
    action = np.argmax(q_table[state, :])
    if action == 0:
        state = max(0, state - 1)
    else:
        state = min(num_states - 1, state + 1)
    path.append(state)
    if len(path) > 10: # 무한 루프 방지
        print("경로를 찾지 못함 또는 너무 김")
        break
print("최적 경로 (추정):", path)

```

## 고려 사항
- **환경 표현**: 실제 구현에서는 환경을 클래스 등으로 더 잘 구조화할 수 있습니다 (예: `step(action)` 함수를 통해 `next_state, reward, done, info` 반환). OpenAI Gym과 유사한 인터페이스를 고려할 수 있습니다.
- **보상 설계**: 보상을 어떻게 설계하느냐에 따라 에이전트의 학습 결과가 크게 달라집니다.
- **하이퍼파라미터 튜닝**: `learning_rate`, `discount_factor`, `epsilon` 관련 값들은 문제와 환경에 따라 적절히 조절해야 합니다.
- **수렴**: 간단한 문제에서는 Q-테이블이 최적 값으로 빠르게 수렴할 수 있지만, 복잡한 문제에서는 많은 에피소드가 필요할 수 있습니다.

## 추가 학습 자료
- [Q-Learning Example with Python (Towards Data Science)](https://towardsdatascience.com/q-learning-algorithm-from-explanation-to-implementation-9105aa038194)
- [OpenAI Gym FrozenLake Q-Learning (YouTube)](https://www.youtube.com/watch?v=Mut_u40Sqz4) (Gym 환경에서의 예시)

## 다음 학습 내용
- Day 59: 심층 Q-네트워크 (DQN) - 개념 및 장점 (Deep Q-Networks (DQN) - Concept and Advantages)
