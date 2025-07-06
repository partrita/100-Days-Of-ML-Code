# Day 71: 주성분 분석 (PCA) - 이론 및 구현 (Principal Component Analysis (PCA) - Theory and implementation)

## 학습 목표
- 차원 축소(Dimensionality Reduction)의 필요성과 목적 이해
- 주성분 분석(PCA)의 기본 개념과 원리 학습
    - 분산(Variance)을 최대로 보존하는 축 찾기
    - 주성분(Principal Components)의 의미
    - 공분산 행렬(Covariance Matrix)과 고유값/고유벡터(Eigenvalue/Eigenvector)의 역할
- PCA 수행 단계 이해
- `scikit-learn`을 사용한 PCA 구현 방법 숙지

## 1. 차원 축소 (Dimensionality Reduction)
- **정의**: 고차원 데이터의 차원(특성의 수)을 줄이면서 데이터의 중요한 정보는 최대한 유지하는 과정.
- **필요성 및 목적**:
    - **차원의 저주(Curse of Dimensionality) 완화**: 차원이 증가할수록 데이터 간의 거리 계산이 어려워지고, 모델 학습에 필요한 데이터 양이 기하급수적으로 증가하며, 과적합 위험이 커지는 문제.
    - **계산 효율성 증대**: 모델 학습 시간 및 메모리 사용량 감소.
    - **과적합(Overfitting) 방지**: 불필요한 노이즈나 중복된 특성을 제거하여 모델의 일반화 성능 향상.
    - **데이터 시각화**: 고차원 데이터를 2차원 또는 3차원으로 축소하여 시각적으로 탐색 가능.
    - **특징 추출(Feature Extraction)**: 기존 특성들의 조합으로 새로운, 더 의미 있는 특성을 생성.

- **주요 차원 축소 기법**:
    - **주성분 분석 (PCA)**: 대표적인 선형 차원 축소 기법.
    - 선형 판별 분석 (LDA, Linear Discriminant Analysis): 지도 학습 기반의 차원 축소 (분류 목적).
    - t-SNE (t-Distributed Stochastic Neighbor Embedding): 비선형 차원 축소, 주로 시각화에 사용.
    - UMAP (Uniform Manifold Approximation and Projection): 비선형 차원 축소, t-SNE보다 빠르고 전역 구조 보존에 유리.

## 2. 주성분 분석 (Principal Component Analysis, PCA)

### 가. 기본 개념
- 데이터의 **분산(Variance)**을 가장 잘 나타내는 새로운 좌표축(주성분)을 찾는 방식으로 차원을 축소하는 통계적 기법입니다.
- 즉, 데이터가 가장 넓게 퍼져 있는 방향을 첫 번째 주성분(PC1)으로 삼고, PC1에 직교하면서 다음으로 분산이 큰 방향을 두 번째 주성분(PC2)으로 삼는 방식으로 진행됩니다.
- 원래 특성 공간에서 주성분 공간으로 데이터를 변환(사영, Projection)합니다.
- 비지도 학습 기법으로, 타겟 변수(레이블)를 사용하지 않습니다.

### 나. 핵심 원리
- **분산 최대화**: 각 주성분은 해당 축으로 데이터를 사영했을 때 분산이 최대가 되는 방향으로 결정됩니다.
- **직교성**: 각 주성분은 서로 직교(Orthogonal)합니다. 이는 주성분들이 서로 독립적임을 의미하며, 정보의 중복을 최소화합니다.
- **고유값과 고유벡터**: 데이터의 공분산 행렬(Covariance Matrix)을 사용하여 주성분을 찾습니다.
    - **공분산 행렬**: 특성들 간의 선형 관계(공분산)를 나타내는 정방 행렬.
    - **고유벡터(Eigenvector)**: 공분산 행렬의 고유벡터는 주성분의 방향을 나타냅니다.
    - **고유값(Eigenvalue)**: 해당 고유벡터 방향으로 데이터가 얼마나 큰 분산을 가지는지(주성분의 중요도)를 나타냅니다. 고유값이 클수록 더 많은 분산을 설명하는 중요한 주성분입니다.

### 다. PCA 수행 단계
1.  **(선택) 데이터 스케일링 (Data Scaling)**:
    - PCA는 분산에 기반하므로, 특성들의 스케일(단위)이 다르면 분산이 큰 특성이 주성분에 과도한 영향을 미칠 수 있습니다.
    - 따라서, 각 특성을 평균 0, 분산 1로 만드는 표준화(Standardization)를 수행하는 것이 일반적입니다. (`StandardScaler` 사용)

2.  **공분산 행렬 계산**:
    - 스케일링된 데이터의 공분산 행렬을 계산합니다. (데이터가 n개의 특성을 가지면 n x n 크기의 행렬)

3.  **고유값 분해 (Eigen Decomposition)**:
    - 공분산 행렬에 대해 고유값 분해를 수행하여 고유값과 해당 고유벡터를 구합니다.

4.  **주성분 결정**:
    - 고유값이 큰 순서대로 고유벡터를 정렬합니다. 이 고유벡터들이 주성분이 됩니다.
    - 첫 번째 고유벡터가 PC1, 두 번째 고유벡터가 PC2, ...

5.  **차원 축소**:
    - 원래 데이터에서 유지하고자 하는 주성분의 개수 `k`를 선택합니다 (k < 원래 차원 수).
    - 선택된 `k`개의 주성분(고유벡터)으로 이루어진 행렬을 만듭니다.
    - 원래 데이터를 이 `k`개의 주성분으로 이루어진 새로운 공간으로 사영(Projection)시켜 `k`차원의 데이터로 변환합니다.
    - 변환된 데이터 = 원본 데이터 (스케일링된) × 선택된 고유벡터 행렬

6.  **(선택) 설명된 분산 (Explained Variance) 확인**:
    - 각 주성분이 전체 분산 중 얼마나 많은 부분을 설명하는지 확인합니다. (`explained_variance_ratio_`)
    - 누적 설명된 분산 비율을 보고 몇 개의 주성분을 선택할지 결정하는 데 도움을 받을 수 있습니다. (예: 95% 이상의 분산을 설명하는 최소한의 주성분 선택)

## 3. `scikit-learn`을 사용한 PCA 구현

### 가. 예제 데이터 생성
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 예제 데이터 생성 (2차원)
np.random.seed(42)
X = np.dot(np.random.rand(2, 2), np.random.randn(2, 200)).T # 200개의 샘플, 2개의 특성
# 데이터가 원점에 모여있지 않도록 약간 이동
X[:, 0] += 5
X[:, 1] -= 2

plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], alpha=0.7)
plt.title("Original Data (2D)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.axis('equal') # 축 스케일을 동일하게
plt.grid(True)
plt.show()
```

### 나. 데이터 스케일링
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

plt.figure(figsize=(6, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], alpha=0.7)
plt.title("Scaled Data (Mean 0, Var 1)")
plt.xlabel("Feature 1 (scaled)")
plt.ylabel("Feature 2 (scaled)")
plt.axis('equal')
plt.grid(True)
plt.show()
```

### 다. PCA 적용
- `n_components`: 축소할 차원(주성분)의 수.
    - 정수: 선택할 주성분의 개수.
    - 0과 1 사이의 실수 (예: 0.95): 설명된 분산의 비율이 해당 값 이상이 되도록 주성분 개수를 자동으로 선택.
    - `None`: 모든 주성분을 선택 (차원 축소 안 함, 분산 분석 목적).

```python
# PCA 객체 생성 및 학습
# 예: 2차원에서 1차원으로 축소
pca_1d = PCA(n_components=1)
X_pca_1d = pca_1d.fit_transform(X_scaled)

print("Original data shape:", X_scaled.shape)
print("PCA transformed data shape (1D):", X_pca_1d.shape)

# 주성분 확인
print("\nPrincipal components (Eigenvectors):\n", pca_1d.components_) # 각 행이 주성분 벡터
# 설명된 분산 비율 확인
print("Explained variance ratio (1D):", pca_1d.explained_variance_ratio_) # 각 주성분이 설명하는 분산의 비율
print("Explained variance (1D):", pca_1d.explained_variance_) # 각 주성분의 고유값 (분산 크기)

# 1차원으로 축소된 데이터를 다시 2차원으로 복원 (정보 손실 발생)
X_restored_1d_to_2d = pca_1d.inverse_transform(X_pca_1d)

# 시각화
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], alpha=0.3, label='Original Scaled Data')
plt.scatter(X_restored_1d_to_2d[:, 0], X_restored_1d_to_2d[:, 1], alpha=0.7, label='Restored from 1D PCA', marker='x', color='red')
# 주성분 벡터 시각화 (첫 번째 주성분)
# components_는 (n_components, n_features) 형태
# 각 주성분 벡터는 원점에서 시작한다고 가정
origin = [0, 0] # 원점
pc1_vector = pca_1d.components_[0] * np.sqrt(pca_1d.explained_variance_[0]) * 2 # 스케일링하여 잘 보이도록
plt.quiver(*origin, *pc1_vector, color=['green'], scale=5, label=f'PC1 (Expl. Var: {pca_1d.explained_variance_ratio_[0]:.2f})')

plt.title("PCA: Original vs Restored from 1D")
plt.xlabel("Feature 1 (scaled)")
plt.ylabel("Feature 2 (scaled)")
plt.axis('equal')
plt.legend()
plt.grid(True)
plt.show()


# 예: 모든 주성분 확인 (n_components=None 또는 생략)
pca_all = PCA() # 또는 PCA(n_components=None) 또는 PCA(n_components=2) for 2D data
X_pca_all = pca_all.fit_transform(X_scaled) # 이 경우 X_pca_all은 X_scaled를 회전시킨 형태

print("\nPrincipal components (All):\n", pca_all.components_)
print("Explained variance ratio (All):", pca_all.explained_variance_ratio_)
print("Cumulative explained variance ratio (All):", np.cumsum(pca_all.explained_variance_ratio_))

# 주성분 시각화 (PC1, PC2)
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], alpha=0.3, label='Original Scaled Data')
for i, (comp, var) in enumerate(zip(pca_all.components_, pca_all.explained_variance_ratio_)):
    # 주성분 벡터를 분산 크기만큼 스케일링하여 화살표로 표시
    v = comp * np.sqrt(pca_all.explained_variance_[i]) * 2 # 스케일링하여 잘 보이도록
    plt.quiver(0, 0, v[0], v[1], color=f'C{i+2}', scale=5, label=f'PC{i+1} (Expl. Var: {var:.2f})')

plt.title("Principal Components of Scaled Data")
plt.xlabel("Feature 1 (scaled)")
plt.ylabel("Feature 2 (scaled)")
plt.axis('equal')
plt.legend()
plt.grid(True)
plt.show()

# Scree Plot (설명된 분산 시각화)
plt.figure(figsize=(6, 4))
plt.bar(range(1, len(pca_all.explained_variance_ratio_) + 1), pca_all.explained_variance_ratio_, alpha=0.7, align='center',
        label='Individual explained variance')
plt.step(range(1, len(pca_all.explained_variance_ratio_) + 1), np.cumsum(pca_all.explained_variance_ratio_), where='mid',
         label='Cumulative explained variance', color='red')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.xticks(range(1, len(pca_all.explained_variance_ratio_) + 1))
plt.title('Scree Plot')
plt.legend(loc='best')
plt.grid(True)
plt.show()
```

### 라. 실제 데이터셋에 적용 (예: Iris 데이터셋)
```python
from sklearn.datasets import load_iris

iris = load_iris()
X_iris = iris.data # (150, 4) - 4개의 특성
y_iris = iris.target

# 1. 데이터 스케일링
scaler_iris = StandardScaler()
X_iris_scaled = scaler_iris.fit_transform(X_iris)

# 2. PCA 적용 (예: 4차원 -> 2차원으로 축소)
pca_iris = PCA(n_components=2)
X_iris_pca = pca_iris.fit_transform(X_iris_scaled)

print("\nIris data original shape:", X_iris_scaled.shape)
print("Iris data PCA transformed shape:", X_iris_pca.shape)
print("Explained variance ratio by 2 components:", pca_iris.explained_variance_ratio_)
print("Cumulative explained variance ratio:", np.sum(pca_iris.explained_variance_ratio_))

# 3. 시각화
plt.figure(figsize=(8, 6))
for i, target_name in enumerate(iris.target_names):
    plt.scatter(X_iris_pca[y_iris == i, 0], X_iris_pca[y_iris == i, 1],
                label=target_name, alpha=0.8)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Iris Dataset (2 Components)')
plt.legend()
plt.grid(True)
plt.show()
```

## 4. PCA의 장단점
- **장점**:
    - 구현이 간단하고 계산 효율성이 높습니다.
    - 데이터의 노이즈를 줄이고 과적합을 방지하는 데 도움이 됩니다.
    - 데이터 시각화에 유용합니다.
- **단점**:
    - 선형 변환이므로 비선형 구조를 가진 데이터의 차원 축소에는 한계가 있습니다.
    - 스케일링에 민감하므로 사전에 데이터 스케일링이 필요합니다.
    - 주성분의 해석이 어려울 수 있습니다 (원래 특성들의 선형 결합이므로).
    - 분류 문제에서 클래스 간의 분리 정보를 활용하지 못합니다 (LDA와 비교).

## 추가 학습 자료
- [A Step-by-Step Explanation of Principal Component Analysis (PCA) - Built In](https://builtin.com/data-science/principal-component-analysis)
- [StatQuest: PCA main ideas in only 5 minutes!!! (YouTube)](https://www.youtube.com/watch?v=FgakZw6K1QQ)
- [Scikit-learn PCA Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

## 다음 학습 내용
- Day 72: 선형 판별 분석 (LDA) (Linear Discriminant Analysis (LDA)) - PCA와 비교되는 또 다른 차원 축소 기법.
