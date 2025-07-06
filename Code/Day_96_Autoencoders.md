# Day 96: 오토인코더 (Autoencoders)

## 학습 목표
- 오토인코더의 기본 개념과 구조(인코더, 디코더, 잠재 공간) 이해.
- 오토인코더의 학습 목표: 입력 데이터의 효율적인 압축 및 복원.
- 다양한 종류의 오토인코더 소개:
    - 기본 오토인코더 (Vanilla Autoencoder)
    - 희소 오토인코더 (Sparse Autoencoder)
    - 디노이징 오토인코더 (Denoising Autoencoder)
    - 변이형 오토인코더 (Variational Autoencoder, VAE) - 간략 소개
- 오토인코더의 주요 응용 분야 학습.

## 1. 오토인코더 (Autoencoder)란?
- **정의**: 입력 데이터를 효율적으로 압축(인코딩)했다가 다시 원래 입력으로 복원(디코딩)하도록 학습하는 비지도 학습(Unsupervised Learning) 기반의 인공 신경망.
- **목표**: 입력 데이터의 중요한 특징(Feature)만을 추출하여 저차원의 **잠재 공간(Latent Space)** 표현으로 압축하고, 이 잠재 표현으로부터 원본 입력과 최대한 유사하게 복원하는 것.
- "자기 자신을 복제하도록 학습하는 네트워크"라고 생각할 수 있습니다. (입력 = 출력 목표)

## 2. 오토인코더의 기본 구조
오토인코더는 크게 두 부분으로 구성됩니다:

### 가. 인코더 (Encoder)
- **역할**: 입력 데이터(x)를 받아 저차원의 잠재 공간 표현(z, 또는 h)으로 압축(인코딩)합니다.
- **구조**: 입력층에서 시작하여 점차 뉴런 수가 줄어드는 신경망 층들로 구성됩니다.
- **출력**: 잠재 벡터(Latent Vector) z = f(x). 이 벡터는 입력 데이터의 핵심 정보를 압축적으로 담고 있어야 합니다.

### 나. 디코더 (Decoder)
- **역할**: 인코더가 생성한 잠재 벡터(z)를 받아 원래 입력 데이터와 유사한 형태로 복원(디코딩)합니다.
- **구조**: 잠재 공간 표현을 입력으로 받아 점차 뉴런 수가 늘어나 원래 입력 차원과 같아지는 신경망 층들로 구성됩니다.
- **출력**: 복원된 데이터 x' = g(z).

### 잠재 공간 (Latent Space) / 병목층 (Bottleneck Layer)
- 인코더의 출력과 디코더의 입력이 만나는 부분으로, 오토인코더에서 가장 중요한 부분입니다.
- 입력 데이터보다 낮은 차원을 가지도록 설계되어, 데이터의 본질적인 특징만을 학습하도록 강제합니다. (차원 축소 효과)
- 이 잠재 공간의 표현(z)이 얼마나 정보를 잘 담고 있느냐가 오토인코더의 성능을 좌우합니다.

![Autoencoder Architecture](https://upload.wikimedia.org/wikipedia/commons/thumb/2/28/Autoencoder_structure.svg/1200px-Autoencoder_structure.svg.png)
*(이미지 출처: Wikipedia)*

## 3. 오토인코더의 학습 과정
- **손실 함수 (Loss Function)**: 입력 데이터(x)와 디코더가 복원한 데이터(x') 사이의 차이를 최소화하는 방향으로 학습합니다. 이 차이를 **재구성 손실(Reconstruction Loss)**이라고 합니다.
    - **수치형 데이터**: 평균 제곱 오차 (Mean Squared Error, MSE)
        L(x, x') = ||x - x'||<sup>2</sup>
    - **이진 데이터 (베르누이 분포 가정)**: 교차 엔트로피 (Cross-Entropy)
        L(x, x') = - Σ (x<sub>i</sub> log(x'<sub>i</sub>) + (1 - x<sub>i</sub>) log(1 - x'<sub>i</sub>))
- **학습 방식**: 일반적인 신경망과 동일하게 역전파(Backpropagation) 알고리즘을 사용하여 인코더와 디코더의 가중치를 업데이트합니다.
- 비지도 학습이지만, 실제로는 입력을 타겟으로 사용하는 일종의 "자기 지도 학습(Self-supervised Learning)"으로 볼 수 있습니다.

## 4. 다양한 종류의 오토인코더

### 가. 기본 오토인코더 (Vanilla Autoencoder / Undercomplete Autoencoder)
- 가장 기본적인 형태로, 잠재 공간의 차원이 입력 데이터의 차원보다 작은 경우입니다.
- 주로 데이터 압축 및 차원 축소에 사용됩니다.
- 너무 강력한 인코더와 디코더를 사용하면 (예: 레이어가 너무 많거나 뉴런 수가 너무 많으면) 단순히 입력을 그대로 복사하는 항등 함수(Identity Function)를 학습할 위험이 있습니다. 이를 방지하기 위해 잠재 공간의 차원을 제한합니다.

### 나. 희소 오토인코더 (Sparse Autoencoder)
- 잠재 공간의 차원이 입력 데이터의 차원과 같거나 더 클 수도 있지만, **잠재 공간 표현(z)의 대부분의 뉴런이 비활성화(값이 0에 가깝게)**되도록 규제(Regularization)를 추가한 오토인코더입니다.
- **목표**: 데이터의 중요한 특징을 소수의 활성화된 뉴런에 집중시켜 학습하도록 유도합니다.
- **규제 방식**:
    - L1 규제: 잠재 공간 활성화 값의 L1 노름을 손실 함수에 추가.
    - KL 발산 (KL Divergence): 잠재 공간 뉴런의 평균 활성화도가 특정 작은 값(예: 0.05)에 가깝도록 하는 제약 추가.
- 특징 추출(Feature Learning)에 유용합니다.

### 다. 디노이징 오토인코더 (Denoising Autoencoder, DAE)
- **개념**: 입력 데이터에 의도적으로 노이즈(Noise)를 추가한 후, 이를 원래의 깨끗한 입력 데이터로 복원하도록 학습하는 오토인코더입니다.
- **학습 과정**:
    1.  원본 입력 데이터 x에 노이즈를 추가하여 손상된 입력 x̃를 만듭니다.
    2.  인코더는 x̃를 잠재 표현 z로 인코딩합니다: z = f(x̃).
    3.  디코더는 z로부터 원본 데이터 x를 복원하도록 학습합니다: x' = g(z).
    4.  손실 함수는 L(x, x')로 계산 (원본 x와 복원된 x' 간의 차이).
- **효과**: 입력의 작은 변화에 강인한(Robust) 특징을 학습하고, 데이터의 주요 구조를 더 잘 파악하도록 유도합니다. 노이즈 제거 및 특징 학습에 효과적입니다.

![Denoising Autoencoder](https://www.researchgate.net/publication/330324309/figure/fig danneggiato1/AS:713801354010624@1547195110922/Denoising-autoencoder-architecture-The-original-input-x-is-corrupted-into-x-by.png)
*(이미지 출처: ResearchGate)*

### 라. 변이형 오토인코더 (Variational Autoencoder, VAE) - (GAN과 함께 대표적인 생성 모델)
- **개념**: 기본 오토인코더를 확률적으로 확장하여, 잠재 공간(z)이 특정 확률 분포(보통 정규분포)를 따르도록 학습하는 생성 모델입니다.
- **특징**:
    - 인코더는 입력 x에 대해 잠재 변수 z의 **평균(μ)과 분산(σ<sup>2</sup>)**을 출력합니다.
    - 이 평균과 분산을 사용하여 정규분포 N(μ, σ<sup>2</sup>)에서 z를 샘플링합니다.
    - 디코더는 샘플링된 z로부터 새로운 데이터를 생성합니다.
    - **손실 함수**: 재구성 손실 + 잠재 공간의 분포를 정규분포에 가깝게 만드는 규제항 (KL 발산 사용).
- **장점**: 잠재 공간이 연속적이고 잘 구조화되어, 잠재 공간에서 샘플링하여 새로운 데이터를 생성하는 데 용이합니다 (GAN과 유사한 생성 능력).
- **GAN과의 비교**:
    - VAE는 명시적인 확률 분포를 모델링하려 하며, 생성된 이미지의 품질이 GAN보다 다소 흐릿(Blurry)할 수 있습니다.
    - GAN은 학습이 불안정하고 모드 붕괴 문제가 있을 수 있지만, 종종 더 선명하고 사실적인 이미지를 생성합니다.

## 5. 오토인코더의 주요 응용 분야
- **차원 축소 (Dimensionality Reduction)**: PCA와 유사하지만 비선형적인 차원 축소가 가능합니다. (단, PCA만큼 해석이 용이하지는 않음)
- **특징 학습 (Feature Learning / Representation Learning)**: 데이터로부터 유용한 특징을 비지도 방식으로 학습하여 다른 지도 학습 모델의 입력으로 사용.
- **데이터 압축 (Data Compression)**: 인코더를 사용하여 데이터를 압축하고, 디코더로 복원. (손실 압축)
- **노이즈 제거 (Denoising)**: 디노이징 오토인코더를 사용하여 손상된 데이터(이미지, 오디오 등)에서 노이즈를 제거.
- **이상 탐지 (Anomaly Detection / Outlier Detection)**: 정상 데이터로 오토인코더를 학습시킨 후, 새로운 데이터에 대해 재구성 오차가 큰 경우 이상치로 판단. (정상 데이터는 잘 복원되지만, 이상 데이터는 잘 복원되지 않음)
- **생성 모델 (Generative Modeling)**: VAE와 같은 변형 모델을 사용하여 새로운 데이터 샘플 생성.
- **데이터 시각화**: 고차원 데이터를 2D 또는 3D 잠재 공간으로 압축하여 시각화.

## 6. 오토인코더 구현 시 고려사항
- **인코더/디코더 구조**: 레이어 수, 뉴런 수, 활성화 함수 등을 적절히 설계해야 합니다. 보통 대칭적인 구조를 많이 사용합니다 (인코더와 디코더가 반대 구조).
- **잠재 공간 차원**: 너무 작으면 정보 손실이 크고, 너무 크면 단순히 입력을 복사하는 것을 학습할 수 있습니다. 문제와 데이터에 따라 적절히 선택해야 합니다.
- **손실 함수 선택**: 데이터 유형에 맞는 손실 함수를 사용해야 합니다 (MSE, 교차 엔트로피 등).
- **과적합 방지**: 규제(L1, L2), 드롭아웃 등을 사용할 수 있으나, 오토인코더의 목적(정보 압축)과 상충될 수 있어 주의해야 합니다. 디노이징 오토인코더나 희소 오토인코더 자체가 일종의 규제 역할을 합니다.

## 추가 학습 자료
- [Building Autoencoders in Keras (Blog.keras.io)](https://blog.keras.io/building-autoencoders-in-keras.html)
- [Autoencoders Explained (YouTube - CodeEmporium)](https://www.youtube.com/watch?v=xTU79Zs4XKY)
- [Deep Learning Book - Chapter 14: Autoencoders (Ian Goodfellow, Yoshua Bengio, Aaron Courville)](https://www.deeplearningbook.org/contents/autoencoders.html)
- [An Introduction to Variational Autoencoders (VAEs) (Arxiv paper by Kingma & Welling, original authors)](https://arxiv.org/abs/1906.02691)

## 다음 학습 내용
- Day 97: 설명 가능한 AI (XAI) - LIME, SHAP (Explainable AI (XAI) - LIME, SHAP) - 복잡한 머신러닝 모델의 예측 결과를 이해하고 설명하기 위한 기법.
