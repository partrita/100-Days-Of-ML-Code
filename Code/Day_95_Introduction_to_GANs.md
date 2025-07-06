# Day 95: 생성적 적대 신경망 (GAN) 소개 (Introduction to Generative Adversarial Networks (GANs))

## 학습 목표
- 생성 모델(Generative Model)의 개념과 종류 이해.
- GAN (Generative Adversarial Network, 생성적 적대 신경망)의 기본 아이디어와 작동 원리 학습.
    - 생성자 (Generator)와 판별자 (Discriminator)의 역할.
    - 두 네트워크 간의 적대적(Adversarial) 학습 과정.
- GAN의 학습 과정과 손실 함수(Loss Function)의 개념 이해.
- GAN의 주요 응용 분야 및 한계점 인식.

## 1. 생성 모델 (Generative Model)이란?
- **정의**: 학습 데이터의 분포를 학습하여, 기존 데이터와 유사하지만 **새로운 데이터를 생성**할 수 있는 모델.
- **목표**: 데이터가 어떻게 생성되었는지 그 근본적인 확률 분포 p(x)를 모델링하는 것.
- **판별 모델 (Discriminative Model)과의 비교**:
    - **판별 모델**: 데이터 x가 주어졌을 때 레이블 y를 예측하는 조건부 확률 p(y|x)를 학습. (예: 이미지 분류기 - 이 이미지가 고양이인지 개인지 판별)
    - **생성 모델**: 데이터 x 자체의 분포 p(x)를 학습하거나, 특정 레이블 y가 주어졌을 때 데이터 x를 생성하는 p(x|y)를 학습. (예: 고양이 이미지를 새로 생성)

### 주요 생성 모델 종류
- **명시적 밀도 추정 (Explicit Density Estimation)**: 데이터의 확률 밀도 함수 p(x)를 명시적으로 정의하고 학습.
    - 예: PixelRNN, PixelCNN, Variational Autoencoder (VAE) - 일부
- **암시적 밀도 추정 (Implicit Density Estimation)**: 데이터의 확률 밀도 함수를 직접 정의하지 않고, 샘플링 과정을 통해 새로운 데이터를 생성.
    - 예: **Generative Adversarial Network (GAN)**

## 2. GAN (Generative Adversarial Network) 기본 아이디어
- 2014년 Ian Goodfellow 등에 의해 제안된 생성 모델의 한 종류로, 두 개의 신경망이 서로 경쟁하며 학습하는 독특한 구조를 가집니다.
- **핵심 아이디어**: "경찰과 위조지폐범"의 비유
    - **위조지폐범 (생성자, Generator)**: 진짜와 최대한 유사한 위조지폐를 만들려고 노력.
    - **경찰 (판별자, Discriminator)**: 위조지폐와 진짜 지폐를 최대한 잘 구별하려고 노력.
- 이 두 네트워크는 서로 적대적인(Adversarial) 관계에서 경쟁적으로 학습하며, 이 과정에서 생성자는 점점 더 진짜 같은 데이터를 만들게 되고, 판별자는 점점 더 정교하게 진짜와 가짜를 구별하게 됩니다.

## 3. GAN의 구성 요소 및 작동 원리

### 가. 생성자 (Generator, G)
- **역할**: 실제 데이터와 유사한 가짜(Fake) 데이터를 생성.
- **입력**: 무작위 노이즈 벡터 (Random Noise Vector, z). (보통 정규분포나 균등분포에서 샘플링)
- **출력**: 생성된 가짜 데이터 (예: 이미지, 텍스트).
- **목표**: 판별자가 생성된 데이터를 진짜 데이터로 착각하도록 만드는 것. (즉, 판별자를 속이는 것)
- **구조**: 일반적으로 역컨볼루션(Deconvolution) 또는 업샘플링(Upsampling) 레이어를 포함하는 심층 신경망 (예: DCGAN의 경우).

### 나. 판별자 (Discriminator, D)
- **역할**: 입력된 데이터가 진짜(Real) 데이터인지 아니면 생성자(G)가 만든 가짜(Fake) 데이터인지 판별.
- **입력**: 진짜 데이터 또는 생성자가 만든 가짜 데이터.
- **출력**: 입력된 데이터가 진짜일 확률 (0과 1 사이의 스칼라 값). (1에 가까우면 진짜, 0에 가까우면 가짜로 판별)
- **목표**: 진짜 데이터와 가짜 데이터를 최대한 정확하게 구별하는 것.
- **구조**: 일반적으로 컨볼루션(Convolution) 레이어를 포함하는 심층 신경망 (이미지 분류기와 유사).

### 다. 적대적 학습 과정 (Adversarial Training Process)
1.  **판별자(D) 학습 단계**:
    - 생성자(G)는 고정된 상태로 둡니다.
    - 실제 데이터셋에서 샘플(x<sub>real</sub>)을 가져오고, 생성자(G)를 통해 가짜 데이터(x<sub>fake</sub> = G(z))를 생성합니다.
    - 판별자(D)는 x<sub>real</sub>에 대해서는 높은 확률(1에 가깝게)을 출력하고, x<sub>fake</sub>에 대해서는 낮은 확률(0에 가깝게)을 출력하도록 학습합니다.
    - 즉, 판별자는 진짜와 가짜를 잘 구별하도록 손실 함수를 최소화하는 방향으로 업데이트됩니다.

2.  **생성자(G) 학습 단계**:
    - 판별자(D)는 고정된 상태로 둡니다.
    - 생성자(G)는 무작위 노이즈(z)로부터 가짜 데이터(x<sub>fake</sub> = G(z))를 생성합니다.
    - 생성된 x<sub>fake</sub>를 판별자(D)에 입력하여, 판별자가 이를 진짜 데이터로 착각하도록(즉, D(G(z))가 1에 가까워지도록) 학습합니다.
    - 즉, 생성자는 판별자를 속이도록 손실 함수를 최소화(또는 특정 형태의 손실 함수를 최대화)하는 방향으로 업데이트됩니다.

- 위 두 단계를 번갈아 반복하면서 학습을 진행합니다.
- 이상적으로는 학습이 진행됨에 따라 생성자는 점점 더 실제 데이터와 구분하기 어려운 데이터를 생성하게 되고, 판별자는 진짜와 가짜를 구별하는 능력이 향상되다가, 결국 생성된 데이터가 실제 데이터와 매우 유사해져 판별자가 더 이상 구별하기 어려운 상태(내쉬 균형, Nash Equilibrium)에 도달하는 것을 목표로 합니다. (실제로는 이 균형에 도달하기 어려울 수 있음)

![GAN Training Process](https://developers.google.com/machine-learning/gan/images/gan_diagram.svg)
*(이미지 출처: Google Developers - GANs Overview)*

## 4. GAN의 손실 함수 (Loss Function) - Minimax Game
- GAN의 학습은 생성자(G)와 판별자(D) 간의 **최소최대 게임(Minimax Game)**으로 볼 수 있습니다.
- **판별자(D)의 손실 함수 (L<sub>D</sub>)**: 실제 데이터에 대해서는 log(D(x))를 최대화하고, 가짜 데이터에 대해서는 log(1 - D(G(z)))를 최대화합니다. (즉, D(x)는 1로, D(G(z))는 0으로 만들려고 함)
    - L<sub>D</sub> = -E<sub>x~p<sub>data</sub>(x)</sub>[log D(x)] - E<sub>z~p<sub>z</sub>(z)</sub>[log(1 - D(G(z)))]
    - (D는 이 손실을 최소화하려고 함)
- **생성자(G)의 손실 함수 (L<sub>G</sub>)**: 판별자가 가짜 데이터를 진짜로 착각하도록, 즉 D(G(z))를 최대화(또는 log(1 - D(G(z)))를 최소화)하려고 합니다.
    - 초기 논문에서는 L<sub>G</sub> = E<sub>z~p<sub>z</sub>(z)</sub>[log(1 - D(G(z)))] (최소화)
    - 실제 구현에서는 학습 초기 그래디언트 소실 문제를 피하기 위해 L<sub>G</sub> = -E<sub>z~p<sub>z</sub>(z)</sub>[log D(G(z))] (최소화, 즉 D(G(z))를 최대화)를 더 많이 사용합니다. (Non-saturating game)

- **전체 목적 함수 (Value Function, V(D,G))**:
    V(D,G) = E<sub>x~p<sub>data</sub>(x)</sub>[log D(x)] + E<sub>z~p<sub>z</sub>(z)</sub>[log(1 - D(G(z)))]
    - 생성자 G는 이 V(D,G)를 최소화(min<sub>G</sub>)하려고 하고, 판별자 D는 이 V(D,G)를 최대화(max<sub>D</sub>)하려고 합니다.
    - min<sub>G</sub> max<sub>D</sub> V(D,G)

## 5. GAN의 주요 응용 분야
- **이미지 생성**: 고품질의 새로운 이미지 생성 (예: 유명인 얼굴, 동물, 풍경 등).
- **이미지 변환 (Image-to-Image Translation)**: 한 스타일의 이미지를 다른 스타일로 변환 (예: 스케치를 그림으로, 낮 사진을 밤 사진으로 - Pix2Pix, CycleGAN).
- **초해상화 (Super-Resolution)**: 저해상도 이미지를 고해상도 이미지로 변환.
- **텍스트-이미지 변환 (Text-to-Image Synthesis)**: 텍스트 설명을 바탕으로 이미지 생성.
- **데이터 증강 (Data Augmentation)**: 부족한 학습 데이터를 보충하기 위해 새로운 데이터 생성.
- **스타일 전이 (Style Transfer)**: 특정 스타일(예: 특정 화가의 화풍)을 다른 이미지에 적용.
- **비디오 생성 및 편집**.
- **신약 개발, 분자 구조 생성 등 과학 분야 응용**.

## 6. GAN의 한계점 및 과제
- **학습 불안정성 (Training Instability)**:
    - 생성자와 판별자 간의 균형을 맞추기 어려워 학습이 불안정하거나 수렴하지 않을 수 있습니다.
    - **모드 붕괴 (Mode Collapse)**: 생성자가 다양한 종류의 데이터를 생성하지 못하고, 판별자를 속이기 쉬운 소수의 특정 데이터(모드)만 반복적으로 생성하는 현상.
    - **기울기 소실 (Vanishing Gradients)**: 판별자가 너무 뛰어나면 생성자가 학습할 유용한 그래디언트를 받지 못할 수 있습니다.
- **평가의 어려움**: 생성된 결과물의 품질을 정량적으로 평가하기 위한 명확한 지표가 부족합니다. (주관적인 시각적 평가에 의존하는 경우 많음. IS, FID 등 지표 사용)
- **하이퍼파라미터 튜닝의 어려움**: 적절한 하이퍼파라미터 조합을 찾는 것이 까다로울 수 있습니다.
- **다양한 변형 모델**: 위 한계점들을 극복하기 위해 DCGAN, WGAN, LSGAN, StyleGAN, BigGAN 등 수많은 변형 GAN 모델들이 제안되었습니다.

## 추가 학습 자료
- [NIPS 2016 Tutorial: Generative Adversarial Networks (Ian Goodfellow)](https://arxiv.org/abs/1701.00160) - (GAN 창시자의 튜토리얼 논문)
- [GANs from Scratch 1: A deep introduction. (YouTube - deeplizard)](https://www.youtube.com/watch?v=H9cHvV5nK4M)
- [Generative Adversarial Networks (GANs) Specialization on Coursera (DeepLearning.AI)](https://www.coursera.org/specializations/generative-adversarial-networks-gans)
- [GAN Lab (Google AI Experiments) - GAN 학습 과정 시각화 도구](https://poloclub.github.io/ganlab/)

## 다음 학습 내용
- Day 96: 오토인코더 (Autoencoders) - 데이터 압축 및 특징 추출에 사용되는 또 다른 중요한 비지도 학습 신경망. GAN과 비교되는 경우도 있음.
