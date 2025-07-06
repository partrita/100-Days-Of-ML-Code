# Day 97: 설명 가능한 AI (XAI) - LIME, SHAP (Explainable AI (XAI) - LIME, SHAP)

## 학습 목표
- 설명 가능한 AI (XAI)의 필요성과 중요성 이해.
- 모델 해석 가능성(Interpretability)과 설명 가능성(Explainability)의 차이 학습.
- 대표적인 XAI 기법인 LIME과 SHAP의 기본 아이디어와 작동 원리 이해.
- 각 기법의 장단점 및 활용 사례 파악.

## 1. 설명 가능한 AI (Explainable AI, XAI)란?
- **정의**: 인공지능(특히 머신러닝 및 딥러닝) 모델이 내린 결정이나 예측에 대해 **인간이 이해할 수 있는 형태로 설명과 이유를 제공**할 수 있도록 하는 기술 및 방법론입니다.
- **"블랙박스(Black Box)" 모델의 문제 해결**: 복잡한 딥러닝 모델이나 앙상블 모델은 내부 작동 방식을 이해하기 어려워 "블랙박스"로 취급되는 경우가 많습니다. XAI는 이러한 블랙박스 모델의 투명성을 높이는 것을 목표로 합니다.

### XAI의 필요성 및 중요성
- **신뢰성 및 투명성 확보**: 모델의 결정 과정을 이해함으로써 모델을 더 신뢰하고, 예측 결과에 대한 확신을 가질 수 있습니다.
- **디버깅 및 모델 개선**: 모델이 왜 특정 예측을 했는지 알면, 오류의 원인을 파악하고 모델을 개선하는 데 도움이 됩니다. (예: 잘못된 특징에 의존, 데이터 편향 등)
- **공정성 및 편향 탐지**: 모델이 특정 그룹에 대해 불공정한 예측을 하거나 편향된 결정을 내리는지 확인하고 수정할 수 있습니다. (예: 대출 심사, 채용)
- **규제 준수 및 책임성**: 금융, 의료 등 규제가 중요한 분야에서는 모델 결정에 대한 설명이 법적으로 요구될 수 있습니다. 문제 발생 시 책임 소재를 명확히 하는 데도 기여합니다.
- **새로운 지식 발견**: 모델이 학습한 패턴이나 중요한 특징을 파악함으로써 해당 도메인에 대한 새로운 통찰력을 얻을 수 있습니다.
- **사용자 수용성 증대**: 사용자가 AI의 결정을 이해하고 납득할 수 있을 때 AI 기술에 대한 수용성이 높아집니다.

## 2. 해석 가능성 (Interpretability) vs 설명 가능성 (Explainability)
- **해석 가능성 (Interpretability)**: 모델 자체가 내부적으로 어떻게 작동하는지, 각 구성 요소(파라미터, 특징 등)가 어떤 의미를 가지는지 인간이 이해할 수 있는 정도.
    - **화이트박스 모델 (White-box Models)**: 내부 구조가 투명하여 해석이 용이한 모델.
        - 예: 선형 회귀 (계수 확인), 로지스틱 회귀 (계수 확인), 결정 트리 (규칙 시각화).
- **설명 가능성 (Explainability)**: 모델의 내부 작동 방식과 관계없이, 특정 입력에 대한 모델의 예측 결과(출력)에 대해 "왜" 그런 결과가 나왔는지 사후적으로 설명하는 능력.
    - **블랙박스 모델에 대한 사후 설명 (Post-hoc Explanations)**: 이미 학습된 복잡한 모델에 대해 적용.
        - 예: LIME, SHAP.
- XAI는 주로 블랙박스 모델에 대한 설명 가능성을 높이는 데 초점을 맞추지만, 해석 가능한 모델을 사용하는 것도 XAI의 한 접근 방식입니다.

## 3. LIME (Local Interpretable Model-agnostic Explanations)
- **개념**: "로컬에서 해석 가능한 모델-불특정 설명". 복잡한 블랙박스 모델의 **개별 예측(Local)**에 대해, 그 예측 주변에서 **해석 가능한 모델(Interpretable Model, 예: 선형 회귀, 결정 트리)**을 사용하여 근사적으로 설명을 제공하는 기법입니다. (모델-불특정, Model-agnostic)
- **핵심 아이디어**: 아무리 복잡한 모델이라도, 특정 예측 지점의 매우 작은 "국소적(Local)" 영역에서는 선형 모델과 같이 단순한 모델로 근사할 수 있다는 가정에 기반합니다.

### LIME 작동 원리 (간단히)
1.  **설명 대상 샘플 선택**: 설명을 원하는 특정 데이터 샘플(x)과 그에 대한 블랙박스 모델의 예측값(f(x))을 선택합니다.
2.  **샘플 주변 데이터 생성**: 설명 대상 샘플(x) 주변에 약간의 변형(Perturbation)을 가하여 새로운 가상의 데이터 샘플들을 생성합니다.
3.  **생성된 샘플에 대한 예측**: 생성된 가상 샘플들에 대해 블랙박스 모델을 사용하여 예측값을 얻습니다.
4.  **가중치 부여**: 생성된 가상 샘플들 중 원래 샘플(x)에 가까울수록 높은 가중치를 부여합니다. (국소성을 반영)
5.  **해석 가능한 모델 학습**: 가중치가 부여된 (가상 샘플, 블랙박스 모델 예측값) 쌍을 사용하여, 해석 가능한 모델(예: 선형 회귀)을 이 국소적 영역에서 학습시킵니다.
6.  **설명 생성**: 학습된 해석 가능한 모델의 계수(Coefficients)나 규칙(Rules)을 통해, 원래 샘플(x)의 예측에 어떤 특징들이 얼마나 기여했는지를 설명합니다.
    - 예: (선형 회귀 사용 시) "특성 A의 값이 0.5 증가하면 예측 확률이 0.2만큼 높아지는 경향이 있다."

![LIME Explanation](https://christophm.github.io/interpretable-ml-book/images/lime-idea-1.png)
*(이미지 출처: Interpretable Machine Learning by Christoph Molnar)*

### LIME의 장단점
- **장점**:
    - **모델-불특정(Model-agnostic)**: 어떤 종류의 블랙박스 모델(딥러닝, 앙상블 등)에도 적용 가능.
    - **이해하기 쉬운 설명**: 해석 가능한 모델(선형 모델 등)을 사용하므로 설명이 직관적.
    - 텍스트, 이미지, 테이블 데이터 등 다양한 유형의 데이터에 적용 가능.
- **단점**:
    - **국소적 설명의 한계**: 전체 모델의 행동을 설명하는 것이 아니라 개별 예측에 대한 국소적 설명만 제공.
    - **샘플링 방식 및 가중치 설정의 민감성**: 주변 데이터 생성 방식이나 가중치 함수에 따라 설명의 안정성이 달라질 수 있음.
    - **해석 가능한 모델의 표현력 한계**: 국소적 영역이 실제로는 비선형적일 경우, 선형 모델로 근사하는 데 한계가 있을 수 있음.
    - "충실성-해석가능성 트레이드오프 (Fidelity-Interpretability Trade-off)": 설명을 단순하게 만들수록 원래 블랙박스 모델의 국소적 행동을 정확히 반영하지 못할 수 있음.

### LIME 파이썬 라이브러리
- `lime` (https://github.com/marcotcr/lime)

## 4. SHAP (SHapley Additive exPlanations)
- **개념**: 게임 이론(Game Theory)의 **섀플리 값(Shapley Value)** 개념을 머신러닝 모델 설명에 적용한 기법. 각 특징(Feature)이 특정 예측에 얼마나 기여했는지를 공정하게 배분하여 설명합니다.
- **핵심 아이디어**: 특정 특징이 예측값에 기여한 정도를, 해당 특징이 있을 때와 없을 때의 예측값 차이로 계산하되, 모든 가능한 특징 조합(순서)에 대해 평균을 내어 공평하게 기여도를 산출합니다.

### 섀플리 값 (Shapley Value)
- 협력 게임 이론에서, 여러 플레이어가 협력하여 얻은 전체 보상을 각 플레이어의 기여도에 따라 공정하게 분배하는 방법을 제시합니다.
- SHAP에서는 "플레이어" = "특징", "게임" = "모델 예측", "보상" = "예측값"에 해당합니다.

### SHAP 작동 원리 (개념적)
- 특정 예측값과 전체 데이터의 평균 예측값(Baseline)의 차이를 각 특징들의 기여도(SHAP 값)의 합으로 표현합니다:
  `f(x) - E[f(x)] = Σ (SHAP 값<sub>i</sub>)`
- 각 특징 i의 SHAP 값은, 해당 특징이 모델 예측에 긍정적 또는 부정적으로 얼마나 기여했는지를 나타냅니다.
- SHAP 값을 계산하기 위해, 특정 특징을 제외한 모든 가능한 특징 부분집합에 대해 모델 예측을 수행하고, 해당 특징이 추가되었을 때의 예측값 변화를 평균냅니다. (계산 비용이 매우 클 수 있음)
- 실제로는 KernelSHAP, TreeSHAP, DeepSHAP 등 모델 유형에 따라 효율적인 근사 계산 방법을 사용합니다.

![SHAP Force Plot](https://shap.readthedocs.io/en/latest/_images/boston_waterfall.png)
*(이미지 출처: SHAP Documentation - 개별 예측에 대한 SHAP 값 시각화 예시)*

### SHAP의 장단점
- **장점**:
    - **탄탄한 이론적 기반**: 게임 이론의 섀플리 값에 기반하여 공정하고 일관된 설명을 제공 (Local Accuracy, Missingness, Consistency 속성 만족).
    - **전역적 설명 가능**: 개별 예측에 대한 SHAP 값들을 집계하여 전체 모델에 대한 특징 중요도(Global Feature Importance)를 파악할 수 있습니다. (예: 각 특징의 SHAP 값 절대평균)
    - **특징 간 상호작용 효과 분석 가능 (SHAP Interaction Values)**.
    - 다양한 시각화 도구 제공 (Force Plot, Summary Plot, Dependence Plot 등).
    - 모델-불특정 방식(KernelSHAP)과 모델-특화 방식(TreeSHAP, DeepSHAP) 모두 지원.
- **단점**:
    - **계산 비용**: 특히 KernelSHAP과 같이 모델-불특정 방식은 계산량이 많아 시간이 오래 걸릴 수 있습니다. (TreeSHAP은 트리 기반 모델에 매우 효율적)
    - **해석의 복잡성**: 섀플리 값 자체의 개념이 다소 어려울 수 있습니다.
    - **가정**: 특징들이 서로 독립적이라는 가정이 암묵적으로 있을 수 있으며, 상관관계가 높은 특징들의 기여도 배분에 주의가 필요할 수 있습니다.

### SHAP 파이썬 라이브러리
- `shap` (https://github.com/slundberg/shap)

## 5. LIME vs SHAP

| 특징                | LIME                                                      | SHAP                                                              |
| ------------------- | --------------------------------------------------------- | ----------------------------------------------------------------- |
| **기반 이론**       | 국소적 선형 근사                                            | 게임 이론 (섀플리 값)                                               |
| **설명 대상**       | 개별 예측 (Local)                                         | 개별 예측 (Local) 및 전체 모델 (Global)                             |
| **일관성/공정성**   | 보장되지 않을 수 있음                                       | 이론적으로 보장 (Local Accuracy, Missingness, Consistency)          |
| **계산 속도**       | 상대적으로 빠름                                             | KernelSHAP은 느릴 수 있음, TreeSHAP/DeepSHAP은 빠름                  |
| **모델 적용 범위**  | 모델-불특정                                               | 모델-불특정(KernelSHAP) 및 모델-특화(TreeSHAP, DeepSHAP 등) 지원 |
| **주요 결과물**     | 국소적 선형 모델의 계수 (특징 가중치)                       | 각 특징의 SHAP 값 (예측 기여도)                                     |
| **상호작용 분석**   | 제한적                                                    | SHAP Interaction Values로 가능                                    |

- **언제 무엇을 사용할까?**
    - **빠르게 개별 예측에 대한 직관적인 설명을 얻고 싶을 때**: LIME
    - **이론적으로 탄탄하고 일관된 설명을 원하며, 전역적인 특징 중요도나 상호작용 효과까지 분석하고 싶을 때**: SHAP (특히 트리 기반 모델에는 TreeSHAP이 매우 효율적)
    - 두 가지를 함께 사용하여 상호 보완적으로 활용하는 것도 좋은 방법입니다.

## 추가 학습 자료
- [Interpretable Machine Learning by Christoph Molnar (온라인 책)](https://christophm.github.io/interpretable-ml-book/) - (XAI 분야의 교과서적인 자료)
- ["Why Should I Trust You?": Explaining the Predictions of Any Classifier (LIME 논문)](https://arxiv.org/abs/1602.04938)
- [A Unified Approach to Interpreting Model Predictions (SHAP 논문)](https://arxiv.org/abs/1705.07874)
- [Beware of Simpler Models: LIME and SHAP have a Causal Interpretation Problem (블로그 글, 비판적 시각)](https://towardsdatascience.com/be-careful-when-interpreting-lime-and-shap-results-for-feature-importance-2e147a844826) - (XAI 기법 사용 시 주의점)

## 다음 학습 내용
- Day 98: AI 및 머신러닝 윤리 (Ethics in AI and Machine Learning) - AI 기술 발전과 함께 중요성이 커지는 윤리적 문제들과 책임감 있는 AI 개발.
