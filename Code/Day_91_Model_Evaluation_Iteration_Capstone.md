# Day 91: 프로젝트를 위한 모델 평가 및 반복 (Model Evaluation and Iteration for the project)

## 학습 목표
- 초기 학습 및 교차 검증을 통해 선택된 주요 모델(들)에 대해 테스트 세트(Test Set)를 사용하여 최종 성능 평가.
- 다양한 평가지표(정밀도, 재현율, F1 점수, ROC AUC, 오차 행렬 등)를 종합적으로 분석하여 모델의 강점과 약점 파악.
- 오류 분석(Error Analysis)을 통해 모델이 어떤 유형의 샘플에서 실수를 하는지 구체적으로 조사.
- 분석 결과를 바탕으로 모델 개선을 위한 반복(Iteration) 전략 수립:
    - 특징 공학 추가/수정
    - 데이터 추가 수집 또는 증강(Augmentation)
    - 하이퍼파라미터 미세 조정
    - 다른 모델 아키텍처 시도

## 1. 테스트 세트를 사용한 최종 성능 평가
- 모델 선택 및 하이퍼파라미터 튜닝 과정에서는 **학습 데이터(Train Data)**와 (교차) **검증 데이터(Validation Data)**만을 사용해야 합니다.
- **테스트 세트(Test Set)**는 모델 개발의 가장 마지막 단계에서, 최종적으로 선택되고 튜닝된 모델의 일반화 성능을 딱 한 번 평가하는 데 사용됩니다.
- 테스트 세트의 성능은 모델이 실제 운영 환경에서 보일 것으로 기대되는 성능의 추정치가 됩니다.

```python
# Day 90에서 준비된 X_test_tfidf, y_test (또는 X_test_pad 등) 사용
# Day 90에서 선택된 최적 모델 (예: best_model_from_grid_search 또는 tuned_lstm_model) 사용

# --- 예시: Day 90의 Logistic Regression 모델을 최종 선택했다고 가정 ---
# (실제로는 하이퍼파라미터 튜닝까지 완료된 모델이어야 함)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd # 결과 정리를 위해

# (Day 90에서 학습된 최적 모델을 불러오거나, 여기서 간단히 재학습 가정)
# 여기서는 Day 90의 초기 모델 중 하나를 예시로 사용
# 실제로는 하이퍼파라미터 튜닝된 모델이어야 합니다.
# best_model = results_df.index[0] # 가장 성능 좋았던 모델 이름
# print(f"평가를 위해 선택된 모델: {best_model}") # 예시: RandomForestClassifier

# (데이터 준비 - Day 90의 X_train_tfidf, y_train, X_test_tfidf, y_test 사용)
# --- 간략화된 예시 데이터 및 모델 (실제 프로젝트 데이터 및 튜닝된 모델 사용) ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

data_for_modeling = {
    'processed_text': [
        "movie great wonderful", "terrible film hated", "not bad could better",
        "amazing masterpiece cinema", "okay nothing special write home",
        "fantastic story engaging characters", "boring plot uninspired acting",
        "truly enjoyed this experience", "would not recommend this", "a must see film"
    ],
    'sentiment': [1, 0, 0, 1, 0, 1, 0, 1, 0, 1],
}
df_model = pd.DataFrame(data_for_modeling)
X_text_data = df_model['processed_text']
y_target = df_model['sentiment']

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text_data, y_target, test_size=0.2, random_state=42, stratify=y_target
)
tfidf_vectorizer = TfidfVectorizer(max_features=100)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

# 최종 평가할 모델 (예: 로지스틱 회귀 - 실제로는 튜닝된 모델)
final_model = LogisticRegression(solver='liblinear', random_state=42)
final_model.fit(X_train_tfidf, y_train) # 학습 데이터 전체로 최종 학습
# --- 간략화된 예시 데이터 및 모델 끝 ---


# 1. 테스트 세트에 대한 예측
y_pred_test = final_model.predict(X_test_tfidf)
try:
    y_pred_proba_test = final_model.predict_proba(X_test_tfidf)[:, 1] # Positive 클래스 확률
except AttributeError: # 일부 모델은 predict_proba를 지원 안 할 수 있음 (예: SVM 기본)
    y_pred_proba_test = None
    print("선택된 모델이 predict_proba를 지원하지 않아 ROC AUC 계산이 불가능할 수 있습니다.")


# 2. 평가지표 계산
accuracy_test = accuracy_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test, average='macro', zero_division=0)
recall_test = recall_score(y_test, y_pred_test, average='macro', zero_division=0)
f1_test = f1_score(y_test, y_pred_test, average='macro', zero_division=0)

print("\n--- 최종 모델 테스트 세트 평가 결과 ---")
print(f"정확도 (Accuracy): {accuracy_test:.4f}")
print(f"정밀도 (Precision, Macro): {precision_test:.4f}")
print(f"재현율 (Recall, Macro): {recall_test:.4f}")
print(f"F1 점수 (F1 Score, Macro): {f1_test:.4f}")

if y_pred_proba_test is not None:
    try:
        roc_auc_test = roc_auc_score(y_test, y_pred_proba_test) # 이진 분류 시
        # 다중 클래스인 경우: roc_auc_score(y_test, final_model.predict_proba(X_test_tfidf), multi_class='ovr' or 'ovo')
        print(f"ROC AUC: {roc_auc_test:.4f}")
    except ValueError as e: # 예: 단일 클래스만 예측된 경우 등
        print(f"ROC AUC 계산 중 오류: {e}")
        roc_auc_test = None
else:
    roc_auc_test = None


# 3. 오차 행렬 (Confusion Matrix) 시각화
cm_test = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
            xticklabels=final_model.classes_ if hasattr(final_model, 'classes_') else ['Neg', 'Pos'],
            yticklabels=final_model.classes_ if hasattr(final_model, 'classes_') else ['Neg', 'Pos'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix on Test Set')
plt.show()

# 4. 분류 리포트 (Classification Report)
print("\n분류 리포트 (Classification Report):\n", classification_report(y_test, y_pred_test, zero_division=0))
```

## 2. 오류 분석 (Error Analysis)
- 모델이 어떤 부분에서 실수를 하는지 구체적으로 살펴보는 과정입니다. 이는 모델 개선에 매우 중요한 단서를 제공합니다.
- **주요 방법**:
    - **오차 행렬 상세 분석**:
        - **FN (False Negative)**: 실제 Positive인데 Negative로 잘못 예측한 경우. (예: 실제 긍정 리뷰를 부정으로) 왜 이런 실수가 발생했을까?
        - **FP (False Positive)**: 실제 Negative인데 Positive로 잘못 예측한 경우. (예: 실제 부정 리뷰를 긍정으로) 왜 이런 실수가 발생했을까?
    - **잘못 분류된 샘플 직접 확인**: 모델이 틀린 예측을 한 실제 데이터 샘플(텍스트, 이미지 등)을 직접 살펴봅니다.
        - 어떤 특징을 가진 샘플에서 주로 오류가 발생하는가?
        - 데이터 레이블링 자체에 오류는 없는가?
        - 모델이 이해하기 어려운 애매모호한 표현, 비꼬는 말투, 문맥 의존적인 내용 등이 포함되어 있는가?
        - 특정 클래스에서 유독 오류가 많이 발생하는가? (클래스 불균형 문제와 연관 가능)
    - **예측 확률(Confidence Score) 분석**:
        - 모델이 매우 높은 확률로 틀린 예측을 한 경우.
        - 모델이 애매하게(예측 확률이 0.5 근처) 예측하여 틀린 경우.
    - **특성 중요도(Feature Importance) 또는 모델 내부 가중치 확인 (해석 가능한 모델의 경우)**:
        - 모델이 어떤 특징에 의존하여 예측하는지, 잘못된 특징에 과도하게 의존하고 있지는 않은지 확인.
        - (NLP) 어텐션 가중치 시각화 (트랜스포머, 어텐션 기반 RNN 등).

```python
# 오류 분석 예시 (잘못 분류된 샘플 확인)
# (X_test_text, y_test, y_pred_test 가 준비되어 있다고 가정)

# 데이터프레임으로 만들어 분석 용이하게
df_test_results = pd.DataFrame({'text': X_test_text, 'actual': y_test, 'predicted': y_pred_test})

# FN (실제:1, 예측:0) 과 FP (실제:0, 예측:1) 샘플 분리
false_negatives = df_test_results[(df_test_results['actual'] == 1) & (df_test_results['predicted'] == 0)]
false_positives = df_test_results[(df_test_results['actual'] == 0) & (df_test_results['predicted'] == 1)]

print(f"\n--- 오류 분석: False Negatives (실제 긍정 -> 예측 부정) ---")
if not false_negatives.empty:
    for index, row in false_negatives.head().iterrows(): # 상위 몇 개만 출력
        print(f"  실제 텍스트: {row['text']}")
        print(f"  실제 레이블: {row['actual']}, 예측 레이블: {row['predicted']}\n")
else:
    print("  False Negative 샘플이 없습니다.")

print(f"\n--- 오류 분석: False Positives (실제 부정 -> 예측 긍정) ---")
if not false_positives.empty:
    for index, row in false_positives.head().iterrows():
        print(f"  실제 텍스트: {row['text']}")
        print(f"  실제 레이블: {row['actual']}, 예측 레이블: {row['predicted']}\n")
else:
    print("  False Positive 샘플이 없습니다.")

# (선택) 예측 확률과 함께 보기 (y_pred_proba_test 사용)
# if y_pred_proba_test is not None:
#     df_test_results['probability_positive'] = y_pred_proba_test
#     # 예측 확률이 0.4~0.6 사이인 애매한 예측들 확인 등
```

## 3. 모델 개선을 위한 반복 (Iteration) 전략
- 테스트 세트 평가와 오류 분석 결과를 바탕으로 모델을 개선하기 위한 다음 단계를 결정합니다.
- 이 과정은 한 번에 끝나지 않고, 여러 번 반복될 수 있습니다 (Iterative Process).

### 가. 데이터 관점
- **데이터 추가 수집**: 특정 유형의 오류가 많이 발생한다면, 해당 유형의 데이터를 더 많이 수집하여 모델이 학습하도록 합니다. (예: 비꼬는 표현이 포함된 리뷰 데이터 추가)
- **데이터 증강 (Data Augmentation)**: (특히 딥러닝, 이미지/텍스트) 기존 데이터를 변형하여 학습 데이터의 양과 다양성을 늘립니다.
    - 텍스트: 역번역(Back-translation), 동의어 대체, 무작위 삽입/삭제 등.
- **데이터 레이블링 품질 개선**: 레이블링 오류가 발견되면 수정합니다. 모호한 샘플에 대한 레이블링 가이드라인을 명확히 합니다.
- **클래스 불균형 해소**: 소수 클래스 데이터에 대한 오버샘플링(SMOTE 등), 다수 클래스 데이터에 대한 언더샘플링, 또는 손실 함수 가중치 조절 등을 시도합니다.

### 나. 특징 공학 관점
- **새로운 특징 생성**: 오류 분석을 통해 얻은 인사이트를 바탕으로 모델이 더 잘 일반화할 수 있는 새로운 특징을 만듭니다.
    - 예: (감성 분석) 문장 내 부정어의 존재 여부, 감정 단어의 강도, 특정 문장 부호의 빈도 등을 특징으로 추가.
- **기존 특징 수정/제거**: 중요도가 낮거나 노이즈로 작용하는 특징을 제거하거나, 다른 방식으로 변환합니다.
- **특징 스케일링 재검토**: 사용한 스케일링 방법이 적절했는지 확인합니다.

### 다. 모델 관점
- **하이퍼파라미터 미세 조정 (Fine-tuning)**: Day 75에서 배운 그리드 서치, 랜덤 서치, 베이지안 최적화 등을 사용하여 현재 모델의 하이퍼파라미터를 더 정교하게 튜닝합니다. (교차 검증 사용)
- **다른 모델 아키텍처 시도**:
    - 현재 모델군 외에 다른 유형의 모델을 시도해봅니다. (예: 전통적 ML 모델에서 딥러닝 모델로, 또는 그 반대)
    - (딥러닝) 레이어 추가/제거, 유닛 수 변경, 다른 활성화 함수나 옵티마이저 사용, 어텐션 메커니즘 도입 등.
- **앙상블 기법 활용**: 여러 모델의 예측을 결합하여 성능을 향상시킵니다. (배깅, 부스팅, 스태킹)
- **전이 학습 (Transfer Learning)**: (특히 딥러닝, NLP/비전) 대규모 데이터로 사전 훈련된 모델을 가져와 현재 문제에 맞게 미세 조정합니다.

### 라. 반복 주기
1.  **가설 수립**: 오류 분석 결과를 바탕으로 "어떤 변경이 모델 성능을 향상시킬 것이다"라는 가설을 세웁니다.
2.  **실험**: 가설에 따라 데이터, 특징, 모델 등을 수정하고 다시 학습 및 (검증 세트) 평가를 수행합니다.
3.  **결과 분석**: 변경 사항이 실제로 성능 향상에 기여했는지, 새로운 문제는 없는지 분석합니다.
4.  **반복 또는 종료**: 목표 성능에 도달했거나 더 이상 개선의 여지가 크지 않다고 판단되면 반복을 종료합니다. 그렇지 않으면 다시 1단계로 돌아갑니다.

**주의**: 모델 개선을 위한 모든 실험과 변경은 **학습 데이터와 검증 데이터를 사용**하여 이루어져야 합니다. 테스트 세트는 최종 평가에만 사용되어야 그 의미가 있습니다. 만약 테스트 세트를 반복적으로 사용하여 모델을 수정한다면, 테스트 세트에 과적합될 위험이 있습니다.

## 실습 아이디어
- 본인의 캡스톤 프로젝트에서 Day 90에 선택한 주요 모델(들)에 대해 테스트 세트 성능을 상세히 평가해보세요.
    - 다양한 평가지표(정확도, 정밀도, 재현율, F1, ROC AUC 등)를 계산하고 기록합니다.
    - 오차 행렬을 시각화하고, 분류 리포트를 출력하여 각 클래스별 성능을 확인합니다.
- 모델이 잘못 예측한 샘플들(FN, FP)을 최소 5~10개 이상 직접 살펴보세요.
    - 어떤 특징을 가진 샘플에서 오류가 주로 발생하는지 패턴을 찾아보세요.
    - 가능한 오류의 원인(데이터 문제, 특징 부족, 모델 한계 등)을 추측해보세요.
- 오류 분석 결과를 바탕으로, 모델 성능을 개선하기 위한 아이디어를 2~3가지 이상 구체적으로 정리해보세요. (예: "부정어를 포함한 문장에 대한 특징을 추가한다", "특정 단어들에 대한 임베딩을 개선한다", "LSTM 모델의 유닛 수를 늘려본다" 등)

## 다음 학습 내용
- Day 92: 프로젝트를 위한 미세 조정 및 최적화 (Fine-tuning and Optimization for the project) - 오류 분석 및 개선 아이디어를 바탕으로 실제 모델 하이퍼파라미터 튜닝, 특징 공학 수정 등을 통해 모델 성능을 최적화하는 작업.
