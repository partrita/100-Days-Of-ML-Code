# Day 89: 프로젝트를 위한 탐색적 데이터 분석 (EDA) 심화 및 특징 공학 (Exploratory Data Analysis (EDA) for the project and Feature Engineering)

## 학습 목표
- 어제 수집 및 기본 전처리된 데이터를 바탕으로 심층적인 탐색적 데이터 분석(EDA) 수행.
- 데이터 시각화 및 통계적 분석을 통해 데이터 내 패턴, 관계, 이상 현상 등을 깊이 있게 파악.
- EDA 결과를 바탕으로 모델 성능 향상에 기여할 수 있는 새로운 특징(Feature)을 생성하거나 기존 특징을 변환하는 특징 공학(Feature Engineering) 기법 적용.
- 최종적으로 모델 학습에 사용할 데이터셋 준비.

## 1. 탐색적 데이터 분석 (EDA) 심화
- 어제 수행한 기초 EDA를 바탕으로, 더 구체적인 질문을 던지고 데이터로부터 인사이트를 얻는 과정입니다.

### 가. 분석 질문 정의
- 프로젝트 목표와 관련된 구체적인 질문들을 정의합니다.
- 예시 (영화 리뷰 감성 분석 프로젝트):
    - 긍정 리뷰와 부정 리뷰는 어떤 단어 사용에서 차이를 보이는가?
    - 리뷰 길이는 감성과 관련이 있는가?
    - 특정 단어가 포함된 리뷰는 특정 감성으로 치우치는 경향이 있는가?
    - (만약 사용자 정보가 있다면) 특정 사용자 그룹이 특정 감성의 리뷰를 더 많이 작성하는가?
    - (만약 영화 장르 정보가 있다면) 특정 장르의 영화에 대한 감성 분포는 어떠한가?

### 나. 데이터 시각화 심화
- **단변량 분석 (Univariate Analysis)**: 개별 특성의 분포를 더 자세히 탐색.
    - 수치형: 히스토그램(구간 조정), 밀도 플롯(Density Plot), QQ 플롯(정규성 확인).
    - 범주형: 막대 그래프(빈도, 비율), 파이 차트.
    - 텍스트: N-gram(바이그램, 트라이그램) 빈도 분석, 핵심 단어 추출.
- **이변량 분석 (Bivariate Analysis)**: 두 특성 간의 관계 탐색.
    - 수치형 vs 수치형: 산점도(Scatter Plot), 상관계수 히트맵.
    - 범주형 vs 수치형: 그룹별 박스 플롯, 그룹별 바이올린 플롯, 그룹별 평균 막대 그래프.
    - 범주형 vs 범주형: 교차표(Contingency Table), 그룹화된 막대 그래프, 모자이크 플롯.
    - 타겟 변수와 각 특성 간의 관계 분석이 중요.
- **다변량 분석 (Multivariate Analysis)**: 세 개 이상의 특성 간 관계 탐색.
    - 산점도에 색상이나 크기로 세 번째 변수 표현.
    - Pair Plot (Seaborn의 `pairplot`): 모든 수치형 특성 쌍에 대한 산점도와 대각선에는 각 특성의 분포 표시.
    - 3D 산점도 (표현의 한계가 있음).
    - 차원 축소(PCA 등) 후 시각화.

```python
# EDA 심화 예시 (어제 코드 이어서)
# (데이터프레임 df가 전처리된 상태라고 가정)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
# from nltk.util import ngrams # N-gram 생성용

# --- 예시 데이터 (어제와 유사하게, 'processed_text' 컬럼이 있다고 가정) ---
data_dict_processed = {
    'text': ["movie great wonderful", "terrible film hated", "not bad could better", "amazing masterpiece cinema", "", "okay nothing special write home"],
    'sentiment': [1, 0, 0, 1, 1, 0],
    'rating': [5, 1, 3, 5, 4, 3],
    'text_length': [20, 20, 20, 25, 0, 35] # 임의의 길이
}
df = pd.DataFrame(data_dict_processed)
df.rename(columns={'text':'processed_text'}, inplace=True) # processed_text로 가정
# --- 예시 데이터 끝 ---


# 1. 긍정/부정 리뷰별 단어 빈도 비교 (간단 예시)
if 'processed_text' in df.columns and 'sentiment' in df.columns:
    positive_texts = " ".join(df[df['sentiment'] == 1]['processed_text'])
    negative_texts = " ".join(df[df['sentiment'] == 0]['processed_text'])

    positive_word_counts = Counter(positive_texts.split())
    negative_word_counts = Counter(negative_texts.split())

    print("\n가장 흔한 긍정 단어 (상위 5개):", positive_word_counts.most_common(5))
    print("가장 흔한 부정 단어 (상위 5개):", negative_word_counts.most_common(5))

    # (선택) N-gram 분석
    # positive_bigrams = list(ngrams(positive_texts.split(), 2))
    # positive_bigram_counts = Counter(positive_bigrams)
    # print("\n가장 흔한 긍정 바이그램 (상위 3개):", positive_bigram_counts.most_common(3))

# 2. 리뷰 길이와 감성 간의 관계 (박스 플롯)
if 'text_length' in df.columns and 'sentiment' in df.columns:
    plt.figure(figsize=(7, 5))
    sns.boxplot(x='sentiment', y='text_length', data=df)
    plt.title('Text Length Distribution by Sentiment')
    plt.xticks([0, 1], ['Negative', 'Positive'])
    plt.show()

# 3. 평점(rating)과 감성(sentiment) 간의 관계 (교차표 및 시각화)
if 'rating' in df.columns and 'sentiment' in df.columns:
    rating_sentiment_crosstab = pd.crosstab(df['rating'], df['sentiment'])
    print("\n평점과 감성 간 교차표:\n", rating_sentiment_crosstab)

    rating_sentiment_crosstab.plot(kind='bar', stacked=False, figsize=(8,5))
    plt.title('Sentiment Distribution by Rating')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.legend(title='Sentiment', labels=['Negative', 'Positive'])
    plt.show()

# 4. (만약 수치형 특성이 더 있다면) Pair Plot
# num_features_df = df[['text_length', 'rating']] # 예시 수치형 특성
# if not num_features_df.empty:
#     sns.pairplot(num_features_df)
#     plt.suptitle('Pair Plot of Numerical Features', y=1.02)
#     plt.show()

# 5. (만약 수치형 특성이 더 있다면) 상관관계 히트맵
# if not num_features_df.empty:
#     plt.figure(figsize=(6,4))
#     sns.heatmap(num_features_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
#     plt.title('Correlation Heatmap of Numerical Features')
#     plt.show()
```

## 2. 특징 공학 (Feature Engineering)
- **정의**: 도메인 지식이나 EDA를 통해 얻은 통찰력을 바탕으로, 기존 특징으로부터 새로운 특징을 만들거나, 특징을 변환하여 모델의 성능을 향상시키는 과정.
- 머신러닝 프로젝트 성공에 매우 중요한 단계 중 하나입니다. "Garbage in, garbage out" - 좋은 특징이 좋은 모델을 만듭니다.

### 가. 특징 생성 (Feature Creation)
- 기존 특징들을 결합하거나 변형하여 새로운 정보를 담는 특징을 만듭니다.
- **예시**:
    - **날짜/시간 데이터**: 연, 월, 일, 요일, 시간, 분기, 주말 여부, 공휴일 여부 등 파생 변수 생성.
    - **텍스트 데이터**:
        - **텍스트 길이**: 문장/문서의 단어 수, 문자 수.
        - **특정 단어/구문 포함 여부**: 긍정/부정 사전 단어 포함 수, 특정 키워드 존재 여부.
        - **N-gram 특징**: 단어의 순서를 일부 고려 (예: "not good"은 "not"과 "good"을 따로 보는 것보다 의미 파악에 유리).
        - **가독성 지수**: 텍스트의 복잡도나 가독성을 나타내는 지표.
        - **문장 부호 사용 빈도**: 물음표, 느낌표 등의 빈도.
    - **수치형 데이터**:
        - **다항 특성 (Polynomial Features)**: 기존 특성들의 거듭제곱 또는 교호작용 항 (예: x<sup>2</sup>, x*y). `sklearn.preprocessing.PolynomialFeatures`.
        - **비율 또는 차이**: 두 특성 간의 비율(예: 소득 대비 부채 비율)이나 차이.
        - **그룹별 통계량**: 특정 그룹(예: 카테고리) 내에서의 평균, 합계, 표준편차 등.
    - **범주형 데이터**:
        - **빈도 인코딩 (Frequency Encoding)**: 범주의 등장 빈도로 인코딩.
        - **타겟 인코딩 (Target Encoding)**: 범주별 타겟 변수의 평균값 등으로 인코딩 (데이터 누수 주의).

### 나. 특징 변환 (Feature Transformation)
- 기존 특징의 형태나 분포를 변경하여 모델이 더 잘 학습할 수 있도록 합니다.
- **예시**:
    - **로그 변환 (Log Transform)**: 왜도(Skewness)가 큰 데이터의 분포를 정규분포에 가깝게 만듦. 분산을 안정화.
    - **제곱근 변환 (Square Root Transform)**.
    - **박스-칵스 변환 (Box-Cox Transform)**: 정규성을 만족하도록 하는 변환. (데이터가 양수여야 함)
    - **수치형 데이터를 범주형으로 변환 (Binning/Discretization)**: 연속형 변수를 특정 구간(Bin)으로 나누어 범주형으로 만듦. (예: 나이를 청년, 중년, 노년으로)

### 다. 특징 선택 (Feature Selection) - (필요시)
- 모델 성능에 중요하지 않거나 중복되는 특징을 제거하여 모델을 단순화하고 과적합을 방지하며 계산 효율성을 높입니다. (Day 71, 72의 PCA, LDA도 일종의 특징 선택/추출)
- **방법**:
    - **필터 방법 (Filter Methods)**: 통계적 측정값(상관계수, 카이제곱 검정, 분산 분석 등)을 사용하여 각 특징과 타겟 변수 간의 관련성을 평가하고 순위를 매겨 선택. 모델과 독립적으로 수행.
    - **래퍼 방법 (Wrapper Methods)**: 특정 모델의 성능을 직접 평가 기준으로 삼아, 다양한 특징 부분집합을 시도하며 최적의 조합을 찾음. (예: 전진 선택, 후진 제거, 재귀적 특징 제거 - `RFE`) 계산 비용이 높을 수 있음.
    - **임베디드 방법 (Embedded Methods)**: 모델 학습 과정 자체에 특징 선택이 포함된 방식. (예: L1 규제(Lasso)를 사용한 선형 모델, 트리 기반 모델의 특성 중요도)

### 특징 공학 예시 (Python 코드)

```python
# 특징 공학 예시 (어제 코드 이어서)
# df에 'processed_text', 'sentiment', 'rating', 'text_length'가 있다고 가정

# 1. 텍스트 길이 제곱 특징 추가 (다항 특성 예시)
if 'text_length' in df.columns:
    df['text_length_sq'] = df['text_length'] ** 2

# 2. 긍정/부정 단어 포함 수 (간단한 텍스트 특징 예시)
positive_keywords = ['great', 'amazing', 'wonderful', 'love', 'excellent']
negative_keywords = ['terrible', 'hated', 'bad', 'awful', 'poor']

def count_keywords(text, keywords):
    count = 0
    if pd.notnull(text): # 결측이 아닌 경우에만
        for keyword in keywords:
            if keyword in text.lower(): # 소문자로 변환 후 키워드 포함 여부 확인
                count += 1
    return count

if 'processed_text' in df.columns:
    df['positive_keyword_count'] = df['processed_text'].apply(lambda x: count_keywords(x, positive_keywords))
    df['negative_keyword_count'] = df['processed_text'].apply(lambda x: count_keywords(x, negative_keywords))

# 3. (만약 날짜 데이터가 있다면) 요일, 월 등 파생 변수 생성
# df['date_column'] = pd.to_datetime(df['date_column'])
# df['day_of_week'] = df['date_column'].dt.dayofweek
# df['month'] = df['date_column'].dt.month

# 4. 로그 변환 (예시: text_length가 매우 왜곡된 분포를 가질 경우)
# if 'text_length' in df.columns and df['text_length'].min() > 0: # 로그 변환은 양수 값에만 적용
#     df['text_length_log'] = np.log(df['text_length'])
# else:
#     df['text_length_log'] = np.log(df['text_length'] + 1) # 0을 피하기 위해 +1 (log1p와 유사)


print("\n특징 공학 후 데이터 샘플:")
print(df.head())

# 최종적으로 모델 학습에 사용할 특징 선택
# 예: features_for_model = df[['text_length', 'text_length_sq', 'positive_keyword_count', 'negative_keyword_count', 'rating']]
# 텍스트 자체를 사용하는 경우: processed_text 컬럼을 TF-IDF 등으로 변환
```

## 4. 최종 데이터셋 준비
- EDA와 특징 공학을 거쳐 최종적으로 모델 학습에 사용할 특징들을 선택하고, 이들을 포함하는 데이터셋을 구성합니다.
- 이 데이터셋은 이후 모델 학습, 평가, 하이퍼파라미터 튜닝 단계에서 사용됩니다.
- **주의**: 특징 공학 과정에서도 데이터 누수(Data Leakage)가 발생하지 않도록 주의해야 합니다. 예를 들어, 타겟 변수의 정보를 사용하여 새로운 특징을 만들 때, 이 정보가 검증 세트나 테스트 세트에서 온 것이라면 안 됩니다. 일반적으로 학습 데이터에 대해서만 `fit`하고, 전체 데이터(학습, 검증, 테스트)에 `transform`을 적용하는 파이프라인을 구성합니다.

## 실습 아이디어
- 본인의 캡스톤 프로젝트 데이터에 대해 오늘 배운 EDA 심화 기법들을 적용해보세요.
    - 다양한 시각화를 통해 데이터의 숨겨진 패턴이나 관계를 찾아보세요.
    - 프로젝트 목표와 관련된 구체적인 질문을 설정하고, EDA를 통해 답을 찾아보세요.
- EDA 결과를 바탕으로, 모델 성능에 도움이 될 만한 새로운 특징들을 2~3개 이상 생성해보세요.
    - 텍스트 데이터라면 길이, 특정 단어 포함 여부, N-gram 등을 고려해보세요.
    - 수치형 데이터라면 다항 특성, 비율, 그룹 통계 등을 고려해보세요.
- 생성한 특징들이 실제로 유용한지 간단한 모델(예: 로지스틱 회귀)로 테스트해보거나, 다음 모델링 단계에서 평가할 수 있도록 준비해두세요.

## 다음 학습 내용
- Day 90: 프로젝트를 위한 모델 선정 및 초기 학습 (Model Selection and initial training for the project) - 준비된 데이터를 사용하여 다양한 머신러닝/딥러닝 모델을 선택하고 초기 학습 및 평가를 진행.
