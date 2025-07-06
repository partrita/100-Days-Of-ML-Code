# Day 88: 프로젝트를 위한 데이터 수집 및 전처리 (Data Collection and Preprocessing for the project)

## 학습 목표
- 선정된 캡스톤 프로젝트에 필요한 실제 데이터를 수집하는 방법 실행.
- 수집된 데이터의 초기 탐색(EDA 기초)을 통해 데이터의 특성 파악.
- 프로젝트 목표에 맞게 데이터를 정제하고, 결측치 및 이상치를 처리하며, 필요한 경우 특징 스케일링 등의 전처리 작업 수행.
- 머신러닝 모델 학습에 적합한 형태로 데이터를 가공.

## 1. 데이터 수집 (Data Collection)
- 프로젝트 주제와 범위에 따라 데이터 수집 방법이 달라집니다.

### 가. 공개 데이터셋 활용
- **Kaggle Datasets**: 다양한 주제의 방대한 데이터셋 제공. (예: 영화 리뷰, 고객 이탈, 주가 등)
    - 예시: IMDB Movie Reviews, Amazon Product Reviews, Twitter US Airline Sentiment.
- **UCI Machine Learning Repository**: 고전적이고 잘 알려진 머신러닝 데이터셋 다수 보유.
- **공공데이터포털 (data.go.kr)**: 국내 공공기관에서 제공하는 다양한 분야의 데이터.
- **기타 연구기관 또는 기업에서 공개하는 데이터셋**: (예: AI Hub, Dacon 등)
- **데이터셋 선택 시 고려사항**:
    - 프로젝트 주제와의 관련성.
    - 데이터의 크기 (너무 작거나 너무 크지 않은 적절한 규모).
    - 데이터의 품질 (결측치, 오류 등).
    - 데이터 사용 라이선스 확인.

### 나. 웹 크롤링 (Web Crawling/Scraping)
- 웹사이트에서 직접 필요한 정보를 수집하는 방법.
- **파이썬 라이브러리**: `Requests` (HTTP 요청), `Beautiful Soup` (HTML 파싱), `Selenium` (동적 웹사이트 크롤링).
- **주의사항**:
    - 웹사이트의 `robots.txt` 파일을 확인하여 크롤링 허용 여부 및 규칙 준수.
    - 과도한 요청으로 서버에 부담을 주지 않도록 주의 (시간 간격 설정).
    - 수집한 데이터의 저작권 및 개인정보보호 관련 법규 확인.
- **예시 (영화 리뷰 감성 분석 프로젝트)**: 네이버 영화, 다음 영화 등에서 리뷰 데이터 수집.

### 다. API 활용
- 많은 서비스(트위터, 유튜브, 공공 API 등)에서 데이터를 제공하는 API를 운영.
- API 문서를 참고하여 요청 방식, 인증 방법 등을 숙지하고 데이터 수집.
- **파이썬 라이브러리**: `Requests`, 각 서비스별 API 클라이언트 라이브러리.

### 데이터 수집 후 초기 작업
- 수집된 데이터는 보통 CSV, JSON, TXT, 데이터베이스 등의 형태로 저장합니다.
- **데이터 로드**: Pandas DataFrame 등으로 데이터를 불러옵니다.
- **기본 정보 확인**: `df.head()`, `df.info()`, `df.describe()`, `df.shape` 등을 통해 데이터의 구조, 타입, 기초 통계량, 크기 등을 파악합니다.

## 2. 데이터 탐색 및 시각화 (Exploratory Data Analysis, EDA) - 기초
- 데이터의 특성을 이해하고, 패턴을 발견하며, 문제점을 식별하는 과정.
- **주요 활동**:
    - **데이터 분포 확인**: 각 특성(Feature)의 분포를 시각화 (히스토그램, 박스 플롯 등).
    - **결측치 확인**: `df.isnull().sum()` 등으로 결측치의 수와 위치 파악.
    - **이상치 확인**: 박스 플롯이나 산점도 등으로 극단적인 값 확인.
    - **타겟 변수 분포 확인**: 분류 문제의 경우 클래스별 데이터 불균형 확인.
    - **특성 간 관계 분석**: 산점도 행렬(Scatter Matrix), 상관관계 히트맵(Correlation Heatmap) 등으로 특성 간의 관계 파악.
    - **텍스트 데이터의 경우**: 단어 빈도 분석, 워드 클라우드, 문장 길이 분포 등.

```python
# EDA 예시 (Pandas, Matplotlib, Seaborn 사용)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CSV 파일 로드 가정
# df = pd.read_csv('your_data.csv')

# --- 예시 데이터 생성 (실제로는 수집한 데이터 사용) ---
data_dict = {
    'text': [
        "This movie is great and wonderful!",
        "Absolutely terrible film, I hated it.",
        "Not bad, but could have been better.",
        "An amazing masterpiece of cinema.",
        None, # 결측치 예시
        "Just okay, nothing special to write home about."
    ],
    'sentiment': [1, 0, 0, 1, 1, 0], # 1: 긍정, 0: 부정
    'rating': [5, 1, 3, 5, 4, 3] # 1-5점 척도
}
df = pd.DataFrame(data_dict)
# --- 예시 데이터 생성 끝 ---


print("데이터 샘플:")
print(df.head())

print("\n데이터 정보:")
df.info()

print("\n수치형 데이터 기술 통계:")
print(df.describe())

print("\n결측치 확인:")
print(df.isnull().sum())

# 타겟 변수 분포 (분류 문제의 경우)
if 'sentiment' in df.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(x='sentiment', data=df)
    plt.title('Sentiment Distribution')
    plt.show()

# 텍스트 길이 분포 (텍스트 데이터의 경우)
if 'text' in df.columns:
    df['text_length'] = df['text'].astype(str).apply(len) # 결측치 때문에 str로 변환 후 길이 계산
    plt.figure(figsize=(8, 5))
    sns.histplot(df['text_length'], kde=True)
    plt.title('Distribution of Text Length')
    plt.show()

# (선택) 워드 클라우드 (텍스트 데이터)
# from wordcloud import WordCloud
# if 'text' in df.columns:
#     all_text = " ".join(review for review in df['text'].astype(str) if review) # 결측치 제외
#     if all_text:
#         wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
#         plt.figure(figsize=(10, 5))
#         plt.imshow(wordcloud, interpolation='bilinear')
#         plt.axis("off")
#         plt.title("Word Cloud of Reviews")
#         plt.show()
```

## 3. 데이터 전처리 (Data Preprocessing)
- EDA 결과를 바탕으로 모델 학습에 적합하도록 데이터를 가공하고 정제하는 과정.

### 가. 결측치 처리 (Handling Missing Values)
- **결측치 확인**: `df.isnull().sum()`
- **처리 방법**:
    - **삭제 (Deletion)**:
        - 결측치가 포함된 행(Row) 또는 열(Column)을 삭제.
        - 결측치가 매우 적거나, 해당 데이터가 중요하지 않을 때 사용.
        - `df.dropna(axis=0)` (행 삭제), `df.dropna(axis=1)` (열 삭제).
    - **대치 (Imputation)**:
        - 결측치를 특정 값으로 채움.
        - **수치형 데이터**: 평균(Mean), 중앙값(Median), 최빈값(Mode) 등으로 대치. 또는 예측 모델을 사용하여 대치.
          `df['column'].fillna(df['column'].mean(), inplace=True)`
        - **범주형 데이터**: 최빈값으로 대치. 또는 "Unknown"과 같은 새로운 카테고리 생성.
          `df['column'].fillna(df['column'].mode()[0], inplace=True)`
        - **텍스트 데이터**: 빈 문자열("") 또는 특정 토큰("<unk>")으로 대치.
          `df['text_column'].fillna("", inplace=True)`
        - `scikit-learn`의 `SimpleImputer` 사용 가능.

### 나. 이상치 처리 (Handling Outliers) - (필요시)
- 데이터의 일반적인 분포에서 크게 벗어난 값.
- **탐지 방법**: 박스 플롯, Z-score, IQR(Interquartile Range) 등.
- **처리 방법**:
    - **삭제**: 이상치가 오류로 판단되거나 모델에 큰 악영향을 줄 경우.
    - **변환 (Transformation)**: 로그 변환 등으로 데이터 분포를 조정.
    - **값 수정 (Capping/Flooring)**: 특정 임계값(예: IQR 기준 상한/하한)으로 값을 제한.
    - 그대로 사용: 이상치가 실제 현상을 반영하는 중요한 정보일 수 있음.

### 다. 텍스트 데이터 전처리 (NLP 프로젝트의 경우)
- Day 61에서 다룬 내용 복습 및 적용:
    - **정제**: HTML 태그, 특수 문자, 숫자 제거 또는 대체.
    - **정규화**: 소문자화, 축약형 처리.
    - **토큰화**: 단어 또는 형태소 단위로 분리 (영어: NLTK, spaCy / 한국어: KoNLPy).
    - **불용어 제거**: 의미 없는 단어 제거.
    - **어간 추출 (Stemming) 또는 표제어 추출 (Lemmatization)**: 단어의 기본형으로 통일.

```python
# 텍스트 전처리 예시 (간단화)
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# nltk.download('punkt') # 최초 실행 시
# nltk.download('stopwords') # 최초 실행 시

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if pd.isnull(text): # 결측치 처리
        return ""
    text = text.lower() # 소문자화
    text = re.sub(r'<[^>]+>', '', text) # HTML 태그 제거
    text = re.sub(r'[^a-z\s]', '', text) # 알파벳과 공백 외 문자 제거
    tokens = word_tokenize(text) # 토큰화
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1] # 불용어 및 한 글자 단어 제거
    # (선택) 표제어 추출 등 추가 가능
    return " ".join(tokens)

if 'text' in df.columns:
    df['processed_text'] = df['text'].apply(preprocess_text)
    print("\n전처리된 텍스트 샘플:")
    print(df[['text', 'processed_text']].head())
```

### 라. 범주형 데이터 인코딩 (Categorical Data Encoding) - (해당되는 경우)
- 머신러닝 모델은 숫자 입력을 가정하므로, 범주형 특성(예: 성별, 지역명)을 숫자로 변환해야 합니다.
- **레이블 인코딩 (Label Encoding)**: 각 범주를 정수로 매핑 (예: 'Red':0, 'Green':1, 'Blue':2). 순서가 없는 명목형 변수에 부적절할 수 있음 (모델이 순서 관계로 오해 가능).
    - `sklearn.preprocessing.LabelEncoder`
- **원-핫 인코딩 (One-Hot Encoding)**: 각 범주를 새로운 이진 특성(0 또는 1)으로 변환. 범주 수만큼 차원이 증가.
    - `sklearn.preprocessing.OneHotEncoder`, `pandas.get_dummies()`

### 마. 특징 스케일링 (Feature Scaling) - (수치형 데이터, 거리 기반 모델 등에 중요)
- 서로 다른 범위의 값을 가진 특성들을 일정한 범위로 조정합니다.
- **목적**: 특정 특성이 값의 크기 때문에 모델 학습에 과도한 영향을 미치는 것을 방지. 거리 기반 알고리즘(KNN, SVM, K-Means 등)이나 경사 하강법 기반 알고리즘(선형 회귀, 로지스틱 회귀, 신경망 등)에서 중요.
- **종류**:
    - **표준화 (Standardization)**: 특성의 평균을 0, 표준편차를 1로 변환. (Z-score 정규화)
      `sklearn.preprocessing.StandardScaler`
    - **정규화 (Normalization / Min-Max Scaling)**: 특성의 값을 0과 1 사이의 범위로 변환.
      `sklearn.preprocessing.MinMaxScaler`

### 바. 데이터 분할 (Train-Test Split)
- 전처리된 데이터를 모델 학습용(Train Set)과 평가용(Test Set)으로 분리합니다.
- `sklearn.model_selection.train_test_split`
- **주의**: 데이터 분할은 모든 전처리(특히 스케일링, 인코딩 등)가 완료된 후, 또는 **학습 데이터에 대해서만 `fit`하고 테스트 데이터에는 `transform`만 적용하는 방식으로 수행**되어야 데이터 누수(Data Leakage)를 방지할 수 있습니다. (Pipeline 사용 권장)

## 4. 다음 단계: 특징 공학 및 모델링
- 오늘 전처리된 데이터를 바탕으로, 내일은 모델 성능을 더욱 향상시키기 위한 특징 공학(Feature Engineering) 단계를 진행하고, 본격적인 모델 학습 및 평가를 시작합니다.
- **특징 공학 예시**:
    - 텍스트 데이터: TF-IDF, Word Embeddings (Word2Vec, FastText, GloVe) 생성.
    - 수치형 데이터: 다항 특성 생성, 상호작용 특성 생성.
    - 날짜/시간 데이터: 연, 월, 일, 요일, 시간 등 파생 변수 생성.

## 실습 아이디어
- 본인이 선정한 캡스톤 프로젝트에 필요한 데이터를 수집하는 과정을 시작해보세요.
    - 공개 데이터셋이라면 다운로드하고 로드합니다.
    - 웹 크롤링이 필요하다면, 간단한 스크립트를 작성하여 소량의 데이터를 수집해봅니다. (대상 웹사이트의 정책 확인 필수)
- 수집한 데이터를 Pandas DataFrame으로 로드하고, 오늘 배운 EDA 기초 기법을 적용하여 데이터의 특성을 파악해보세요.
- 결측치, 이상치 등을 확인하고, 적절한 전처리 계획을 세워보세요. (실제 적용은 내일 특징 공학과 함께 진행해도 좋습니다.)
- 텍스트 데이터가 주된 프로젝트라면, 기본적인 텍스트 정제 및 토큰화 함수를 만들어 적용해보세요.

## 다음 학습 내용
- Day 89: 프로젝트를 위한 탐색적 데이터 분석 (EDA) 심화 및 특징 공학 (Exploratory Data Analysis (EDA) for the project and Feature Engineering) - 수집/전처리된 데이터에 대한 심층적인 분석과 모델 성능 향상을 위한 특징 생성.
