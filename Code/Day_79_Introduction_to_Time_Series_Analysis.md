# Day 79: 시계열 분석 소개 (Introduction to Time Series Analysis)

## 학습 목표
- 시계열 데이터(Time Series Data)의 정의와 특징 이해
- 시계열 분석의 주요 목적(예측, 패턴 파악 등) 학습
- 시계열 데이터의 주요 구성 요소 파악:
    - 추세 (Trend)
    - 계절성 (Seasonality)
    - 주기성 (Cyclicality)
    - 불규칙 요소 (Irregular/Random Component)
- 시계열 데이터 분석의 기본 단계 소개
- 정상성(Stationarity)의 개념과 중요성 학습

## 1. 시계열 데이터 (Time Series Data)란?
- **정의**: 시간의 흐름에 따라 일정한 간격으로 관찰되거나 기록된 데이터들의 수열(Sequence).
- **특징**:
    - **시간적 순서 의존성**: 데이터 포인트들이 시간 순서대로 정렬되어 있으며, 과거의 값이 미래의 값에 영향을 미칩니다. (자기상관성, Autocorrelation)
    - **일정한 시간 간격**: 보통 초, 분, 시간, 일, 주, 월, 분기, 연 단위 등 일정한 시간 간격으로 수집됩니다.
- **예시**:
    - 일별 주가 (Stock prices)
    - 월별 강수량 (Monthly rainfall)
    - 연도별 GDP 성장률 (Annual GDP growth rate)
    - 시간별 웹사이트 방문자 수 (Hourly website traffic)
    - 일별 코로나19 확진자 수
    - 센서에서 주기적으로 수집되는 데이터

## 2. 시계열 분석 (Time Series Analysis)의 목적
- **과거 패턴 이해**: 데이터에 내재된 추세, 계절성, 주기성 등의 패턴을 파악합니다.
- **미래 예측 (Forecasting)**: 과거의 패턴을 기반으로 미래의 값을 예측합니다. (가장 일반적인 목적)
- **이상 탐지 (Anomaly Detection)**: 일반적인 패턴에서 벗어나는 특이한 관측치를 찾아냅니다.
- **제어 (Control)**: 시계열 데이터를 모니터링하고 특정 목표를 달성하기 위해 시스템을 제어합니다. (예: 공정 관리)
- **가설 검정**: 시간의 흐름에 따른 특정 현상의 변화나 관계를 통계적으로 검증합니다.

## 3. 시계열 데이터의 주요 구성 요소
시계열 데이터는 일반적으로 다음과 같은 요소들의 결합으로 구성될 수 있다고 가정합니다. (분해, Decomposition)

### 가. 추세 (Trend, T<sub>t</sub>)
- 데이터가 장기적으로 증가하거나 감소하거나 또는 일정한 수준을 유지하는 경향.
- 예: 기술 발전으로 인한 생산성 증가, 인구 증가에 따른 수요 증가.

### 나. 계절성 (Seasonality, S<sub>t</sub>)
- 특정 기간(보통 1년 이내)을 주기로 반복적으로 나타나는 패턴.
- 달력상의 요인(월, 분기, 요일, 휴일 등)과 관련이 깊습니다.
- 예: 여름철 아이스크림 판매량 증가, 연말연시 쇼핑객 증가, 주말 교통량 변화.

### 다. 주기성 (Cyclicality / Cycle, C<sub>t</sub>)
- 계절성보다 긴 기간(보통 1년 이상)을 가지며, 고정된 주기가 아닌 불규칙적인 변동 패턴.
- 주로 경제 순환(경기 변동), 비즈니스 사이클 등과 관련됩니다.
- 계절성만큼 예측하기 어렵고, 주기가 일정하지 않을 수 있습니다.

### 라. 불규칙 요소 (Irregular / Random Component, ε<sub>t</sub> / I<sub>t</sub>)
- 추세, 계절성, 주기성으로 설명되지 않는 예측 불가능한 무작위 변동.
- 노이즈(Noise) 또는 잔차(Residual)라고도 불립니다.
- 이상치(Outlier)나 갑작스러운 사건의 영향을 포함할 수 있습니다.

### 시계열 분해 모델
- **덧셈 모델 (Additive Model)**: Y<sub>t</sub> = T<sub>t</sub> + S<sub>t</sub> + C<sub>t</sub> + ε<sub>t</sub>
    - 각 구성 요소의 크기가 시계열의 수준과 관계없이 일정하다고 가정.
- **곱셈 모델 (Multiplicative Model)**: Y<sub>t</sub> = T<sub>t</sub> * S<sub>t</sub> * C<sub>t</sub> * ε<sub>t</sub>
    - 각 구성 요소의 크기가 시계열의 수준에 비례하여 변한다고 가정. (예: 추세가 증가하면 계절적 변동폭도 커짐)
    - 로그 변환을 통해 덧셈 모델로 변환하여 분석하기도 합니다: log(Y<sub>t</sub>) = log(T<sub>t</sub>) + log(S<sub>t</sub>) + log(C<sub>t</sub>) + log(ε<sub>t</sub>)

![Time Series Components](https://otexts.com/fpp3/fpp3_files/figure-html/classical-decomp-1.png)
*(이미지 출처: Forecasting: Principles and Practice by Hyndman & Athanasopoulos)*

## 4. 시계열 데이터 분석의 기본 단계
1.  **데이터 수집 및 시각화**:
    - 시간의 흐름에 따른 데이터를 수집하고, 라인 플롯(Line Plot) 등을 통해 시각화하여 전반적인 패턴을 파악합니다.
2.  **데이터 전처리**:
    - 결측치 처리, 이상치 탐지 및 처리.
    - 필요시 로그 변환, 차분(Differencing) 등을 수행하여 데이터 안정화.
3.  **시계열 분해 (Decomposition)**:
    - 추세, 계절성, 불규칙 요소 등을 분리하여 각 구성 요소의 특징을 분석합니다. (예: 이동 평균법, STL 분해)
4.  **정상성 확인 (Stationarity Check)**:
    - 시계열 데이터가 정상성을 만족하는지 확인합니다. 많은 시계열 모델은 정상성을 가정합니다.
    - 비정상 시계열의 경우 차분 등을 통해 정상 시계열로 변환합니다.
5.  **모델 선택 및 학습**:
    - 데이터의 특징과 분석 목적에 맞는 시계열 모델을 선택합니다. (예: ARIMA, 지수평활법, Prophet, LSTM 등)
    - 학습 데이터를 사용하여 모델 파라미터를 추정합니다.
6.  **모델 진단 및 평가**:
    - 학습된 모델이 데이터를 잘 설명하는지, 잔차가 백색 잡음(White Noise)의 특성을 보이는지 등을 진단합니다.
    - 예측 성능 평가지표(MAE, MSE, RMSE, MAPE 등)를 사용하여 모델을 평가합니다.
7.  **예측 (Forecasting)**:
    - 최종 선택된 모델을 사용하여 미래 값을 예측합니다.

## 5. 정상성 (Stationarity)
- **정의**: 시계열의 통계적 특성(평균, 분산, 자기공분산 등)이 시간의 흐름에 따라 변하지 않고 일정하게 유지되는 성질.
- **중요성**: 많은 전통적인 시계열 모델(예: ARIMA)은 데이터가 정상성을 만족한다고 가정합니다. 비정상 시계열은 예측의 불확실성을 높이고 모델링을 어렵게 만듭니다.
- **종류**:
    - **강한 정상성 (Strict Stationarity)**: 시계열의 모든 분포가 시간에 따라 변하지 않음. (매우 엄격한 조건)
    - **약한 정상성 (Weak Stationarity / Covariance Stationarity)**:
        1.  평균이 시간에 관계없이 일정 (No Trend).
        2.  분산이 시간에 관계없이 일정 (Homoscedasticity).
        3.  두 시점 간의 공분산(자기공분산)은 시차(Lag)에만 의존하고, 특정 시점에는 의존하지 않음.
- **정상성 확인 방법**:
    - **시각적 확인**: 시계열 그림에서 추세나 계절성, 변동하는 분산 등이 관찰되는지 확인.
    - **자기상관 함수 (ACF, Autocorrelation Function) 플롯**: ACF가 천천히 감소하면 비정상성을 의심.
    - **단위근 검정 (Unit Root Test)**: 통계적 가설 검정 방법. (예: ADF 테스트 - Augmented Dickey-Fuller Test, KPSS 테스트)
        - ADF 검정: 귀무가설(H0)은 "단위근이 존재한다 (비정상 시계열이다)". p-value가 유의수준(예: 0.05)보다 작으면 귀무가설을 기각하고 정상 시계열로 판단.
- **비정상 시계열을 정상 시계열로 변환하는 방법**:
    - **차분 (Differencing)**: 현재 시점의 값에서 이전 시점의 값을 빼는 것. (d<sub>t</sub> = Y<sub>t</sub> - Y<sub>t-1</sub>). 추세를 제거하는 데 효과적.
    - **로그 변환 (Log Transformation)**: 분산이 시간에 따라 증가하는 경우 분산을 안정화시키는 데 도움.
    - **계절 차분 (Seasonal Differencing)**: 계절성을 제거하기 위해 특정 계절 주기만큼 떨어진 값을 빼는 것.

## 추가 학습 자료
- [Forecasting: Principles and Practice (3rd ed) by Rob J Hyndman and George Athanasopoulos (온라인 책)](https://otexts.com/fpp3/) - (시계열 분석 및 예측 분야의 교과서적인 자료)
- [Stationarity in time series analysis (Towards Data Science)](https://towardsdatascience.com/stationarity-in-time-series-analysis-90c94f27322)
- [Introduction to Time Series Analysis in Python (DataCamp)](https://www.datacamp.com/courses/introduction-to-time-series-analysis-in-python)

## 다음 학습 내용
- Day 80: 시계열을 위한 ARIMA 모델 (ARIMA models for Time Series) - 대표적인 전통적 시계열 예측 모델.
