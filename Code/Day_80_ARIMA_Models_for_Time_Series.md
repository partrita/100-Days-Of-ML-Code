# Day 80: 시계열을 위한 ARIMA 모델 (ARIMA models for Time Series)

## 학습 목표
- ARIMA (AutoRegressive Integrated Moving Average) 모델의 개념과 구성 요소 이해
    - AR (AutoRegressive, 자기회귀) 모델
    - MA (Moving Average, 이동평균) 모델
    - I (Integrated, 통합/차분)
- ACF (Autocorrelation Function)와 PACF (Partial Autocorrelation Function)를 사용한 ARIMA 모델 식별 (차수 결정) 방법 학습
- ARIMA 모델 구축 및 예측 과정 이해
- `statsmodels` 라이브러리를 사용한 ARIMA 모델 구현 방법 숙지

## 1. ARIMA 모델 소개
- **정의**: 과거 자신의 관측값(AR)과 과거 예측 오차(MA)를 사용하여 현재 시점의 값을 설명하고 예측하는, 가장 널리 사용되는 전통적인 시계열 예측 모델 중 하나입니다.
- **ARIMA(p, d, q)** 형태로 표현됩니다:
    - **p**: AR 모델의 차수 (자기회귀 항의 수, 과거 몇 개의 관측값을 사용할지).
    - **d**: 차분(Differencing) 횟수 (시계열을 정상성(Stationary)으로 만들기 위해 필요한 차분 횟수).
    - **q**: MA 모델의 차수 (이동평균 항의 수, 과거 몇 개의 예측 오차를 사용할지).
- ARIMA 모델은 **정상 시계열(Stationary Time Series)**을 가정합니다. 만약 시계열이 비정상이면, 차분(d)을 통해 정상 시계열로 변환한 후 ARMA(p,q) 모델을 적용합니다.

## 2. ARIMA 모델의 구성 요소

### 가. AR (AutoRegressive, 자기회귀) 모델 - AR(p)
- **개념**: 현재 시점의 값(Y<sub>t</sub>)이 과거 p개의 자신의 관측값(Y<sub>t-1</sub>, ..., Y<sub>t-p</sub>)들의 선형 결합으로 설명된다고 가정하는 모델.
- **수식**: Y<sub>t</sub> = c + φ<sub>1</sub>Y<sub>t-1</sub> + φ<sub>2</sub>Y<sub>t-2</sub> + ... + φ<sub>p</sub>Y<sub>t-p</sub> + ε<sub>t</sub>
    - Y<sub>t</sub>: 현재 시점의 값
    - c: 상수항
    - φ<sub>i</sub>: i번째 자기회귀 계수 (모델 파라미터)
    - Y<sub>t-i</sub>: i 시점 전의 과거 값
    - ε<sub>t</sub>: 현재 시점의 백색 잡음(White Noise) 오차항 (평균 0, 일정한 분산)
- **특징**: 과거의 값이 현재 값에 직접적인 영향을 미치는 시계열 패턴을 모델링합니다.

### 나. MA (Moving Average, 이동평균) 모델 - MA(q)
- **개념**: 현재 시점의 값(Y<sub>t</sub>)이 과거 q개의 예측 오차(ε<sub>t-1</sub>, ..., ε<sub>t-q</sub>)들의 선형 결합과 현재 시점의 예측 오차(ε<sub>t</sub>)로 설명된다고 가정하는 모델.
    - **주의**: 여기서 이동평균은 시계열 데이터의 단순 이동평균(Smoothing 기법)과는 다른 개념입니다. 예측 오차의 이동평균을 의미합니다.
- **수식**: Y<sub>t</sub> = μ + ε<sub>t</sub> + θ<sub>1</sub>ε<sub>t-1</sub> + θ<sub>2</sub>ε<sub>t-2</sub> + ... + θ<sub>q</sub>ε<sub>t-q</sub>
    - μ: 시계열의 평균 (또는 상수항)
    - ε<sub>t</sub>: 현재 시점의 백색 잡음 오차항
    - θ<sub>i</sub>: i번째 이동평균 계수 (모델 파라미터)
    - ε<sub>t-i</sub>: i 시점 전의 예측 오차
- **특징**: 과거의 예측 불가능했던 충격(Shock)이나 오차가 현재 값에 영향을 미치는 시계열 패턴을 모델링합니다.

### 다. I (Integrated, 통합/차분) - d
- **개념**: 비정상 시계열을 정상 시계열로 변환하기 위해 필요한 차분(Differencing) 횟수를 의미합니다.
- **d=0**: 시계열이 이미 정상성을 만족하는 경우 (ARMA(p,q) 모델).
- **d=1**: 1차 차분을 통해 정상성을 만족하는 경우 (Y'<sub>t</sub> = Y<sub>t</sub> - Y<sub>t-1</sub>).
- **d=2**: 2차 차분을 통해 정상성을 만족하는 경우. (보통 0, 1, 2 정도의 값을 가짐)

### ARMA(p,q) 모델
- AR(p) 모델과 MA(q) 모델을 결합한 형태로, 정상 시계열에 적용됩니다.
- **수식**: Y<sub>t</sub> = c + φ<sub>1</sub>Y<sub>t-1</sub> + ... + φ<sub>p</sub>Y<sub>t-p</sub> + ε<sub>t</sub> + θ<sub>1</sub>ε<sub>t-1</sub> + ... + θ<sub>q</sub>ε<sub>t-q</sub>

## 3. ARIMA 모델 차수 결정 (p, d, q 식별)
- 적절한 p, d, q 값을 결정하는 것은 ARIMA 모델링에서 매우 중요한 단계입니다.

### 가. 차분 횟수 (d) 결정
1.  **시계열 시각화**: 원본 시계열 그림을 보고 추세나 계절성이 있는지 확인합니다.
2.  **단위근 검정 (Unit Root Test)**: ADF(Augmented Dickey-Fuller) Test, KPSS Test 등을 수행하여 정상성을 통계적으로 검정합니다.
    - ADF 검정: 귀무가설 "단위근이 존재한다 (비정상)" vs 대립가설 "단위근이 없다 (정상)". p-value < 0.05 이면 정상으로 판단.
3.  비정상으로 판단되면 1차 차분을 수행하고, 다시 정상성 검정을 합니다. 여전히 비정상이면 2차 차분을 시도합니다.
4.  일반적으로 d는 0, 1, 또는 2의 값을 가집니다. 과도한 차분은 불필요한 복잡성을 야기할 수 있습니다.

### 나. AR 차수 (p) 및 MA 차수 (q) 결정
- 차분된 (정상화된) 시계열에 대해 **ACF (Autocorrelation Function, 자기상관 함수)**와 **PACF (Partial Autocorrelation Function, 편자기상관 함수)** 플롯을 사용하여 p와 q를 추정합니다.

1.  **ACF (자기상관 함수)**:
    - 시차(Lag) k에 따른 현재 시점과 k 시점 전의 값 사이의 상관관계를 나타냅니다.
    - **MA(q) 모델의 특징**: ACF는 시차 q 이후에 급격히 감소하여 0에 가까워집니다 ( 절단점, Cut-off at lag q).
    - **AR(p) 모델의 특징**: ACF는 지수적으로 또는 사인파 형태로 천천히 감소합니다.

2.  **PACF (편자기상관 함수)**:
    - 시차 k에 따른 현재 시점과 k 시점 전의 값 사이의 "순수한" 상관관계를 나타냅니다. (즉, 그 사이의 시차들의 영향을 제거한 후의 상관관계)
    - **AR(p) 모델의 특징**: PACF는 시차 p 이후에 급격히 감소하여 0에 가까워집니다 (절단점, Cut-off at lag p).
    - **MA(q) 모델의 특징**: PACF는 지수적으로 또는 사인파 형태로 천천히 감소합니다.

#### ACF와 PACF를 이용한 (p, q) 선택 가이드라인 (Box-Jenkins 방법론)

| 모델 유형    | ACF 패턴                               | PACF 패턴                              |
| :----------- | :------------------------------------- | :------------------------------------- |
| **AR(p)**    | 점진적 감소 (Tails off)                | 시차 p 이후 절단 (Cuts off after lag p)  |
| **MA(q)**    | 시차 q 이후 절단 (Cuts off after lag q)  | 점진적 감소 (Tails off)                |
| **ARMA(p,q)** | 점진적 감소 (Tails off after lag q-p) | 점진적 감소 (Tails off after lag p-q) |
| (일반적)     | 점진적 감소                            | 점진적 감소                            |

- 실제로는 ACF/PACF 플롯이 명확하게 나타나지 않는 경우가 많아, 여러 (p,q) 조합을 시도하고 AIC, BIC와 같은 정보 기준(Information Criteria)을 사용하여 최적의 모델을 선택하기도 합니다.
    - **AIC (Akaike Information Criterion)**
    - **BIC (Bayesian Information Criterion)**
    - 이 값들이 낮을수록 더 좋은 모델로 간주합니다. (모델의 적합도와 복잡도 간의 균형 고려)

## 4. ARIMA 모델 구축 및 예측 과정
1.  **데이터 준비 및 시각화**: 시계열 데이터를 로드하고 시간의 흐름에 따른 패턴을 확인합니다.
2.  **정상성 확인 및 차분 (d 결정)**: 단위근 검정 등을 통해 정상성을 확인하고, 필요하면 차분을 수행합니다.
3.  **ACF/PACF 분석 (p, q 결정)**: 차분된 시계열의 ACF와 PACF 플롯을 그려 AR 및 MA 차수를 잠정적으로 결정합니다.
4.  **모델 학습**: 결정된 (p, d, q) 값으로 ARIMA 모델을 학습시킵니다. (모델 파라미터 추정)
5.  **모델 진단**: 학습된 모델의 잔차(Residuals)가 백색 잡음의 특성을 보이는지 확인합니다. (잔차의 ACF/PACF 플롯, 정규성 검정 등)
6.  **(선택) 모델 비교 및 선택**: 여러 (p,d,q) 조합에 대해 AIC, BIC 등을 비교하여 최적 모델을 선택합니다.
7.  **예측**: 최종 선택된 모델을 사용하여 미래 값을 예측합니다.
8.  **예측 성능 평가**: 실제 값과 예측 값을 비교하여 MAE, MSE, RMSE, MAPE 등으로 예측 성능을 평가합니다.

## 5. `statsmodels` 라이브러리를 사용한 ARIMA 모델 구현

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA # ARIMA 모델
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # ACF, PACF 플롯
from statsmodels.tsa.stattools import adfuller # ADF 검정
# import pmdarima as pm # auto_arima 사용 시 (선택 사항)

# 예제 시계열 데이터 생성 (간단한 ARMA(1,1) 과정 + 추세)
np.random.seed(42)
n_samples = 200
ar_params = np.array([0.5]) # AR(1) 계수
ma_params = np.array([0.3]) # MA(1) 계수
# ARMA 과정 생성 함수 (statsmodels에 있음)
from statsmodels.tsa.arima_process import ArmaProcess
arma_process = ArmaProcess(ar_params, ma_params)
data_stationary = arma_process.generate_sample(nsample=n_samples)
# 추세 추가 (비정상 시계열로 만들기)
trend = np.linspace(0, 10, n_samples)
data_nonstationary = data_stationary + trend
time_index = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
series = pd.Series(data_nonstationary, index=time_index)

# 1. 데이터 시각화
plt.figure(figsize=(10, 4))
plt.plot(series)
plt.title("Original Time Series")
plt.show()

# 2. 정상성 확인 및 차분 (d 결정)
# ADF 검정
result_adf = adfuller(series)
print(f'ADF Statistic: {result_adf[0]}')
print(f'p-value: {result_adf[1]}') # p-value가 크므로 비정상으로 판단 (귀무가설 기각 실패)

# 1차 차분
series_diff1 = series.diff().dropna() # diff()는 차분, dropna()는 결측치 제거
plt.figure(figsize=(10, 4))
plt.plot(series_diff1)
plt.title("1st Differenced Time Series")
plt.show()

result_adf_diff1 = adfuller(series_diff1)
print(f'\nADF Statistic (1st Diff): {result_adf_diff1[0]}')
print(f'p-value (1st Diff): {result_adf_diff1[1]}') # p-value가 작으므로 정상으로 판단 (d=1)
# d = 1 로 결정

# 3. ACF/PACF 분석 (p, q 결정) - 차분된 시계열 사용
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(series_diff1, ax=ax[0], lags=20) # lags는 최대 시차
plot_pacf(series_diff1, ax=ax[1], lags=20, method='ywm') # method='ols' or 'ywm' 등
plt.suptitle("ACF and PACF of 1st Differenced Series")
plt.show()
# ACF: lag 1 이후 절단? -> q=1 가능성
# PACF: lag 1 이후 절단? -> p=1 가능성
# 잠정적으로 ARIMA(1,1,1) 또는 ARIMA(1,1,0), ARIMA(0,1,1) 등을 고려

# 4. 모델 학습 (예: ARIMA(1,1,1))
# order=(p, d, q)
model = ARIMA(series, order=(1, 1, 1)) # 원본 시계열과 차분 횟수 d를 함께 전달
results = model.fit()
print(results.summary())

# 5. 모델 진단 (잔차 분석)
residuals = results.resid
fig, ax = plt.subplots(1, 2, figsize=(12, 3))
residuals.plot(title="Residuals", ax=ax[0])
plot_acf(residuals, ax=ax[1], lags=20)
plt.suptitle("Residuals Analysis")
plt.show() # 잔차가 백색 잡음인지 확인 (ACF가 거의 0에 가까워야 함)

# 6. (선택) auto_arima 사용 (pmdarima 라이브러리 필요: pip install pmdarima)
# from pmdarima.arima import auto_arima
# auto_model = auto_arima(series,
#                         start_p=0, start_q=0,
#                         max_p=3, max_q=3,
#                         d=1, # 또는 D=None으로 두어 자동 결정
#                         seasonal=False, # 비계절성 ARIMA
#                         trace=True, # 모델 탐색 과정 출력
#                         error_action='ignore',
#                         suppress_warnings=True,
#                         stepwise=True) # 단계적 탐색
# print("\nAuto ARIMA Best Model Summary:")
# print(auto_model.summary())
# best_order = auto_model.order # (p,d,q)

# 7. 예측
n_forecast = 30
forecast_results = results.get_forecast(steps=n_forecast)
forecast_mean = forecast_results.predicted_mean # 예측값
forecast_ci = forecast_results.conf_int() # 신뢰구간

plt.figure(figsize=(12, 5))
plt.plot(series, label='Observed')
plt.plot(forecast_mean, label='Forecast', color='red')
plt.fill_between(forecast_ci.index,
                 forecast_ci.iloc[:, 0], # 하한
                 forecast_ci.iloc[:, 1], # 상한
                 color='pink', alpha=0.3, label='Confidence Interval')
plt.title("Time Series Forecast with ARIMA(1,1,1)")
plt.legend()
plt.show()

# 8. 예측 성능 평가 (실제로는 테스트 데이터셋으로 평가해야 함)
# 예: 마지막 30개를 테스트 데이터로 사용했다고 가정하고 평가
# train_series = series[:-30]
# test_series = series[-30:]
# model_eval = ARIMA(train_series, order=(1,1,1))
# results_eval = model_eval.fit()
# forecast_eval_mean = results_eval.get_forecast(steps=30).predicted_mean
# from sklearn.metrics import mean_squared_error
# mse = mean_squared_error(test_series, forecast_eval_mean)
# print(f"\nMean Squared Error (Test): {mse:.4f}")
```

## 6. SARIMA (Seasonal ARIMA) 모델
- ARIMA 모델은 계절성(Seasonality)을 직접적으로 다루지 못합니다.
- 시계열 데이터에 계절성이 포함된 경우, **SARIMA(p, d, q)(P, D, Q)<sub>m</sub>** 모델을 사용합니다.
    - (p, d, q): 비계절성 부분의 차수 (ARIMA와 동일).
    - (P, D, Q): 계절성 부분의 차수.
    - m: 계절 주기의 길이 (예: 월별 데이터면 m=12, 분기별 데이터면 m=4).
- `statsmodels.tsa.statespace.SARIMAX` 클래스를 사용하여 구현할 수 있습니다.

## 추가 학습 자료
- [ARIMA Model for Time Series Forecasting (Machine Learning Mastery)](https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/)
- [A Gentle Introduction to SARIMA for Time Series Forecasting in Python (Machine Learning Mastery)](https://machinelearningmastery.com/sarima-for-time-series-forecasting-in-python/)
- [Statsmodels ARIMA Documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)
- [Forecasting: Principles and Practice - Chapter 9: ARIMA models (온라인 책)](https://otexts.com/fpp3/arima.html)

## 다음 학습 내용
- Day 81: 모델 배포 소개 (Introduction to Model Deployment) - 개발한 머신러닝 모델을 실제 환경에서 사용 가능하도록 만드는 과정.
