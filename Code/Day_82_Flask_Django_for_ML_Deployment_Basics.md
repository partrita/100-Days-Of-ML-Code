# Day 82: ML 모델 배포를 위한 Flask/Django - 기초 (Flask/Django for deploying ML models - Basics)

## 학습 목표
- 파이썬 웹 프레임워크 Flask와 Django의 기본 개념 이해
- 머신러닝 모델을 API 형태로 배포하는 데 웹 프레임워크가 왜 필요한지 학습
- Flask를 사용하여 간단한 머신러닝 모델 예측 API를 만드는 기본 과정 숙지
- Django를 사용한 모델 배포의 간략한 개념 소개 및 Flask와의 비교

## 1. 웹 프레임워크(Web Framework)란?
- 웹 애플리케이션, 웹 서비스, 웹 API 등을 개발하는 데 필요한 기본적인 구조와 도구들을 제공하는 소프트웨어 프레임워크입니다.
- 반복적인 작업(요청 처리, 라우팅, 데이터베이스 연동, 템플릿 렌더링 등)을 단순화하고 개발 생산성을 높여줍니다.

## 2. 왜 ML 모델 배포에 웹 프레임워크를 사용하는가?
- 학습된 머신러닝 모델을 다른 애플리케이션이나 사용자가 쉽게 접근하여 사용할 수 있도록 하려면, 모델을 **웹 서비스 또는 API 형태로 노출**하는 것이 일반적입니다.
- 웹 프레임워크는 다음과 같은 기능을 제공하여 ML 모델 API 개발을 용이하게 합니다:
    - **HTTP 요청 처리**: 클라이언트(사용자, 다른 시스템)로부터 HTTP 요청(GET, POST 등)을 받아 처리합니다.
    - **라우팅 (Routing)**: 특정 URL 경로(엔드포인트)로 들어오는 요청을 적절한 처리 함수(핸들러)로 연결합니다.
    - **데이터 직렬화/역직렬화**: 요청으로 들어온 데이터(JSON, 폼 데이터 등)를 파이썬 객체로 변환하고, 모델 예측 결과를 다시 클라이언트가 이해할 수 있는 형태(JSON 등)로 변환합니다.
    - **요청/응답 관리**: HTTP 헤더, 상태 코드 등을 관리합니다.
    - **확장성**: 필요에 따라 기능을 추가하거나 다른 서비스와 연동하기 용이합니다.

## 3. Flask 소개
- **정의**: 파이썬으로 작성된 마이크로(Micro) 웹 프레임워크입니다. "마이크로"라는 의미는 핵심 기능을 작고 가볍게 유지하며, 필요한 기능은 확장(Extension)을 통해 추가할 수 있다는 뜻입니다.
- **특징**:
    - **가볍고 단순함**: 배우기 쉽고 빠르게 웹 애플리케이션을 개발할 수 있습니다.
    - **유연성**: 개발자가 원하는 방식으로 구조를 설계하고 필요한 확장을 선택하여 사용할 수 있습니다.
    - **최소한의 기능 제공**: URL 라우팅, 요청/응답 처리, 템플릿 엔진(Jinja2) 연동 등 핵심 기능에 집중합니다. 데이터베이스 ORM, 폼 처리 등은 기본으로 제공하지 않고 확장으로 지원.
    - **작은 프로젝트나 프로토타이핑, 간단한 API 개발에 적합**.

### Flask 설치
```bash
pip install Flask
```

### 간단한 Flask 애플리케이션 예제
```python
# app.py
from flask import Flask, request, jsonify

app = Flask(__name__) # Flask 애플리케이션 객체 생성

# 루트 URL ('/')로 GET 요청이 오면 실행될 함수
@app.route('/')
def home():
    return "Hello, Flask!"

# '/greet' URL로 GET 요청이 오고, 'name' 파라미터가 있으면 실행
@app.route('/greet', methods=['GET'])
def greet():
    name = request.args.get('name', 'Guest') # URL 파라미터에서 'name' 값 가져오기
    return f"Hello, {name}!"

# '/predict' URL로 POST 요청이 오면 실행 (JSON 데이터 예상)
@app.route('/predict', methods=['POST'])
def predict_example():
    if request.is_json:
        data = request.get_json() # POST 요청의 JSON 데이터 가져오기
        # 여기서 data를 사용하여 모델 예측 수행 (예시)
        number = data.get('number', 0)
        prediction = number * 2
        return jsonify({'input_number': number, 'prediction': prediction}) # JSON 형태로 응답
    else:
        return jsonify({'error': 'Request must be JSON'}), 400 # 400 Bad Request

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) # 개발 서버 실행
    # debug=True: 코드 변경 시 자동 재시작, 디버깅 정보 제공 (운영 환경에서는 False)
    # host='0.0.0.0': 모든 네트워크 인터페이스에서 접속 허용
    # port=5000: 사용할 포트 번호
```
- 위 코드를 `app.py`로 저장하고 터미널에서 `python app.py` 실행.
- 웹 브라우저나 `curl` 등으로 테스트:
    - `http://localhost:5000/` -> "Hello, Flask!"
    - `http://localhost:5000/greet?name=World` -> "Hello, World!"
    - `curl -X POST -H "Content-Type: application/json" -d '{"number": 10}' http://localhost:5000/predict` -> `{"input_number": 10, "prediction": 20}`

## 4. Flask를 사용한 간단한 ML 모델 예측 API 만들기

### 가. 모델 준비 (학습 및 직렬화)
- 먼저, 학습된 머신러닝 모델이 파일 형태로 저장되어 있어야 합니다. (예: `pickle`, `joblib` 사용)
- 여기서는 간단한 Scikit-learn 모델을 예시로 사용합니다.

```python
# train_model.py (모델 학습 및 저장 - 별도 파일로 실행)
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import joblib # 또는 pickle

# 1. 데이터 로드 및 모델 학습 (간단 예시)
iris = load_iris()
X, y = iris.data, iris.target
# 간단화를 위해 두 개의 클래스만 사용 (0, 1)
X_binary = X[y != 2]
y_binary = y[y != 2]

model = LogisticRegression(solver='liblinear')
model.fit(X_binary, y_binary)
print("모델 학습 완료!")

# 2. 모델 저장
joblib.dump(model, 'iris_binary_model.pkl')
print("모델 저장 완료: iris_binary_model.pkl")

# (선택) 스케일러 등 전처리기도 함께 저장해야 할 수 있음
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_binary)
# model.fit(X_scaled, y_binary)
# joblib.dump(scaler, 'iris_scaler.pkl')
# joblib.dump(model, 'iris_binary_model_scaled.pkl')
```
- 위 `train_model.py`를 먼저 실행하여 `iris_binary_model.pkl` 파일을 생성합니다.

### 나. Flask API 서버 구현
```python
# flask_api.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# 모델 로드 (애플리케이션 시작 시 한 번만 로드)
try:
    model = joblib.load('iris_binary_model.pkl')
    print("모델 로드 성공!")
    # scaler = joblib.load('iris_scaler.pkl') # 스케일러 사용 시 함께 로드
except FileNotFoundError:
    model = None
    # scaler = None
    print("저장된 모델 파일을 찾을 수 없습니다. 'train_model.py'를 먼저 실행하세요.")
except Exception as e:
    model = None
    # scaler = None
    print(f"모델 로드 중 오류 발생: {e}")


@app.route('/')
def home():
    return "Iris Binary Classification API"

@app.route('/predict_iris', methods=['POST'])
def predict_iris():
    if model is None:
        return jsonify({'error': '모델이 로드되지 않았습니다.'}), 500

    if request.is_json:
        try:
            data = request.get_json()
            # 입력 데이터 형식 가정: {"features": [sepal_length, sepal_width, petal_length, petal_width]}
            features = data.get('features')

            if features is None or not isinstance(features, list) or len(features) != 4:
                return jsonify({'error': '입력 데이터 형식이 올바르지 않습니다. "features" 키에 4개의 숫자 리스트를 제공해야 합니다.'}), 400

            # 입력 데이터를 NumPy 배열로 변환
            input_array = np.array(features).reshape(1, -1) # (1, 4) 형태

            # (선택) 스케일링 적용 (학습 시 사용한 스케일러와 동일하게)
            # if scaler:
            #     input_array_scaled = scaler.transform(input_array)
            #     prediction_proba = model.predict_proba(input_array_scaled)[0] # 각 클래스에 대한 확률
            #     prediction = model.predict(input_array_scaled)[0] # 예측 클래스
            # else:
            prediction_proba = model.predict_proba(input_array)[0]
            prediction = model.predict(input_array)[0]

            class_names = ['setosa', 'versicolor'] # 이진 분류 클래스 이름 (0, 1)

            return jsonify({
                'input_features': features,
                'predicted_class_id': int(prediction),
                'predicted_class_name': class_names[int(prediction)],
                'probabilities': {'setosa': float(prediction_proba[0]), 'versicolor': float(prediction_proba[1])}
            })

        except Exception as e:
            return jsonify({'error': f'예측 중 오류 발생: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Request must be JSON'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) # 다른 포트 사용
```

### 다. API 테스트 (curl 사용)
```bash
# flask_api.py 실행 후 터미널에서 아래 명령어 실행
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}' \
  http://localhost:5001/predict_iris

# 예상 응답:
# {
#   "input_features": [5.1, 3.5, 1.4, 0.2],
#   "predicted_class_id": 0,
#   "predicted_class_name": "setosa",
#   "probabilities": {
#     "setosa": 0.9...,
#     "versicolor": 0.0...
#   }
# }

curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"features": [6.0, 2.2, 4.0, 1.0]}' \
  http://localhost:5001/predict_iris
# 예상 응답 (versicolor에 가까운 값)
```

## 5. Django 소개 및 Flask와의 비교
- **정의**: 파이썬으로 작성된 고수준(High-level) 웹 프레임워크로, "배터리 포함(Batteries-included)" 철학을 따릅니다. 즉, 웹 개발에 필요한 대부분의 기능(ORM, 관리자 인터페이스, 폼 처리, 인증 등)을 기본적으로 제공합니다.
- **특징**:
    - **풀 스택 프레임워크**: 웹 개발의 모든 측면을 다룰 수 있는 포괄적인 기능 제공.
    - **ORM (Object-Relational Mapper)**: 데이터베이스 작업을 파이썬 객체로 쉽게 처리.
    - **강력한 관리자 인터페이스**: 모델 데이터를 관리할 수 있는 웹 기반 관리자 페이지 자동 생성.
    - **보안 기능**: CSRF 방지, XSS 방지 등 기본적인 보안 기능 내장.
    - **대규모 프로젝트, 복잡한 웹 애플리케이션 개발에 적합**.

### Django를 사용한 ML 모델 배포
- Django를 사용하여 ML 모델 API를 만드는 것도 가능하며, Flask와 유사한 방식으로 라우팅, 요청 처리, 모델 로드 및 예측 로직을 구현합니다.
- Django REST framework와 같은 확장을 사용하면 강력한 REST API를 더 쉽게 구축할 수 있습니다.
- 더 많은 기능을 기본 제공하므로 초기 설정이나 학습 곡선이 Flask보다 가파를 수 있지만, 복잡한 요구사항이나 대규모 서비스에는 더 적합할 수 있습니다.

### Flask vs Django (ML 모델 API 배포 관점)

| 특징           | Flask                                     | Django                                         |
|----------------|-------------------------------------------|------------------------------------------------|
| **철학**       | 마이크로 프레임워크, 최소 기능, 높은 유연성   | 풀 스택 프레임워크, "배터리 포함"                 |
| **학습 곡선**  | 낮음, 배우기 쉬움                           | 상대적으로 높음                                  |
| **기본 기능**  | 핵심 기능만 제공, 나머지는 확장으로          | ORM, 관리자, 폼, 인증 등 대부분 기능 내장        |
| **프로젝트 규모**| 작거나 중간 규모, 간단한 API, 프로토타입    | 중간 규모 이상, 복잡한 웹 애플리케이션, 대규모 서비스 |
| **ML API 개발**| 간단한 API 빠르게 구축 가능                | 더 많은 기능과 구조 제공, DRF 등 활용 시 강력    |
| **유연성**     | 매우 높음                                   | 상대적으로 낮으나, 필요한 대부분 기능 제공         |

- **선택 가이드**:
    - **빠르게 간단한 예측 API를 만들어야 할 때, 또는 다른 마이크로서비스와 통합할 때**: Flask가 좋은 선택일 수 있습니다.
    - **모델 API 외에 사용자 인터페이스, 데이터 관리, 인증 등 복합적인 웹 애플리케이션 기능이 함께 필요할 때**: Django가 더 적합할 수 있습니다.
    - 최근에는 FastAPI와 같은 ASGI(Asynchronous Server Gateway Interface) 기반의 현대적인 프레임워크도 ML API 개발에 많이 사용됩니다 (비동기 처리, 자동 API 문서 생성 등 장점).

## 추가 학습 자료
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Deploy Machine Learning Model using Flask (DataCamp Tutorial)](https://www.datacamp.com/tutorial/machine-learning-model-deployment-flask)
- [Django Documentation](https://www.djangoproject.com/)
- [Building a Machine Learning API with Django REST framework (Real Python)](https://realpython.com/django-rest-framework-class-based-views/)

## 다음 학습 내용
- Day 83: ML 모델을 위한 간단한 API 만들기 (Creating a simple API for an ML model) - 오늘 배운 내용을 바탕으로 실제 API 엔드포인트 설계 및 추가 기능 고려.
