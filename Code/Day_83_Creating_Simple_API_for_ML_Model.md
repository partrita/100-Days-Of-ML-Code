# Day 83: ML 모델을 위한 간단한 API 만들기 (Creating a simple API for an ML model)

## 학습 목표
- 어제 학습한 Flask 기본 지식을 바탕으로, 실제 머신러닝 모델을 위한 API 엔드포인트를 설계하고 구현하는 과정 심화.
- API 요청(Request) 및 응답(Response) 형식 설계.
- 입력 데이터 유효성 검사(Validation)의 중요성과 간단한 구현 방법 학습.
- 오류 처리(Error Handling) 및 로깅(Logging)의 기초 개념 이해.
- API 문서화의 필요성 인지.

## 1. API 엔드포인트 설계 복습 및 구체화
- 어제 만든 `/predict_iris` 엔드포인트를 기준으로 좀 더 구체적인 설계를 고려합니다.

### 가. 요청 (Request) 형식
- **HTTP 메소드**: `POST` (예측을 위해 데이터를 서버로 전송하므로)
- **Content-Type**: `application/json` (JSON 형식으로 데이터 전송)
- **요청 본문 (Request Body) 예시**:
    ```json
    {
      "features": [5.1, 3.5, 1.4, 0.2] // Iris 데이터의 4개 특성 값
    }
    ```
    또는 좀 더 명시적으로:
    ```json
    {
      "sepal_length": 5.1,
      "sepal_width": 3.5,
      "petal_length": 1.4,
      "petal_width": 0.2
    }
    ```
    - 후자의 방식이 각 특성이 무엇인지 명확하게 알 수 있어 더 좋습니다. 여기서는 간단함을 위해 전자의 리스트 형태를 유지하되, 실제로는 명시적인 키-값 쌍을 권장합니다.

### 나. 응답 (Response) 형식
- **Content-Type**: `application/json`
- **성공 시 응답 본문 (HTTP Status Code: 200 OK) 예시**:
    ```json
    {
      "input_features": [5.1, 3.5, 1.4, 0.2],
      "predicted_class_id": 0,
      "predicted_class_name": "setosa",
      "probabilities": {
        "setosa": 0.98,
        "versicolor": 0.02
        // "virginica": 0.0 (다중 클래스였다면)
      },
      "model_version": "1.0.0" // (선택) 사용된 모델 버전 정보
    }
    ```
- **오류 시 응답 본문 (HTTP Status Code: 4xx, 5xx) 예시**:
    - 입력 데이터 오류 (400 Bad Request):
        ```json
        {
          "error_type": "InvalidInputError",
          "message": "입력된 'features'는 4개의 숫자 값으로 이루어진 리스트여야 합니다."
        }
        ```
    - 서버 내부 오류 (500 Internal Server Error):
        ```json
        {
          "error_type": "PredictionError",
          "message": "모델 예측 중 내부 오류가 발생했습니다."
        }
        ```

## 2. 입력 데이터 유효성 검사 (Input Validation)
- API는 다양한 경로로 호출될 수 있으므로, 예상치 못한 입력에 대해 방어적으로 코드를 작성해야 합니다.
- 유효성 검사는 잘못된 입력으로 인한 모델 오류나 서버 다운을 방지하고, 사용자에게 명확한 피드백을 제공합니다.

### 유효성 검사 항목 예시
- 필수 필드 존재 여부 (예: `features` 키가 있는지)
- 데이터 타입 확인 (예: `features`가 리스트인지, 각 요소가 숫자인지)
- 데이터 범위 확인 (예: 특성 값이 특정 범위 내에 있는지 - 필요하다면)
- 리스트 길이 확인 (예: `features` 리스트의 길이가 정확히 4인지)

### 간단한 유효성 검사 구현 (어제 코드에 추가/수정)
```python
# flask_api_v2.py (어제 코드 기반으로 수정)
from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging # 로깅 추가

app = Flask(__name__)

# 로깅 설정 (간단 설정)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# 모델 로드
try:
    model = joblib.load('iris_binary_model.pkl')
    app.logger.info("모델 로드 성공: iris_binary_model.pkl")
    # scaler = joblib.load('iris_scaler.pkl')
except FileNotFoundError:
    model = None
    # scaler = None
    app.logger.error("저장된 모델 파일을 찾을 수 없습니다. 'train_model.py'를 먼저 실행하세요.")
except Exception as e:
    model = None
    # scaler = None
    app.logger.error(f"모델 로드 중 오류 발생: {e}")

MODEL_VERSION = "1.0.0" # 모델 버전 정보

@app.route('/')
def home():
    return f"Iris Binary Classification API (v{MODEL_VERSION})"

@app.route('/health', methods=['GET']) # 헬스 체크 엔드포인트 추가
def health_check():
    if model:
        return jsonify({'status': 'ok', 'message': 'API is healthy and model is loaded.'}), 200
    else:
        return jsonify({'status': 'error', 'message': 'API is unhealthy, model not loaded.'}), 500


@app.route('/predict_iris', methods=['POST'])
def predict_iris():
    if model is None:
        app.logger.error("'/predict_iris' 호출 시 모델이 로드되지 않음.")
        return jsonify({'error_type': 'ModelNotLoadedError', 'message': '모델이 로드되지 않았습니다. 서버 로그를 확인하세요.'}), 500

    if not request.is_json:
        app.logger.warning("'/predict_iris' 호출 시 JSON 형식이 아닌 요청 받음.")
        return jsonify({'error_type': 'InvalidContentTypeError', 'message': 'Request Content-Type must be application/json'}), 400

    try:
        data = request.get_json()
        app.logger.info(f"요청 데이터: {data}")

        # --- 입력 데이터 유효성 검사 시작 ---
        if 'features' not in data:
            app.logger.warning("요청 데이터에 'features' 키가 없음.")
            return jsonify({'error_type': 'MissingFieldError', 'message': "'features' 필드가 요청에 포함되어야 합니다."}), 400

        features = data['features']

        if not isinstance(features, list):
            app.logger.warning("'features' 필드가 리스트 타입이 아님.")
            return jsonify({'error_type': 'InvalidInputTypeError', 'message': "'features' 필드는 리스트여야 합니다."}), 400

        if len(features) != 4:
            app.logger.warning(f"'features' 리스트 길이가 4가 아님: {len(features)}")
            return jsonify({'error_type': 'InvalidInputLengthError', 'message': f"'features' 리스트는 4개의 요소를 가져야 합니다. (현재: {len(features)}개)"}), 400

        if not all(isinstance(f, (int, float)) for f in features):
            app.logger.warning("'features' 리스트에 숫자가 아닌 값이 포함됨.")
            return jsonify({'error_type': 'InvalidFeatureTypeError', 'message': "'features' 리스트의 모든 요소는 숫자여야 합니다."}), 400
        # --- 입력 데이터 유효성 검사 끝 ---

        input_array = np.array(features).reshape(1, -1)

        # if scaler:
        #     input_array_scaled = scaler.transform(input_array)
        #     prediction_proba = model.predict_proba(input_array_scaled)[0]
        #     prediction = model.predict(input_array_scaled)[0]
        # else:
        prediction_proba = model.predict_proba(input_array)[0]
        prediction = model.predict(input_array)[0]

        class_names = ['setosa', 'versicolor']

        response_data = {
            'input_features': features,
            'predicted_class_id': int(prediction),
            'predicted_class_name': class_names[int(prediction)],
            'probabilities': {'setosa': round(float(prediction_proba[0]), 4),
                              'versicolor': round(float(prediction_proba[1]), 4)},
            'model_version': MODEL_VERSION
        }
        app.logger.info(f"예측 결과: {response_data}")
        return jsonify(response_data), 200

    except Exception as e:
        app.logger.error(f"예측 중 예외 발생: {str(e)}", exc_info=True) # exc_info=True로 스택 트레이스 로깅
        return jsonify({'error_type': 'InternalPredictionError', 'message': f'예측 중 내부 오류가 발생했습니다: {str(e)}'}), 500

if __name__ == '__main__':
    # 운영 환경에서는 Gunicorn, uWSGI 같은 WSGI 서버 사용 권장
    app.run(debug=False, host='0.0.0.0', port=5001) # debug=False로 변경
```

## 3. 오류 처리 (Error Handling) 및 로깅 (Logging)

### 가. 오류 처리
- `try-except` 블록을 사용하여 예측 과정에서 발생할 수 있는 예외(Exception)를 처리합니다.
- 사용자에게는 일반적인 오류 메시지를 반환하고, 서버 로그에는 자세한 오류 정보를 기록하여 디버깅에 활용합니다.
- HTTP 상태 코드를 적절히 사용하여 클라이언트가 오류 상황을 인지하도록 합니다.
    - `200 OK`: 성공
    - `400 Bad Request`: 클라이언트 요청 오류 (예: 잘못된 입력 형식)
    - `401 Unauthorized`: 인증 실패
    - `403 Forbidden`: 권한 없음
    - `404 Not Found`: 요청한 리소스 없음
    - `500 Internal Server Error`: 서버 내부 오류
    - `503 Service Unavailable`: 서비스 일시 사용 불가

### 나. 로깅
- 애플리케이션의 상태, 요청 정보, 오류 발생 등을 기록하는 것은 매우 중요합니다.
- 로깅을 통해 문제 발생 시 원인을 파악하고, 시스템 사용 현황을 분석할 수 있습니다.
- 파이썬의 내장 `logging` 모듈을 사용하거나, Flask 확장(예: `Flask-Logging`)을 사용할 수 있습니다.
- 로그 레벨(DEBUG, INFO, WARNING, ERROR, CRITICAL)을 적절히 사용하여 필요한 정보만 기록합니다.
- 위 `flask_api_v2.py` 코드에 간단한 로깅(`app.logger`)이 추가되었습니다.
    - `app.logger.info()`: 정보성 메시지.
    - `app.logger.warning()`: 경고 메시지.
    - `app.logger.error()`: 오류 메시지.
- 실제 운영 환경에서는 로그 파일로 저장하고, 로그 로테이션, 중앙 집중식 로그 관리 시스템(ELK Stack, Splunk 등)을 고려해야 합니다.

## 4. API 문서화 (API Documentation)
- API를 사용하는 다른 개발자나 시스템이 API를 올바르게 호출하고 응답을 이해할 수 있도록 명확한 문서가 필요합니다.
- 문서화 내용:
    - 엔드포인트 URL 및 HTTP 메소드
    - 요청 파라미터 설명 (필수 여부, 데이터 타입, 예시)
    - 요청 본문 형식 및 예시
    - 응답 형식 및 예시 (성공 시, 오류 시)
    - 인증 방법 (필요한 경우)
    - 사용 예제 코드
- **도구**:
    - **Swagger / OpenAPI Specification**: API 설계를 위한 표준 명세. 이를 기반으로 자동 문서 생성, 클라이언트 코드 생성 등이 가능.
    - **Flask-RESTx, FastAPI**: Swagger UI를 자동으로 생성해주는 기능을 가진 Flask 확장 또는 프레임워크.
    - **Postman**: API 테스트 도구지만, API 컬렉션을 문서화하고 공유하는 기능도 제공.
    - **Markdown 파일**: 간단한 API의 경우 직접 Markdown으로 작성할 수도 있습니다.

## 5. 추가 고려 사항 (간략히)
- **인증 및 권한 부여 (Authentication & Authorization)**: API 키, OAuth, JWT 등을 사용하여 허가된 사용자/시스템만 API에 접근하도록 제어.
- **요청 속도 제한 (Rate Limiting)**: 특정 시간 동안 특정 클라이언트가 보낼 수 있는 요청 수를 제한하여 API 남용 방지.
- **비동기 처리**: 예측 시간이 오래 걸리는 모델의 경우, 비동기 작업 큐(Celery, RabbitMQ 등)를 사용하여 요청을 즉시 반환하고 백그라운드에서 예측을 처리하는 방식 고려.
- **테스팅**: 단위 테스트(Unit Test), 통합 테스트(Integration Test)를 작성하여 API의 안정성 확보.
- **배포 환경**: 개발 서버(`app.run()`)는 운영 환경에 적합하지 않음. Gunicorn, uWSGI와 같은 WSGI(Web Server Gateway Interface) 서버와 Nginx 같은 웹 서버를 함께 사용하여 배포. (다음 주제에서 다룰 Docker도 활용)

## 실습 아이디어
- 어제 만든 `flask_api.py`를 기반으로 `flask_api_v2.py`와 같이 유효성 검사, 오류 처리, 로깅 기능을 추가해보세요.
- 다양한 잘못된 입력을 `curl`이나 Postman으로 보내면서 오류 응답과 서버 로그를 확인해보세요.
- (선택) `/health` 엔드포인트를 추가하여 API 서버의 상태를 확인할 수 있도록 해보세요.

## 다음 학습 내용
- Day 84: ML을 위한 Docker - 기초 (Docker for ML - Basics) - 모델과 API 서버를 컨테이너화하여 배포 환경을 일관되게 만드는 방법.
