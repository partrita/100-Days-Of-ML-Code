# Day 84: ML을 위한 Docker - 기초 (Docker for ML - Basics)

## 학습 목표
- 컨테이너화(Containerization)의 개념과 Docker의 역할 이해
- Docker 이미지(Image)와 컨테이너(Container)의 차이점 학습
- Dockerfile을 사용하여 머신러닝 애플리케이션(예: Flask API)을 위한 Docker 이미지를 빌드하는 방법 숙지
- 빌드된 이미지를 사용하여 Docker 컨테이너를 실행하고 API를 테스트하는 방법 학습
- Docker가 ML 모델 배포에 제공하는 이점 이해

## 1. 컨테이너화 (Containerization)와 Docker

### 가. 컨테이너화란?
- 애플리케이션과 그 실행에 필요한 모든 종속성(라이브러리, 시스템 도구, 코드, 런타임 등)을 패키지화하여 격리된 환경에서 실행할 수 있도록 하는 기술입니다.
- "내 컴퓨터에서는 잘 되는데, 다른 컴퓨터에서는 안 돼요"와 같은 문제를 해결하는 데 도움을 줍니다.
- 가상 머신(VM)과 유사하지만, 컨테이너는 호스트 운영체제(OS)의 커널을 공유하므로 더 가볍고 빠르며 효율적입니다.

### 나. Docker란?
- 컨테이너화 기술을 사용하는 대표적인 오픈소스 플랫폼입니다.
- Docker를 사용하면 애플리케이션을 쉽게 개발, 배포, 실행할 수 있습니다.
- **주요 구성 요소**:
    - **Docker Engine**: Docker 컨테이너를 만들고 실행하는 핵심 구성 요소 (Docker 데몬, REST API, CLI 클라이언트 포함).
    - **Docker Image**: 애플리케이션과 그 종속성을 포함하는 읽기 전용 템플릿. 컨테이너를 만드는 데 사용됩니다.
    - **Docker Container**: Docker 이미지의 실행 가능한 인스턴스. 격리된 환경에서 애플리케이션을 실행합니다.
    - **Dockerfile**: Docker 이미지를 빌드하기 위한 지침(명령어)을 담고 있는 텍스트 파일.
    - **Docker Hub / Docker Registry**: Docker 이미지를 저장하고 공유하는 레지스트리 서비스 (Docker Hub은 공용 레지스트리).

![Docker Concept](https://docs.docker.com/images/architecture.svg)
*(이미지 출처: Docker Documentation)*

## 2. Docker 이미지 (Image) vs 컨테이너 (Container)
- **이미지 (Image)**:
    - 애플리케이션을 실행하는 데 필요한 모든 것을 담고 있는 "설계도" 또는 "템플릿".
    - 운영체제, 라이브러리, 애플리케이션 코드 등이 계층(Layer) 구조로 쌓여 있습니다.
    - 읽기 전용(Read-only)입니다.
    - 예: 특정 버전의 파이썬, 필요한 라이브러리, Flask API 코드가 포함된 이미지.
- **컨테이너 (Container)**:
    - 이미지의 "실행 인스턴스".
    - 이미지를 기반으로 생성되며, 격리된 프로세스로 실행됩니다.
    - 컨테이너 내에서 파일을 생성하거나 수정하는 등의 변경 사항은 해당 컨테이너에만 적용됩니다 (이미지 자체는 변경되지 않음).
    - 하나의 이미지로 여러 개의 동일한 컨테이너를 생성하여 실행할 수 있습니다.

## 3. Dockerfile 작성하기 (ML API 예제)
- `Dockerfile`은 Docker 이미지를 어떻게 빌드할지를 정의하는 텍스트 파일입니다.
- 어제 만든 Flask API (`flask_api_v2.py`)와 학습된 모델(`iris_binary_model.pkl`)을 Docker 이미지로 만들어 보겠습니다.

### 프로젝트 구조 예시:
```
my_ml_api/
├── flask_api_v2.py       # Flask 애플리케이션 코드
├── iris_binary_model.pkl # 학습된 모델 파일
├── requirements.txt      # 필요한 파이썬 라이브러리 목록
└── Dockerfile            # Docker 이미지 빌드 지침
```

### `requirements.txt` 파일 내용 예시:
```
Flask==2.0.1  # 실제 사용하는 Flask 버전에 맞게
joblib==1.1.0
numpy==1.21.2
scikit-learn==0.24.2 # 모델 학습 시 사용한 scikit-learn 버전과 일치시키는 것이 좋음
# gunicorn # (선택) 운영 환경용 WSGI 서버
```
(위 버전은 예시이므로, 실제 환경에 맞게 `pip freeze > requirements.txt` 명령으로 생성하는 것이 좋습니다.)

### `Dockerfile` 내용 예시:
```dockerfile
# 1. 베이스 이미지 선택 (Base Image)
# 파이썬 3.8 버전을 기반으로 하는 공식 이미지 사용
FROM python:3.8-slim

# 2. 작업 디렉토리 설정 (Working Directory)
# 컨테이너 내에서 명령이 실행될 기본 경로 설정
WORKDIR /app

# 3. 필요한 파일 복사 (Copy Files)
# 현재 디렉토리(Dockerfile이 있는 위치)의 모든 파일을 컨테이너의 /app 디렉토리로 복사
COPY . /app
# 또는 특정 파일만 복사:
# COPY requirements.txt .
# COPY flask_api_v2.py .
# COPY iris_binary_model.pkl .

# 4. 종속성 설치 (Install Dependencies)
# requirements.txt 파일에 명시된 라이브러리 설치
RUN pip install --no-cache-dir -r requirements.txt

# 5. 포트 노출 (Expose Port)
# 컨테이너가 외부와 통신할 포트 지정 (Flask 앱이 5001번 포트 사용 가정)
EXPOSE 5001

# 6. 애플리케이션 실행 명령어 (CMD or ENTRYPOINT)
# 컨테이너가 시작될 때 실행될 기본 명령어
# 개발용: python flask_api_v2.py
# 운영용: gunicorn -w 4 -b 0.0.0.0:5001 flask_api_v2:app (Gunicorn 사용 시)
# 여기서는 개발용으로 간단히 실행
CMD ["python", "flask_api_v2.py"]
```

### Dockerfile 명령어 설명:
- `FROM`: 베이스 이미지를 지정합니다. (예: 특정 OS, 특정 프로그래밍 언어 런타임)
- `WORKDIR`: 컨테이너 내의 작업 디렉토리를 설정합니다. 이후 명령어들은 이 디렉토리를 기준으로 실행됩니다.
- `COPY <src> <dest>`: 호스트 머신의 파일이나 디렉토리를 컨테이너 내부로 복사합니다.
- `RUN <command>`: 이미지를 빌드하는 과정에서 컨테이너 내에서 실행할 명령입니다. (예: 패키지 설치)
- `EXPOSE <port>`: 컨테이너가 특정 포트를 사용함을 알립니다. 실제 포트 매핑은 컨테이너 실행 시 이루어집니다.
- `CMD ["executable","param1","param2"]` 또는 `ENTRYPOINT ["executable","param1","param2"]`: 컨테이너가 시작될 때 실행될 기본 명령을 지정합니다.
    - `CMD`: 컨테이너 실행 시 다른 명령어로 쉽게 덮어쓸 수 있습니다.
    - `ENTRYPOINT`: 컨테이너를 실행 파일처럼 사용하며, 실행 시 전달되는 인자들은 `ENTRYPOINT` 명령어의 파라미터로 추가됩니다.

## 4. Docker 이미지 빌드 및 컨테이너 실행

### 가. Docker 이미지 빌드
- `Dockerfile`이 있는 디렉토리에서 다음 명령을 실행합니다.
- `docker build -t <이미지_이름>:<태그> .`
    - `-t`: 이미지에 이름과 태그(버전 등)를 지정합니다. (예: `my-iris-api:v1`)
    - `.` : 현재 디렉토리의 `Dockerfile`을 사용하라는 의미입니다.

```bash
# Dockerfile이 있는 my_ml_api 디렉토리로 이동했다고 가정
cd path/to/my_ml_api

# 이미지 빌드 (예: 이미지 이름 my-iris-api, 태그 v1)
docker build -t my-iris-api:v1 .
```
- 빌드가 성공하면 로컬 Docker 이미지 목록에서 확인할 수 있습니다: `docker images`

### 나. Docker 컨테이너 실행
- 빌드된 이미지를 사용하여 컨테이너를 실행합니다.
- `docker run [OPTIONS] IMAGE [COMMAND] [ARG...]`
    - `-d`: 백그라운드에서 컨테이너 실행 (Detached mode).
    - `-p <호스트_포트>:<컨테이너_포트>`: 호스트 머신의 포트와 컨테이너의 포트를 매핑합니다. (예: `-p 5001:5001` -> 호스트의 5001번 포트를 컨테이너의 5001번 포트로 연결)
    - `--name <컨테이너_이름>`: 컨테이너에 이름을 지정합니다.
    - `<이미지_이름>:<태그>`: 실행할 이미지.

```bash
# 컨테이너 실행 (예: my-iris-api:v1 이미지 사용, 컨테이너 이름 my-api-container)
# 호스트의 5001번 포트를 컨테이너의 5001번 포트(Dockerfile에서 EXPOSE한 포트)로 매핑
docker run -d -p 5001:5001 --name my-api-container my-iris-api:v1
```
- 실행 중인 컨테이너 확인: `docker ps`
- 컨테이너 로그 확인: `docker logs my-api-container`

### 다. API 테스트 (컨테이너 실행 후)
- 이제 호스트 머신의 `localhost:5001` (또는 Docker가 실행 중인 머신의 IP:5001)로 API 요청을 보낼 수 있습니다.
```bash
# 어제 사용한 curl 명령어로 테스트
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}' \
  http://localhost:5001/predict_iris
```
- Docker 컨테이너 내에서 Flask API가 정상적으로 응답하는지 확인합니다.

### 라. 컨테이너 중지 및 삭제
- 컨테이너 중지: `docker stop my-api-container`
- 컨테이너 삭제: `docker rm my-api-container` (중지된 컨테이너만 삭제 가능)
- 이미지 삭제: `docker rmi my-iris-api:v1` (해당 이미지를 사용하는 컨테이너가 없어야 삭제 가능)

## 5. Docker가 ML 모델 배포에 제공하는 이점
- **환경 일관성**: 개발, 테스트, 운영 환경 간의 환경 차이로 인한 문제를 최소화합니다. 모델과 모든 종속성이 이미지에 포함되므로 "내 컴퓨터에서는 되는데..." 문제가 줄어듭니다.
- **재현성**: 동일한 Dockerfile과 코드를 사용하면 언제 어디서든 동일한 환경을 재현할 수 있습니다.
- **이식성**: Docker 이미지는 Docker가 설치된 어떤 환경(로컬 머신, 테스트 서버, 클라우드 등)에서도 동일하게 실행될 수 있습니다.
- **격리**: 각 컨테이너는 독립된 환경에서 실행되므로, 다른 애플리케이션이나 시스템과의 충돌을 방지합니다.
- **쉬운 배포 및 확장**: Docker 이미지를 사용하여 새로운 서버에 애플리케이션을 쉽게 배포할 수 있습니다. Kubernetes와 같은 컨테이너 오케스트레이션 도구와 함께 사용하면 서비스 확장 및 관리가 용이해집니다.
- **빠른 배포 사이클**: 이미지 빌드 및 배포 과정을 자동화하여 개발 및 배포 속도를 높일 수 있습니다 (CI/CD 파이프라인).
- **자원 효율성**: 가상 머신보다 가볍고 빠르게 시작되며, 호스트 시스템의 자원을 더 효율적으로 사용합니다.

## Docker 사용 시 추가 팁
- **`.dockerignore` 파일 사용**: 이미지 빌드 시 불필요한 파일(예: `.git`, `__pycache__`, 가상환경 폴더 등)이 이미지에 포함되지 않도록 제외합니다.
- **이미지 크기 최적화**:
    - 가벼운 베이스 이미지 사용 (예: `python:3.8-slim-buster`).
    - `RUN` 명령어를 최소화하고, 여러 명령어를 `&&`로 연결하여 레이어 수 줄이기.
    - 불필요한 파일이나 빌드 중간 산출물 삭제.
    - 멀티 스테이지 빌드(Multi-stage builds) 사용.
- **보안**: 베이스 이미지의 취약점 점검, 최소 권한 원칙 적용 등.

## 추가 학습 자료
- [Docker 공식 문서 - Get Started](https://docs.docker.com/get-started/)
- [Docker Tutorial for Beginners (YouTube - Programming with Mosh)](https://www.youtube.com/watch?v=pTFZFxd4hOI)
- [A Docker Tutorial for Data Scientists (Towards Data Science)](https://towardsdatascience.com/a-docker-tutorial-for-data-scientists-part-i-232725033758)
- [Dockerizing a Python Flask Application (Real Python)](https://realpython.com/dockerizing-flask-with-compose-and-machine-learning/)

## 다음 학습 내용
- Day 85: Docker를 사용한 간단한 ML 모델 배포 (Deploying a simple ML model using Docker) - 오늘 배운 내용을 바탕으로 실제 배포 시나리오 및 클라우드 연동 가능성 탐색.
