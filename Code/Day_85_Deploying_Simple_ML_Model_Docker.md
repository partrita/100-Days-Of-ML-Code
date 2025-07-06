# Day 85: Docker를 사용한 간단한 ML 모델 배포 (Deploying a simple ML model using Docker)

## 학습 목표
- 어제 학습한 Docker 기본 지식을 바탕으로, 실제 머신러닝 모델 API를 Docker 컨테이너로 실행하고 외부에서 접근하는 전체 과정 복습 및 심화.
- Docker 이미지 레지스트리(예: Docker Hub)에 이미지를 푸시(Push)하고 풀(Pull)하는 방법 이해.
- 간단한 클라우드 환경(예: 가상 머신)에 Docker 컨테이너를 배포하는 개념 학습.
- Docker Compose를 사용한 다중 컨테이너 애플리케이션 관리의 기초 소개 (선택 사항).

## 1. 전체 배포 시나리오 복습
어제까지의 과정을 통해 다음을 수행했습니다:
1.  **머신러닝 모델 학습 및 저장**: `iris_binary_model.pkl` (및 필요시 `iris_scaler.pkl`)
2.  **Flask API 서버 작성**: `flask_api_v2.py` (모델 로드, 예측 엔드포인트 `/predict_iris` 구현)
3.  **필요 라이브러리 목록 작성**: `requirements.txt`
4.  **Dockerfile 작성**: Flask API 서버를 실행하는 Docker 이미지 빌드 지침 정의
5.  **Docker 이미지 빌드**: `docker build -t my-iris-api:v1 .`
6.  **Docker 컨테이너 로컬 실행**: `docker run -d -p 5001:5001 --name my-api-container my-iris-api:v1`
7.  **로컬에서 API 테스트**: `curl` 또는 Postman 사용

오늘은 이 로컬에서 실행한 Docker 컨테이너를 다른 환경에서도 사용할 수 있도록 하는 단계를 추가적으로 고려합니다.

## 2. Docker 이미지 레지스트리 (Docker Image Registry)
- **정의**: Docker 이미지를 저장하고 관리하며 공유하는 중앙 저장소입니다.
- **종류**:
    - **Docker Hub**: Docker에서 제공하는 공식 공용 레지스트리. 무료로 공개 이미지를 저장하거나, 유료로 비공개 이미지를 저장할 수 있습니다.
    - **프라이빗 레지스트리 (Private Registry)**: 기업이나 조직 내부에 자체적으로 구축하거나, 클라우드 서비스 제공업체(AWS ECR, GCP Container Registry/Artifact Registry, Azure ACR 등)가 제공하는 비공개 레지스트리를 사용할 수 있습니다.

### 가. Docker Hub에 이미지 푸시 (Push)
1.  **Docker Hub 계정 생성**: [Docker Hub 웹사이트](https://hub.docker.com/)에서 계정을 만듭니다.
2.  **로컬 Docker에 로그인**: 터미널에서 `docker login` 명령을 실행하고 Docker Hub 사용자 이름과 비밀번호를 입력합니다.
    ```bash
    docker login
    # Username: <your_dockerhub_username>
    # Password: <your_dockerhub_password>
    ```
3.  **이미지 태그 변경 (선택 사항, 권장)**: Docker Hub에 푸시하려면 이미지 이름을 `<DockerHub_사용자이름>/<이미지_이름>:<태그>` 형식으로 지정해야 합니다. 기존 이미지에 새 태그를 추가하거나, 빌드 시 이 형식으로 이름을 지정합니다.
    ```bash
    # 기존 이미지에 새 태그 추가
    docker tag my-iris-api:v1 <your_dockerhub_username>/my-iris-api:v1

    # 또는 빌드 시 바로 지정
    # docker build -t <your_dockerhub_username>/my-iris-api:v1 .
    ```
4.  **이미지 푸시**: `docker push <DockerHub_사용자이름>/<이미지_이름>:<태그>`
    ```bash
    docker push <your_dockerhub_username>/my-iris-api:v1
    ```
    - 푸시가 완료되면 Docker Hub 웹사이트의 내 레포지토리에서 이미지를 확인할 수 있습니다. (기본적으로 공개 이미지로 생성됨)

### 나. 다른 환경에서 이미지 풀 (Pull) 및 실행
- Docker가 설치된 다른 서버나 컴퓨터에서 Docker Hub (또는 다른 레지스트리)에 있는 이미지를 가져와 실행할 수 있습니다.
1.  **(선택) 해당 환경에서 Docker 로그인**: 비공개 이미지의 경우 필요. 공개 이미지는 로그인 없이 풀 가능.
2.  **이미지 풀**: `docker pull <DockerHub_사용자이름>/<이미지_이름>:<태그>`
    ```bash
    docker pull <your_dockerhub_username>/my-iris-api:v1
    ```
3.  **컨테이너 실행**: 로컬에서 실행했던 것과 동일한 `docker run` 명령 사용.
    ```bash
    docker run -d -p 5001:5001 --name my-remote-api-container <your_dockerhub_username>/my-iris-api:v1
    ```
    - 이제 해당 서버의 IP 주소와 5001번 포트를 통해 API에 접근할 수 있습니다. (방화벽 설정 등 필요할 수 있음)

## 3. 간단한 클라우드 환경에 Docker 컨테이너 배포 (개념)
- 클라우드 서비스 제공업체(AWS, GCP, Azure 등)의 가상 머신(VM) 인스턴스를 사용하여 Docker 컨테이너를 배포할 수 있습니다.

### 일반적인 단계 (예: AWS EC2, GCP Compute Engine)
1.  **클라우드 플랫폼에서 VM 인스턴스 생성**:
    - 운영체제 선택 (예: Ubuntu, Amazon Linux 등 Docker 지원 OS).
    - 인스턴스 크기(CPU, 메모리) 선택.
    - 네트워크 설정 (보안 그룹/방화벽 규칙에서 API 포트 - 예: 5001 - 허용).
2.  **VM 인스턴스에 접속 (SSH)**.
3.  **Docker 설치**: VM 인스턴스에 Docker Engine을 설치합니다. (각 클라우드 플랫폼 및 OS별 설치 가이드 참조)
    ```bash
    # 예시: Ubuntu에 Docker 설치
    # sudo apt update
    # sudo apt install -y docker.io
    # sudo systemctl start docker
    # sudo systemctl enable docker
    # sudo usermod -aG docker $USER # 현재 사용자를 docker 그룹에 추가 (재로그인 필요)
    ```
4.  **Docker 이미지 풀**: Docker Hub 또는 프라이빗 레지스트리에서 배포할 이미지를 풀합니다.
    ```bash
    docker pull <your_dockerhub_username>/my-iris-api:v1
    ```
5.  **Docker 컨테이너 실행**: `docker run` 명령으로 컨테이너를 실행합니다.
    ```bash
    docker run -d -p 5001:5001 --name my-cloud-api-container <your_dockerhub_username>/my-iris-api:v1
    # --restart always 옵션을 추가하면 VM 재부팅 시 컨테이너 자동 시작
    # docker run -d -p 5001:5001 --name my-cloud-api-container --restart always <your_dockerhub_username>/my-iris-api:v1
    ```
6.  **API 테스트**: VM 인스턴스의 공인 IP 주소와 매핑된 포트(예: `http://<VM_Public_IP>:5001/predict_iris`)로 API를 테스트합니다.

### 클라우드 관리형 컨테이너 서비스
- 위 방법은 VM에 직접 Docker를 설치하고 관리하는 방식입니다.
- 클라우드 제공업체들은 컨테이너 배포 및 관리를 더 쉽게 해주는 관리형 서비스도 제공합니다:
    - **AWS**: ECS (Elastic Container Service), EKS (Elastic Kubernetes Service), Fargate, App Runner
    - **GCP**: Cloud Run, GKE (Google Kubernetes Engine), App Engine
    - **Azure**: ACI (Azure Container Instances), AKS (Azure Kubernetes Service), App Service
- 이러한 서비스들은 컨테이너 오케스트레이션, 자동 확장, 로드 밸런싱, 모니터링 등을 더 쉽게 설정하고 관리할 수 있도록 도와줍니다. (더 고급 주제)

## 4. Docker Compose 사용 기초 (선택 사항)
- **정의**: 여러 개의 Docker 컨테이너로 구성된 애플리케이션을 정의하고 실행하기 위한 도구입니다.
- `docker-compose.yml`이라는 YAML 파일을 사용하여 애플리케이션의 서비스, 네트워크, 볼륨 등을 설정합니다.
- **ML 배포에서의 활용 예시**:
    - Flask/Django API 서버 컨테이너.
    - 데이터베이스 컨테이너 (예: PostgreSQL, MySQL).
    - 메시지 큐 컨테이너 (예: RabbitMQ, Redis - 비동기 작업용).
    - 모니터링 도구 컨테이너 (예: Prometheus, Grafana).
- 여러 컨테이너를 한 번의 명령(`docker-compose up`, `docker-compose down`)으로 쉽게 관리할 수 있습니다.

### 간단한 `docker-compose.yml` 예시 (Flask API만 있는 경우)
```yaml
# docker-compose.yml
version: '3.8' # Docker Compose 파일 형식 버전

services:
  my-api-service: # 서비스 이름 (임의 지정)
    image: <your_dockerhub_username>/my-iris-api:v1 # 사용할 Docker 이미지
    # build: . # Dockerfile이 있는 현재 디렉토리에서 이미지를 빌드할 수도 있음
    container_name: my-compose-api-container # 실행될 컨테이너 이름
    ports:
      - "5001:5001" # 호스트 포트:컨테이너 포트
    restart: always # 컨테이너 비정상 종료 시 자동 재시작
    # environment: # 환경 변수 설정 (필요시)
    #   - FLASK_ENV=production
```
- 위 파일을 `docker-compose.yml`로 저장하고, 해당 디렉토리에서 다음 명령 실행:
    - `docker-compose up -d` : 백그라운드에서 서비스 시작 (이미지가 없으면 풀 또는 빌드)
    - `docker-compose down` : 서비스 중지 및 컨테이너, 네트워크 등 제거
    - `docker-compose logs my-api-service` : 해당 서비스 로그 확인

## 5. 운영 환경에서의 추가 고려 사항
- **WSGI 서버 사용**: Flask/Django의 내장 개발 서버는 운영 환경에 적합하지 않습니다. Gunicorn, uWSGI와 같은 WSGI 서버를 Dockerfile의 `CMD` 또는 `ENTRYPOINT`에서 사용하여 애플리케이션을 실행해야 합니다.
    - 예: `CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5001", "flask_api_v2:app"]`
        - `-w 4`: 워커 프로세스 4개 사용
        - `-b 0.0.0.0:5001`: 모든 인터페이스의 5001번 포트에서 바인딩
        - `flask_api_v2:app`: `flask_api_v2.py` 파일의 `app` Flask 객체를 의미
- **HTTPS 설정**: 실제 서비스에서는 보안을 위해 HTTPS를 사용해야 합니다. 이는 보통 로드 밸런서나 리버스 프록시(Nginx 등) 수준에서 처리합니다.
- **환경 변수 관리**: API 키, 데이터베이스 접속 정보 등 민감한 정보는 Docker 이미지에 직접 하드코딩하지 않고, 컨테이너 실행 시 환경 변수로 주입하거나 Docker Secrets, Vault 등을 사용합니다.
- **로깅 및 모니터링**: 컨테이너 로그를 수집하고 분석할 수 있는 시스템(ELK Stack, CloudWatch Logs 등)과 애플리케이션 및 인프라 모니터링 도구(Prometheus, Grafana, Datadog 등)를 설정합니다.
- **CI/CD (Continuous Integration / Continuous Deployment)**: 코드 변경 시 자동으로 이미지를 빌드, 테스트하고 레지스트리에 푸시하며, 배포까지 자동화하는 파이프라인을 구축합니다. (Jenkins, GitLab CI, GitHub Actions 등)

## 실습 아이디어
1.  어제 만든 `my-iris-api:v1` 이미지를 Docker Hub의 본인 계정에 푸시해보세요.
2.  (가능하다면) 다른 컴퓨터나 클라우드 VM에 Docker를 설치하고, 푸시한 이미지를 풀하여 컨테이너를 실행하고 API를 테스트해보세요.
3.  (선택) 간단한 `docker-compose.yml` 파일을 작성하여 어제 만든 API 컨테이너를 실행해보세요.
4.  (선택) Dockerfile의 `CMD`를 Gunicorn을 사용하도록 변경하고 이미지를 다시 빌드하여 실행해보세요. (Gunicorn 설치 필요: `requirements.txt`에 추가)

## 추가 학습 자료
- [Docker Hub Quickstart](https://docs.docker.com/docker-hub/)
- [Deploying Docker containers on AWS (AWS Documentation)](https://aws.amazon.com/docker/)
- [Docker Compose Overview](https://docs.docker.com/compose/)
- [Best practices for writing Dockerfiles](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)

## 다음 학습 내용
- Day 86: 캡스톤 프로젝트 아이디어 브레인스토밍 (Brainstorming Capstone Project ideas) - 지금까지 배운 내용을 종합적으로 활용할 수 있는 프로젝트 아이디어 구상.
