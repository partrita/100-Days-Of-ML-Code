# Day 93: 프로젝트를 위한 간단한 UI 또는 프레젠테이션 구축 (Building a simple UI or presentation for the project)

## 학습 목표
- 캡스톤 프로젝트의 결과물을 효과적으로 시연하거나 전달하기 위한 방법 구상.
- (선택) Streamlit, Gradio와 같은 도구를 사용하여 머신러닝 모델과 상호작용할 수 있는 간단한 웹 UI 프로토타입 제작.
- (선택) 프로젝트의 주요 내용, 과정, 결과를 요약하는 프레젠테이션 자료 구조 설계 및 내용 구상.
- 프로젝트의 가치와 성과를 명확하게 전달하는 방법 고민.

## 1. 결과물 시연/전달의 중요성
- 아무리 훌륭한 모델을 개발했더라도, 그 결과를 다른 사람(동료, 평가자, 잠재적 사용자 등)에게 효과적으로 보여주지 못하면 그 가치를 제대로 인정받기 어렵습니다.
- 시각적인 데모나 잘 정리된 발표 자료는 프로젝트의 이해도를 높이고, 성과를 명확하게 전달하는 데 큰 도움이 됩니다.

## 2. 간단한 웹 UI 프로토타입 제작 (선택 사항)
- 학습된 머신러닝 모델을 사용자가 직접 체험해볼 수 있도록 간단한 웹 기반 인터페이스를 만드는 것은 매우 효과적인 시연 방법입니다.
- 복잡한 웹 개발 지식 없이도 파이썬만으로 빠르게 UI를 만들 수 있는 도구들이 있습니다.

### 가. Streamlit
- **개념**: 데이터 과학자와 머신러닝 엔지니어를 위한 파이썬 기반 오픈소스 앱 프레임워크. 최소한의 코드로 인터랙티브한 웹 앱을 만들 수 있습니다.
- **장점**:
    - 배우기 쉽고 사용이 간편함. HTML/CSS/JavaScript 지식 거의 불필요.
    - 데이터 시각화(Matplotlib, Seaborn, Plotly 등) 및 Pandas DataFrame 연동 용이.
    - 위젯(버튼, 슬라이더, 텍스트 입력 등)을 쉽게 추가하여 사용자 입력 가능.
    - 코드 변경 시 앱 자동 업데이트.
- **설치**: `pip install streamlit`
- **실행**: `streamlit run your_script.py`

#### Streamlit 예시 (간단한 텍스트 입력 -> 예측 결과 출력)
```python
# streamlit_app.py (Day 83의 Flask API와 유사한 모델 로직 가정)
import streamlit as st
import joblib
import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer # 실제로는 학습 시 사용한 vectorizer 로드

# --- 모델 및 벡터라이저 로드 (실제 경로 및 파일명 사용) ---
try:
    model = joblib.load('iris_binary_model.pkl') # Day 83에서 만든 모델 예시
    # tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl') # 학습 시 사용한 TF-IDF 벡터라이저
    st.sidebar.success("모델 로드 성공!")
except FileNotFoundError:
    st.sidebar.error("모델 또는 벡터라이저 파일을 찾을 수 없습니다.")
    model = None
    # tfidf_vectorizer = None
except Exception as e:
    st.sidebar.error(f"로드 중 오류: {e}")
    model = None
    # tfidf_vectorizer = None

# --- 임시 벡터라이저 및 클래스 이름 (실제 프로젝트에 맞게 수정) ---
# 실제로는 학습에 사용된 벡터라이저를 로드해야 함.
# 여기서는 간단한 기능을 위해 임시로 만듦.
class DummyVectorizer:
    def transform(self, text_list):
        # 실제 TF-IDF 변환 로직 대신, 길이 기반의 더미 특징 생성
        return np.array([[len(text) / 10.0, 1 if 'great' in text else 0, 1 if 'bad' in text else 0, 0] for text in text_list])

if model: # 모델이 로드된 경우에만 벡터라이저 초기화 (예시)
    # 실제로는 저장된 tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl') 등을 사용해야 함
    # 아래는 임시 더미 벡터라이저 사용
    # tfidf_vectorizer = DummyVectorizer() # 실제 프로젝트에서는 학습된 벡터라이저 사용!
    # 임시로 모델이 4개 특성을 받는다고 가정
    pass


class_names_iris_binary = ['setosa', 'versicolor'] # Iris 이진 분류 예시
# --- 임시 설정 끝 ---


st.title("간단한 붓꽃 품종 예측 앱 (Binary)")
st.write("붓꽃의 특징을 입력하면 품종을 예측합니다 (Setosa vs Versicolor).")
st.write("(주의: 이 UI는 Day 83의 Iris 이진 분류 모델 `iris_binary_model.pkl`을 사용하며, 4개의 수치형 입력을 가정합니다.)")

# 사용자 입력 (Iris 데이터의 4개 특성)
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

input_features = [sepal_length, sepal_width, petal_length, petal_width]

if st.button("예측하기!"):
    if model is not None: # and tfidf_vectorizer is not None: # 벡터라이저도 확인
        try:
            # 입력 데이터를 모델이 이해할 수 있는 형태로 변환
            # (텍스트 입력이라면: input_vector = tfidf_vectorizer.transform([user_input_text]))
            # (수치 입력이라면: input_array = np.array(input_features).reshape(1, -1))
            input_array = np.array(input_features).reshape(1, -1)

            # 예측 수행
            prediction = model.predict(input_array)[0]
            prediction_proba = model.predict_proba(input_array)[0]

            predicted_class_name = class_names_iris_binary[int(prediction)]

            st.subheader("예측 결과:")
            st.write(f"**예측된 품종:** {predicted_class_name} (ID: {prediction})")

            st.write("**각 품종일 확률:**")
            for i, class_name in enumerate(class_names_iris_binary):
                st.write(f"  - {class_name}: {prediction_proba[i]:.4f}")

        except Exception as e:
            st.error(f"예측 중 오류가 발생했습니다: {e}")
    else:
        st.error("모델이 로드되지 않아 예측을 수행할 수 없습니다.")

st.sidebar.header("모델 정보")
st.sidebar.write(f"사용된 모델: Iris Binary Classifier (Logistic Regression)")
st.sidebar.write(f"모델 파일: iris_binary_model.pkl")
# st.sidebar.write(f"벡터라이저 파일: tfidf_vectorizer.pkl (예시)")

# 실행: 터미널에서 streamlit run streamlit_app.py
```
- **주의**: 위 Streamlit 코드는 Iris 이진 분류 예시이며, 실제 캡스톤 프로젝트의 모델과 입력에 맞게 `model.load`, `tfidf_vectorizer.load` (필요시), 입력 위젯, 예측 로직 등을 수정해야 합니다. 텍스트 입력을 받는다면 `st.text_area` 등을 사용하고, TF-IDF 변환 등을 적용해야 합니다.

### 나. Gradio
- **개념**: 머신러닝 모델을 위한 빠르고 쉬운 UI 생성 도구. Streamlit과 유사하게 몇 줄의 코드만으로 데모 앱을 만들 수 있습니다.
- **장점**:
    - 다양한 입력/출력 인터페이스(이미지, 오디오, 텍스트, DataFrame 등) 지원.
    - 모델 공유 및 임베딩 기능.
- **설치**: `pip install gradio`

#### Gradio 예시 (간단한 텍스트 입력 -> 예측 결과 출력)
```python
# gradio_app.py (Gradio 버전)
import gradio as gr
import joblib
import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer

# --- 모델 및 벡터라이저 로드 (Streamlit 예시와 동일하게 가정) ---
try:
    model = joblib.load('iris_binary_model.pkl')
    # tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("모델 로드 성공!")
except FileNotFoundError:
    print("모델 또는 벡터라이저 파일을 찾을 수 없습니다.")
    model = None
    # tfidf_vectorizer = None
except Exception as e:
    print(f"로드 중 오류: {e}")
    model = None
    # tfidf_vectorizer = None

class_names_iris_binary = ['setosa', 'versicolor']
# --- 임시 설정 끝 ---

def predict_iris_gradio(sl, sw, pl, pw): # 입력 파라미터 순서대로 받음
    if model is None: # or tfidf_vectorizer is None:
        return "모델이 로드되지 않았습니다.", {"오류": 1.0}

    try:
        input_features = [sl, sw, pl, pw]
        input_array = np.array(input_features).reshape(1, -1)

        prediction = model.predict(input_array)[0]
        prediction_proba = model.predict_proba(input_array)[0]

        predicted_class_name = class_names_iris_binary[int(prediction)]

        # Gradio는 보통 딕셔너리 형태로 확률을 반환 (Label 컴포넌트 사용 시)
        confidences = {name: float(prob) for name, prob in zip(class_names_iris_binary, prediction_proba)}

        return f"예측: {predicted_class_name}", confidences
    except Exception as e:
        return f"오류: {e}", {"오류": 1.0}

# Gradio 인터페이스 정의
# inputs: 입력 위젯 리스트 (여기서는 숫자 입력 4개)
# outputs: 출력 위젯 리스트 (여기서는 텍스트와 레이블(확률 시각화) 컴포넌트)
iface = gr.Interface(
    fn=predict_iris_gradio,
    inputs=[
        gr.Number(label="Sepal Length (cm)"),
        gr.Number(label="Sepal Width (cm)"),
        gr.Number(label="Petal Length (cm)"),
        gr.Number(label="Petal Width (cm)")
    ],
    outputs=[
        gr.Textbox(label="예측 결과"),
        gr.Label(label="클래스별 확률", num_top_classes=len(class_names_iris_binary))
    ],
    title="간단한 붓꽃 품종 예측 앱 (Gradio)",
    description="붓꽃의 특징을 입력하면 품종(Setosa/Versicolor)을 예측합니다."
)

if __name__ == '__main__':
    iface.launch() # share=True 옵션으로 외부 공유 링크 생성 가능

# 실행: 터미널에서 python gradio_app.py
```

## 3. 프레젠테이션 자료 구조 설계 및 내용 구상 (선택 사항)
- 만약 프로젝트 결과를 발표해야 한다면, 명확하고 설득력 있는 프레젠테이션 자료가 필요합니다.
- **일반적인 프레젠테이션 구조**:
    1.  **표지 (Title Slide)**: 프로젝트 제목, 발표자 이름, 날짜.
    2.  **목차 (Table of Contents)**.
    3.  **서론 (Introduction)**:
        - 프로젝트 배경 및 동기.
        - 해결하고자 하는 문제 정의 (Problem Statement).
        - 프로젝트 목표 및 중요성.
    4.  **데이터 (Data)**:
        - 사용한 데이터셋 소개 (출처, 크기, 주요 특징).
        - 데이터 수집 및 전처리 과정 요약.
        - 주요 EDA 결과 (시각화 포함).
    5.  **방법론 (Methodology)**:
        - 적용한 머신러닝/딥러닝 모델 및 알고리즘 설명.
        - 특징 공학 내용.
        - 모델 학습 및 평가 방법 (교차 검증, 평가지표 등).
    6.  **결과 (Results)**:
        - 주요 모델들의 성능 비교 (표, 그래프).
        - 최종 선택된 모델의 상세 성능 (테스트 세트 기준).
        - 오류 분석 결과 (중요한 발견 위주).
    7.  **데모 (Demonstration)**: (가능하다면)
        - 개발한 UI 프로토타입 시연.
        - 또는 API 호출 및 결과 확인 과정 시연.
    8.  **결론 (Conclusion)**:
        - 프로젝트 결과 요약 및 주요 성과 강조.
        - 프로젝트의 한계점.
    9.  **향후 과제 (Future Work)**:
        - 추가적으로 개선할 수 있는 부분이나 확장 아이디어.
    10. **Q&A**.
    11. **(선택) 참고 자료 (References)**.

- **프레젠테이션 제작 팁**:
    - **청중 고려**: 청중의 수준과 관심사에 맞춰 내용과 용어 선택.
    - **핵심 메시지 강조**: 전달하고자 하는 가장 중요한 내용을 명확히.
    - **시각 자료 적극 활용**: 그래프, 다이어그램, 이미지 등을 사용하여 이해도 높이기. 텍스트는 간결하게.
    - **스토리텔링**: 문제 정의부터 해결 과정, 결과까지 자연스러운 흐름으로 이야기하듯 구성.
    - **시간 안배**: 발표 시간에 맞춰 각 슬라이드 내용 분량 조절.
    - **연습**: 충분한 연습을 통해 자신감 있고 매끄러운 발표 준비.

## 4. 프로젝트 가치와 성과 전달
- **수치화된 성과 제시**: "정확도 80%에서 88%로 향상", "응답 시간 0.5초 달성" 등 구체적인 수치로 성과 표현.
- **비즈니스/실생활 가치 연결**: 이 프로젝트가 실제 어떤 문제를 해결하고 어떤 긍정적인 영향을 줄 수 있는지 설명.
- **기술적 도전과 해결 과정**: 프로젝트를 진행하면서 겪었던 어려움과 이를 어떻게 해결했는지 공유하면 깊이 있는 인상을 줄 수 있음.
- **배운 점 및 성장**: 프로젝트를 통해 개인적으로 무엇을 배우고 어떤 역량이 향상되었는지 어필.

## 실습 아이디어
- 본인의 캡스톤 프로젝트 모델을 사용하여 Streamlit 또는 Gradio로 간단한 웹 UI 프로토타입을 만들어보세요.
    - 사용자로부터 필요한 입력을 받고, 모델 예측 결과를 보기 쉽게 출력합니다.
    - (텍스트 프로젝트) 텍스트 입력창, (이미지 프로젝트) 이미지 업로드 기능 등을 활용해보세요.
- 만약 발표가 예정되어 있다면, 오늘 배운 내용을 바탕으로 프레젠테이션 자료의 전체적인 구조를 설계하고 각 슬라이드에 들어갈 핵심 내용을 간략하게 정리해보세요.
- 프로젝트의 핵심 성과와 가치를 한두 문장으로 요약해보는 연습을 해보세요.

## 다음 학습 내용
- Day 94: 캡스톤 프로젝트 문서화 (Documenting the Capstone Project) - 프로젝트의 전 과정을 체계적으로 기록하고 정리하여 다른 사람이 이해하기 쉽도록 문서화. (README 작성, 코드 주석, 보고서 초안 등)
