FROM python:3.11-slim

# 라이브러리 빌드에 필요한 리눅스 패키지 설치 (NeuralForecast는 이게 필요할 수 있습니다)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip

# requirements.txt 설치 + 누락된 neuralforecast 추가 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir neuralforecast

COPY . .

EXPOSE 8501

# 실행 파일이 app.py이므로 이름을 맞춰줍니다.
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]