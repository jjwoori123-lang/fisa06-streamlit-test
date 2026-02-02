FROM python:3.11-slim

WORKDIR /app

# 1. 필수 라이브러리 및 CPU용 torch 설치 (용량 다이어트 핵심)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir neuralforecast

# 2. 나머지 라이브러리 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. 소스 복사
COPY . .

# 4. 실행 (Render용 포트 설정)
ENTRYPOINT ["sh", "-c", "streamlit run app.py --server.port $PORT --server.address 0.0.0.0"]