import datetime
from io import BytesIO

import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import plotly.graph_objects as go
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report

# 1. 페이지 기본 설정
st.set_page_config(page_title="Ultra Fast Stock Analyzer", layout="wide")

# 2. 캐싱 로직 (데이터 로딩 속도 최적화)
@st.cache_data
def get_krx_list():
    url = 'http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13'
    df = pd.read_html(url, header=0, flavor='bs4', encoding='EUC-KR')[0]
    return df[['회사명', '종목코드']].copy()

def get_code(name):
    if name.isdigit() and len(name) == 6: return name
    df = get_krx_list()
    res = df[df['회사명'] == name]['종목코드'].values
    return f"{res[0]:06}" if len(res) > 0 else None

# --- UI 레이아웃 ---
st.title("⚡ 초고속 주가 분석 & 리포트")

with st.sidebar:
    st.header("🔍 설정")
    company_name = st.text_input("회사명 또는 코드", value="삼성전자")
    
    # 날짜 범위 설정 (기본 최근 1년)
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=365)
    selected_dates = st.date_input("조회 기간", [start_date, end_date])
    
    run_analysis = st.button("데이터 불러오기", use_container_width=True)

# --- 메인 로직 ---
if "df" not in st.session_state:
    st.session_state.df = None

if run_analysis:
    with st.spinner("데이터를 가져오는 중..."):
        code = get_code(company_name)
        if code:
            # 인덱스를 컬럼으로 변환 (Profiling 최적화)
            df = fdr.DataReader(code, selected_dates[0], selected_dates[1]).reset_index()
            st.session_state.df = df
            st.session_state.code = code
        else:
            st.error("종목을 찾을 수 없습니다.")

if st.session_state.df is not None:
    df = st.session_state.df
    
    tab1, tab2 = st.tabs(["📊 주가 차트", "📑 상세 분석 리포트"])

    # --- tab1 내부에 추가할 분석 로직 ---
    with tab1:
        # 1. 기술적 지표 계산
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        
        # 2. 수익률 계산
        first_price = df['Close'].iloc[0]
        last_price = df['Close'].iloc[-1]
        total_return = ((last_price - first_price) / first_price) * 100

        # 3. 상단에 요약 지표(Metric) 표시
        m1, m2, m3 = st.columns(3)
        m1.metric("현재가", f"{int(last_price):,}원")
        m2.metric("기간 수익률", f"{total_return:.2f}%", f"{total_return:.2f}%")
        m3.metric("최고가", f"{int(df['High'].max()):,}원")

        # 4. 이동평균선이 포함된 차트 업데이트
        fig = go.Figure()
        # 캔들스틱
        fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
        # 이동평균선 추가
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA5'], name="MA5", line=dict(color='orange', width=1)))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA20'], name="MA20", line=dict(color='blue', width=1)))
        
        fig.update_layout(template="plotly_white", height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.info("💡 리포트 생성을 위해 '분석 시작' 버튼을 눌러주세요. (최근 250일치 데이터로 최적화됨)")
        
        # 버튼을 눌러야만 Profiling 실행 (중요: 리소스 절약)
        if st.button("🚀 상세 분석 시작 (약 5초 소요)"):
            with st.spinner("불필요한 연산을 제외하고 핵심 통계만 추출 중..."):
                # 최적화 핵심 설정
                # 1. 데이터 양 제한 (최근 약 1년치 영업일)
                target_df = df.tail(250) 
                
                # 2. ProfileReport 경량화 옵션
                pr = ProfileReport(
                    target_df,
                    title=f"{company_name} Analysis Report",
                    minimal=True,          # 복잡한 상관계수 등 계산 생략
                    correlations=None,     # 속도 저하 주범 1 제거
                    interactions=None,     # 속도 저하 주범 2 제거
                    explorative=False,
                    samples={"head": 5, "tail": 5}
                )
                
                # 결과 출력
                st_profile_report(pr)

# 3. 배포용 파일 갱신 안내
# uv export --format requirements-txt > requirements.txt