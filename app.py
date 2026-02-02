import datetime
from io import BytesIO

import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import plotly.graph_objects as go
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report

# AI 예측을 위한 라이브러리
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST, TSMixer, DLinear

# 1. 페이지 기본 설정
st.set_page_config(page_title="AI Stock Analyzer (Advanced)", layout="wide")

# 2. 캐싱 로직
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
st.title("🚀 AI 주가 분석 & 멀티 모델 예측 리포트")

with st.sidebar:
    st.header("🔍 설정")
    company_name = st.text_input("회사명 또는 코드", value="삼성전자")
    
    end_date = datetime.date.today()
    # 데이터 부족 문제를 위해 조회 기간을 충분히 (최근 5년) 가져옵니다.
    start_date = end_date - datetime.timedelta(days=365*5) 
    selected_dates = st.date_input("조회 기간", [start_date, end_date])
    
    forecast_horizon = st.slider("예측 기간 (일)", min_value=7, max_value=60, value=30)
    
    run_analysis = st.button("데이터 불러오기", use_container_width=True)

# --- 메인 로직 ---
if "df" not in st.session_state:
    st.session_state.df = None

if run_analysis:
    with st.spinner("데이터를 가져오는 중..."):
        code = get_code(company_name)
        if code:
            # FinanceDataReader로 데이터 로드
            df = fdr.DataReader(code, selected_dates[0], selected_dates[1]).reset_index()
            if not df.empty:
                st.session_state.df = df
                st.session_state.code = code
                st.session_state.company_name = company_name
            else:
                st.error("해당 기간에 데이터가 존재하지 않습니다.")
        else:
            st.error("종목 코드를 찾을 수 없습니다.")

if st.session_state.df is not None:
    df = st.session_state.df
    
    tab1, tab2, tab3 = st.tabs(["📊 주가 차트", "📑 데이터 리포트", "🔮 AI 예측 분석"])

    with tab1:
        # 기술적 지표 계산
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        
        m1, m2, m3 = st.columns(3)
        curr_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        change = curr_price - prev_price
        
        m1.metric("현재가", f"{int(curr_price):,}원", f"{int(change):,}원")
        m2.metric("기간 최고가", f"{int(df['High'].max()):,}원")
        m3.metric("거래량 (전일)", f"{int(df['Volume'].iloc[-1]):,}")

        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="주가"))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA5'], name="5일선", line=dict(color='orange', width=1)))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA20'], name="20일선", line=dict(color='blue', width=1)))
        fig.update_layout(template="plotly_white", height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if st.button("🚀 데이터 프로파일링 리포트 생성"):
            with st.spinner("리포트 생성 중..."):
                pr = ProfileReport(df, minimal=True)
                st_profile_report(pr)

    with tab3:
        st.subheader("🧠 딥러닝 모델 기반 주가 예측")
        st.warning("⚠️ AI 예측은 참고용이며, 투자 책임은 본인에게 있습니다. 데이터가 적을수록 변동성이 큽니다.")
        
        if st.button("📈 AI 모델 학습 및 예측 시작"):
            # 1. 데이터 준비
            nf_df = df[['Date', 'Close']].copy()
            nf_df.columns = ['ds', 'y']
            nf_df['unique_id'] = 'STK_01' # 단일 종목 식별자
            
            # 2. 파라미터 설정
            h = int(forecast_horizon)
            # 데이터가 적은 문제를 해결하기 위해 input_size(과거 참조 창)를 
            # 예측 기간의 3~4배로 늘려 더 많은 패턴을 보게 합니다.
            input_size = h * 3 
            
            if len(nf_df) < input_size + h:
                st.error(f"데이터가 부족합니다. (필요: {input_size + h}행, 현재: {len(nf_df)}행)")
            else:
                with st.spinner("AI가 최근 5년 데이터를 학습 중입니다..."):
                    try:
                        # 3. 모델 정의 (n_series=1 명시 및 파라미터 최적화)
                        models = [
                            # 장기 패턴에 강한 PatchTST
                            PatchTST(h=h, input_size=input_size, patch_len=16, max_steps=150, scaler_type='standard'),
                            # 변수 간 혼합 특성을 보는 TSMixer
                            TSMixer(h=h, input_size=input_size, n_series=1, max_steps=150, scaler_type='standard'),
                            # 추세와 계절성을 분해하는 DLinear (가장 안정적)
                            DLinear(h=h, input_size=input_size, max_steps=150, scaler_type='standard')
                        ]
                        
                        # 4. 학습 및 예측
                        nf = NeuralForecast(models=models, freq='B') # 'B'는 Business Day (주말 제외)
                        nf.fit(df=nf_df)
                        forecast = nf.predict().reset_index()

                        # 5. 시각화
                        fig_res = go.Figure()
                        
                        # 최근 120일 실제 데이터
                        history = nf_df.tail(120)
                        fig_res.add_trace(go.Scatter(x=history['ds'], y=history['y'], name="실제 주가", line=dict(color='#333', width=2)))
                        
                        # 모델별 예측 데이터
                        colors = {'PatchTST': 'red', 'TSMixer': 'green', 'DLinear': 'blue'}
                        for m in ['PatchTST', 'TSMixer', 'DLinear']:
                            fig_res.add_trace(go.Scatter(
                                x=forecast['ds'], 
                                y=forecast[m], 
                                name=f"{m} 예측",
                                line=dict(dash='dash', color=colors[m])
                            ))
                        
                        fig_res.update_layout(
                            title=f"향후 {h}일 주가 예측 비교",
                            xaxis_title="날짜",
                            yaxis_title="가격",
                            template="plotly_white",
                            hovermode="x unified"
                        )
                        st.plotly_chart(fig_res, use_container_width=True)
                        
                        # 상세 수치 표
                        st.write("### 📋 모델별 예상 가격")
                        st.dataframe(forecast[['ds', 'PatchTST', 'TSMixer', 'DLinear']].set_index('ds').style.format("{:,.0f}"))
                        
                    except Exception as e:
                        st.error(f"예측 도중 오류가 발생했습니다: {e}")