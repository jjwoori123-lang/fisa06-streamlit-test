import datetime
import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import plotly.graph_objects as go
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST, TSMixer, DLinear

st.set_page_config(page_title="AI Stock Analyzer (Continuous)", layout="wide")

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

st.title("🚀 AI 주가 분석 (예측 연속성 보정 버전)")

with st.sidebar:
    st.header("🔍 설정")
    company_name = st.text_input("회사명 또는 코드", value="삼성전자")
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=365*5) 
    selected_dates = st.date_input("조회 기간", [start_date, end_date])
    forecast_horizon = st.slider("예측 기간 (일)", min_value=7, max_value=60, value=30)
    run_analysis = st.button("데이터 불러오기", use_container_width=True)

if "df" not in st.session_state:
    st.session_state.df = None

if run_analysis:
    with st.spinner("데이터를 가져오는 중..."):
        code = get_code(company_name)
        if code:
            df = fdr.DataReader(code, selected_dates[0], selected_dates[1]).reset_index()
            if not df.empty:
                st.session_state.df = df
                st.session_state.code = code
                st.session_state.company_name = company_name
            else: st.error("데이터가 없습니다.")
        else: st.error("코드를 찾을 수 없습니다.")

if st.session_state.df is not None:
    df = st.session_state.df
    tab1, tab2, tab3 = st.tabs(["📊 차트", "📑 리포트", "🔮 AI 예측"])

    with tab1:
        fig = go.Figure(data=[go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="주가")])
        fig.update_layout(template="plotly_white", height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("🧠 연속성이 보정된 딥러닝 예측")
        if st.button("📈 AI 모델 학습 및 예측 시작"):
            nf_df = df[['Date', 'Close']].copy()
            nf_df.columns = ['ds', 'y']
            nf_df['unique_id'] = 'STK_01'
            last_close = nf_df['y'].iloc[-1]
            
            nf_df['y'] = np.log1p(nf_df['y'])
            h = int(forecast_horizon)
            input_size = h * 2 
            
            with st.spinner("예측 연결 고리를 맞추는 중..."):
                try:
                    models = [
                        PatchTST(h=h, input_size=input_size, max_steps=500, learning_rate=1e-3),
                        TSMixer(h=h, input_size=input_size, n_series=1, max_steps=500, learning_rate=1e-3),
                        DLinear(h=h, input_size=input_size, max_steps=500, learning_rate=1e-3)
                    ]
                    nf = NeuralForecast(models=models, freq='B')
                    nf.fit(df=nf_df)
                    forecast = nf.predict().reset_index()

                    # 역변환 및 연속성 보정
                    for m in ['PatchTST', 'TSMixer', 'DLinear']:
                        forecast[m] = np.expm1(forecast[m])
                        # 마지막 실젯값과 첫 예측값의 차이를 보정
                        offset = last_close - forecast[m].iloc[0]
                        forecast[m] = forecast[m] + offset

                    # 시각화용 실젯값 복구
                    nf_df['y'] = np.expm1(nf_df['y'])
                    
                    fig_res = go.Figure()
                    history = nf_df.tail(90)
                    fig_res.add_trace(go.Scatter(x=history['ds'], y=history['y'], name="실제 주가", line=dict(color='#333', width=2)))
                    
                    colors = {'PatchTST': '#EF553B', 'TSMixer': '#00CC96', 'DLinear': '#636EFA'}
                    for m in ['PatchTST', 'TSMixer', 'DLinear']:
                        fig_res.add_trace(go.Scatter(x=forecast['ds'], y=forecast[m], name=f"{m} 예측", line=dict(dash='dash', color=colors[m])))
                    
                    fig_res.update_layout(title="연속성 보정 완료", template="plotly_white", hovermode="x unified")
                    st.plotly_chart(fig_res, use_container_width=True)
                    st.dataframe(forecast[['ds', 'PatchTST', 'TSMixer', 'DLinear']].set_index('ds').style.format("{:,.0f}"))
                except Exception as e:
                    st.error(f"오류: {e}")