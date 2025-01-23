
import streamlit as st
import FinanceDataReader as fdr
import mplfinance as mpf
import json
import pandas as pd
import warnings
from streamlit_lottie import st_lottie
from datetime import datetime, timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing   
from sklearn.linear_model import LinearRegression              
warnings.filterwarnings("ignore")      

# 지수평활와 예측을 추가하는 함수 
def addESPrediction(data, ax, pred_ndays):
    n = len(data)
    ser = data['Close'].reset_index(drop=True)      
    model = ExponentialSmoothing(ser, trend='add', seasonal='add', seasonal_periods=5).fit()    # 학습 완료.
    past = ser.iloc[-5:]
    predicted = model.predict(start= n, end=n+pred_ndays-1)
    predicted.rename('Close', inplace=True)                 
    joined = pd.concat([past, predicted],axis=0)                                      
    ax.plot(joined, color = 'aqua', linestyle ='--', linewidth=1.5, label = 'ES')
    ax.legend(loc='best')          

# 자기회귀 예측을 추가하는 함수 
def addARPrediction(data, ax, pred_ndays):
    n = len(data)
# 그래프 출력에 유리한 형태로 데이터프레임 변환.
    df = data[ ['Close'] ].reset_index(drop=True)      # Pandas의 DataFrame 객체.     
    df['m1'] = df['Close'].shift(1)                    # t-1 값.
    df['m2'] = df['Close'].shift(2)                    # t-2 값.
    df['m3'] = df['Close'].shift(3)                    # t-3 값.
    df['m4'] = df['Close'].shift(4)                    # t-4 값.
    df['m5'] = df['Close'].shift(5)                    # t-5 값.     
    df = df.iloc[5:]
# 선형회귀 기반  AR(5)모형 학습.
    model = LinearRegression()
    model.fit(df[['m1','m2','m3','m4','m5']], df['Close'])
# 선형회귀 기반  AR(5)모형 예측.
    ser = df['Close'][-5:]                            
    for step in range(pred_ndays):                    
        past = pd.DataFrame(data={ f'm{i}': [ser.iloc[-i]] for i in range(1,6)} ) 
        predicted = model.predict(past)[0]                                        
        ser = pd.concat( [ser, pd.Series({n + step:predicted}) ])
    
    ax.plot(ser, color = 'red', linestyle ='--', linewidth=1.5, label = 'AR(5)')
    ax.legend(loc='best')    


# JSON을 읽어 들이는 함수
def loadJSON(path):
    f = open(path, 'r')
    res = json.load(f)
    f.close()
    return res

# 로고 Lottie와 타이틀 출력
col1, col2 = st.columns([1,2])
with col1:
    lottie = loadJSON('lottie-stock-candle-loading.json')
    st_lottie(lottie, speed=1, loop=True, width=150, height=150)
with col2:
    ''
    ''
    st.title('주식 추세 예측')

# 시장 데이터를 읽어오는 함수들을 정의
@st.cache_data
def getData(code, datestart, dateend):
    df = fdr.DataReader(code,datestart, dateend ).drop(columns='Change')  
    return df

@st.cache_data
def getSymbols(market='KOSPI', sort='Marcap'):
    df = fdr.StockListing(market)
    ascending = False if sort == 'Marcap' else True
    df.sort_values(by=[sort], ascending= ascending, inplace=True)
    return df[ ['Code', 'Name', 'Market'] ]

# 세션 상태를 초기화
if 'ndays' not in st.session_state:
    st.session_state['ndays'] = 30

if 'code_index' not in st.session_state:
    st.session_state['code_index'] = 0

if 'chart_style' not in st.session_state:
    st.session_state['chart_style'] = 'default'

if 'volume' not in st.session_state:
    st.session_state['volume'] = True

if 'pred_ndays' not in st.session_state:
    st.session_state['pred_ndays'] = 5

# 사이드바에서 폼을 통해서 차트 인자를 설정
with st.sidebar.form(key="chartsetting", clear_on_submit=True):
    st.header('차트 설정')
    ''
    ''
    symbols = getSymbols()
    choices = zip( symbols.Code , symbols.Name , symbols.Market )
    choices = [ ' : '.join( x ) for x in choices ]  # Code, Name, Market을 한개의 문자열
    choice = st.selectbox( label='종목:', options = choices, index=st.session_state['code_index'] )
    code_index = choices.index(choice)
    code = choice.split()[0]                        # 실제 code 부분만 떼어 가져옴
    ''
    ''
    ndays = st.slider(
        label='기간 (days):', 
        min_value= 20,
        max_value= 365, 
        value=st.session_state['ndays'],
        step = 1)
    ''
    ''
    chart_styles = ['default', 'binance', 'blueskies', 'brasil', 'charles', 'checkers', 'classic', 'yahoo','mike', 'nightclouds', 'sas', 'starsandstripes']
    chart_style = st.selectbox(label='차트 스타일:',options=chart_styles,index = chart_styles.index(st.session_state['chart_style']))
    ''
    ''
    volume = st.checkbox('거래량', value=st.session_state['volume'])
    ''
    ''

    pred_ndays = st.slider(
        label='예측 기간 (days):',
        min_value= 1,
        max_value= 10,
        value=st.session_state['pred_ndays'],
        step=1,
    )

    if st.form_submit_button(label="OK"):
        st.session_state['ndays'] = ndays
        st.session_state['code_index'] = code_index
        st.session_state['chart_style'] = chart_style
        st.session_state['volume'] = volume
        st.session_state['pred_ndays'] = pred_ndays
        st.rerun()

# 캔들 차트 출력 함수
def plotChart(data, pred_ndays):
    chart_style = st.session_state['chart_style']
    marketcolors = mpf.make_marketcolors(up='red', down='blue')
    mpf_style = mpf.make_mpf_style(base_mpf_style= chart_style, marketcolors=marketcolors)

    fig, ax = mpf.plot(
        data,
        volume=st.session_state['volume'],
        type='candle',
        style=mpf_style,
        figsize=(10,7),
        fontscale=1.1,
        returnfig=True                 
    )

    addESPrediction(data, ax[0], pred_ndays)
    addARPrediction(data, ax[0], pred_ndays)

    st.pyplot(fig)


date_start = (datetime.today()-timedelta(days=st.session_state['ndays'])).date()
df = getData(code, date_start, datetime.today().date())     
chart_title = choices[st.session_state['code_index'] ]
st.markdown(f'<h3 style="text-align: center; color: red;">{chart_title}</h3>', unsafe_allow_html=True)
plotChart(df, st.session_state['pred_ndays'])


