# 지수평활화

import FinanceDataReader as fdr
import mplfinance as mpf
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression              # 선형회귀 모형
warnings.filterwarnings("ignore")                              # warning을 꺼줌

# 시장 데이터를 읽어오는 함수들을 정의한다.
def getData(code, datestart, dateend):
    df = fdr.DataReader(code, datestart, dateend ).drop(columns='Change') 
    return df

def getSymbols(market='KOSPI', sort='Marcap'):
    df = fdr.StockListing(market)
    ascending = False if sort == 'Marcap' else True
    df.sort_values(by=[sort], ascending = ascending, inplace=True)
    return df[ ['Code', 'Name', 'Market'] ]

# code에 해당하는 주식 데이터를 받아온다.
code = '005930'              # 삼성전자.
#code = '373220'             # LG 에너지솔루션.
date_start = (datetime.today()-timedelta(days=100)).date()          
df = getData(code, date_start, datetime.today().date())     

# 캔들차트를 출력해 본다 (이동평균 없이).
chart_style = 'default'                                            
marketcolors = mpf.make_marketcolors(up='red', down='blue')         
mpf_style = mpf.make_mpf_style(base_mpf_style=chart_style, marketcolors=marketcolors)

fig, ax = mpf.plot(
    data=df,                            # 받아온 데이터   
    volume=False,                       # True 또는 False.                   
    type='candle',                      # 캔들 차트
    style=mpf_style,                    # 위에서 정의
    figsize=(10,7),
    fontscale=1.1,
    returnfig=True                      # Figure 객체 반환
)


n = len(df)                 # 시계열의 길이
pred_ndays = 10             # 미래 예측 기간

# 그래프 출력에 유리한 형태로 데이터프레임 변환
# 가로가 두개 있으면 데이터프레임
df = df[ ['Close'] ].reset_index(drop=True)      # Pandas의 DataFrame 객체
df['m1'] = df['Close'].shift(1)                    # t-1 값
df['m2'] = df['Close'].shift(2)                    # t-2 값
df['m3'] = df['Close'].shift(3)                    # t-3 값
df['m4'] = df['Close'].shift(4)                    # t-4 값
df['m5'] = df['Close'].shift(5)                    # t-5 값  
df = df.iloc[5:]

# 선형회귀 기반 AR(5)모형 학습
# fit을 이용해서 트레이닝 
model = LinearRegression()
model.fit(df[['m1','m2','m3','m4','m5']], df['Close'])

# 선형회귀 기반 AR(5)모형 예측
ser = df['Close'][-5:]                              # 데이터 최신 5개 값
for step in range(pred_ndays):                      # 미래 예측
    past = pd.DataFrame(data={ f'm{i}': [ser.iloc[-i]] for i in range(1,6)} ) # 최신 5개값으로 데이터프레임을 만듬
    predicted = model.predict(past)[0]                                        # 예측 결과는 원소가 1개인 데이터프레임
    ser = pd.concat( [ser, pd.Series({n + step:predicted}) ])

# Axis에 추가
ax[0].plot(ser, color = 'red', linestyle ='--', linewidth=1.5, label = 'AR(5)')
ax[0].legend(loc='best') 


plt.show()

