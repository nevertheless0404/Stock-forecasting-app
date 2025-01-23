# 지수평활화

import FinanceDataReader as fdr
import mplfinance as mpf
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing   # Holt Winters 지수평활화 모형
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

#
# 예측선을 추가
#

n = len(df)                 # 시계열의 길이
pred_ndays = 10             # 미래 예측 기간

# 그래프 출력에 유리한 형태로 데이터프레임 변환.
ser = df['Close'].reset_index(drop=True)      # Pandas의 Series 객체

# ES 모델생성 및 학습.
model = ExponentialSmoothing(ser, trend='add', seasonal='add', seasonal_periods=5).fit() 

# 예측
past = ser.iloc[-5:]
predicted = model.predict(start= n, end=n+pred_ndays-1) # 모형 예측
predicted.rename('Close', inplace=True)                 # Name을 past의 'Close'와 같이 맞춤

# 과거 데이터와 예측을 이어붙인다.
joined = pd.concat([past, predicted],axis=0) 

# Axis에 추가
ax[0].plot(joined, color = 'aqua', linestyle ='--', linewidth=1.5, label = 'ES')
ax[0].legend(loc='best')  

#print(ser.head())
#print(ser.tail())
#print(past)
#print(predicted)
#print(joined)

#
# 출력
#

plt.show()
