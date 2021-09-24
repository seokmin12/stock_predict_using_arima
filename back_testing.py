import FinanceDataReader as fdr
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pandas import DataFrame
import numpy as np

now = datetime.now()

symbol = '005930'

start = now - relativedelta(months=6)
start = start.strftime('%Y-%m-%d')
end = now.strftime('%Y-%m-%d')

df = fdr.DataReader(symbol, start, end)
df = df.reset_index()
df['day'] = df['Date']
df['price'] = df['Close']

data = df[['day', 'price']]

data_len = len(data['price'])

now_data_list = list(data['price'].tail(1))
past_data_list = list(data['price'].head(data_len - 1)

i = 0
past_data_len = []
for i in range(len(past_data_list)):
    past_data_len.append(i)
    i += 1
n = 0
now_data_len = []
for n in range(len(now_data_list)):
    now_data_len.append(n)
    n += 1
now_data_raw = {'day': now_data_len,
                'price': now_data_list}
now_data_set = DataFrame(now_data_raw)
now_data_set.index = now_data_set['day']
now_data_set.set_index('day', inplace=True)

past_data_raw = {'day': past_data_len,
                 'price': past_data_list}
past_data_set = DataFrame(past_data_raw)
past_data_set.index = past_data_set['day']
past_data_set.set_index('day', inplace=True)

import pmdarima as pm

arima_model = pm.auto_arima(past_data_set.price.values, start_p=0, start_q=0, test='adf', trace=True, error_action='ignore')
prediction = arima_model.predict()

best_order = tuple(arima_model.order)

from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(past_data_set.price.values.astype('float64'), order=best_order)

model_fit = model.fit(trend='c', full_output=True, disp=True)
forecast_data = model_fit.forecast(steps=steps)

pred_y = forecast_data[0].tolist()

today = datetime.strptime(str(now_data.tail(1)['day'].values[0]), '%Y-%m-%dT%H:%M:%S.%f000')


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


value = int(now_data.tail(1)['price'].values[0])
array = [round(pred_y[-1]), round(prediction[-1])]

best_price = find_nearest(array, value)
print(
    f"오늘 날짜: {today.strftime('%Y-%m-%d')} 실제 가격: {format(value, ',')}원 학습된 데이터의 날짜: {end} 예상 가격: {format(best_price, ',')}원")
