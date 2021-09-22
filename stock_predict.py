import numpy.linalg
from telegram.ext import Updater
from telegram.ext import CommandHandler

BOT_TOKEN = '2016299400:AAEBU2ymHXyC-sWij9u1B_NrjAhBG6mXnH8'

updater = Updater(token=BOT_TOKEN, use_context=True)
dispatcher = updater.dispatcher


def predict(update, context):
    from statsmodels.tsa.arima_model import ARIMA
    import FinanceDataReader as fdr
    from datetime import datetime
    from dateutil.relativedelta import relativedelta
    import pandas as pd

    now = datetime.now()

    start = now - relativedelta(months=6)
    end = now.strftime('%Y-%m-%d')
    try:
        symbol_df = pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download', header=0)[0]
        symbol_df = symbol_df.reset_index()
        symbol_df['code'] = symbol_df['종목코드']
        symbol_df['name'] = symbol_df['회사명']
        symbol_data = symbol_df[['code', 'name']]  # 종목 리스트 만들기

        symbol_input = context.args[0]

        if symbol_input.isdigit():  # 종목 이름으로 받을 때
            symbol = symbol_input
            symbol_input = symbol_input.lstrip('0')
            is_target = symbol_data['code'] == int(symbol_input)
            target = symbol_data[is_target]
            symbol_name = str(target['name'].values[0])
        else:
            is_target = symbol_data['name'] == symbol_input  # 종목 코드로 받을 때
            target = symbol_data[is_target]
            symbol = str(target['code'].values[0])
            if len(symbol) == 4:
                symbol = '00' + symbol
            if len(symbol) == 5:
                symbol = '0' + symbol
            symbol_name = symbol_input
        print(symbol)
        df = fdr.DataReader(symbol, start, end)
        df = df.reset_index()
        df['day'] = df['Date']
        df['price'] = df['Close']

        data = df[['day', 'price']]
        data.index = data['day']
        data.set_index('day', inplace=True)

        import pmdarima as pm

        arima_model = pm.auto_arima(data.price.values, start_p=0, start_q=0, test='adf', trace=True,
                                    error_action='ignore')   # 이 종목에 맞는 p d q 찾기
        best_order = tuple(arima_model.order)

        model = ARIMA(data.price.values.astype('float64'), order=best_order)

        model_fit = model.fit(trend='c', full_output=True, disp=1)  # 학습 진행

        days = 1

        forecast_data = model_fit.forecast(steps=days)
        pred_y = forecast_data[0].tolist()

        now_price = round(data['price'].tail(1).values[0])
        predict_price = round(pred_y[-1])
        if predict_price > now_price:
            context.bot.send_message(chat_id=update.effective_chat.id, text=f"{symbol_name} 다음 날 상승할 것으로 예상됩니다. 현재 가격: {format(now_price, ',')}원 예상 가격: {format(predict_price, ',')}원")
        elif predict_price < now_price:
            context.bot.send_message(chat_id=update.effective_chat.id, text=f"{symbol_name} 다음 날 하락할 것으로 예상됩니다. 현재 가격: {format(now_price, ',')}원 예상 가격: {format(predict_price, ',')}원")
        elif predict_price == now_price:
            context.bot.send_message(chat_id=update.effective_chat.id, text=f"{symbol_name} 다음 날 보합으로 예상됩니다. 현재 가격: {format(now_price, ',')}원 예상 가격: {format(predict_price, ',')}원")
    except IndexError as e:
        context.bot.send_message(chat_id=update.effective_chat.id, text="알 수 없는 종목입니다. 종목코드나 종목이름을 확인해 주세요.")
        print(e)
    except KeyError as e:
        context.bot.send_message(chat_id=update.effective_chat.id, text="알 수 없는 종목입니다. 종목코드나 종목이름을 확인해 주세요.")
        print(e)
    except ValueError as e:
        context.bot.send_message(chat_id=update.effective_chat.id, text="알 수 없는 종목입니다. 종목코드나 종목이름을 확인해 주세요.")
        print(e)
    except numpy.linalg.LinAlgError as e:
        context.bot.send_message(chat_id=update.effective_chat.id, text="일시적 오류입니다. 관리자에게 문의하세요.")
        print(e)
    except numpy.core._exceptions._UFuncOutputCastingError as e:
        context.bot.send_message(chat_id=update.effective_chat.id, text="일시적 오류입니다. 관리자에게 문의하세요.")
        print(e)


def help(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="/predict (종목코드/종목이름)")


predict_handler = CommandHandler('predict', predict)
help_handler = CommandHandler('help', help)

dispatcher.add_handler(predict_handler)
dispatcher.add_handler(help_handler)

updater.start_polling()
updater.idle()
