import os
from contextlib import closing
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sqlalchemy import select, Null, update
from sqlalchemy.orm import Session

from config import config
from models import Strategy
from trade_classes import Api_Trade_Method
from utils import create_session
from utils_db import getEngine

matplotlib.use('Agg')
from tensorflow.keras.models import load_model
import joblib

import emoji #https://carpedm20.github.io/emoji/

cnn_model = {
        'TONUSDT':
                    [
                    {
                            'coin': 'TONUSDT',
                            'interval': '1',
                            'model': '1m.keras',
                            'scelar': '1m.gz',
                            'window_size': 10,
                            'threshold_window': 0.01,
                            'predict_percent': 0.45,
                            'trade': 'long',
                            'numeric': ['open', 'high', 'low', 'close', 'volume'],
                            'comment': 'Обучен на 1 месяце 04.2024'
                    },
                    {
                            'coin': 'TONUSDT',
                            'interval': '5',
                            'model': '5m.keras',
                            'scelar': '5m.gz',
                            'window_size': 4,
                            'threshold_window': 0.01,
                            'predict_percent': 0.45,
                            'trade': 'long',
                            'numeric': ['open', 'high', 'low', 'close', 'volume'],
                            'comment': 'Обучен на 2 месяце 04.2024 и 03.2024'
                    },
                        {
                            'coin': 'TONUSDT',
                            'interval': '15',
                            'model': '15m.keras',
                            'scelar': '15m.gz',
                            'window_size': 3,
                            'threshold_window': 0.01,
                            'predict_percent': 0.61,
                            'trade': 'long',
                            'numeric': ['open', 'high', 'low', 'close', 'volume'],
                            'comment': 'Обучен на 2 месяцах 04.2024 и 03.2024'
                        },
                        {
                            'coin': 'TONUSDT',
                            'interval': '30',
                            'model': '30m.keras',
                            'scelar': '30m.gz',
                            'window_size': 3,
                            'threshold_window': 0.03,
                            'predict_percent': 0.5,
                            'trade': 'long',
                            'numeric': ['open', 'high', 'low', 'close', 'volume'],
                            'comment': 'Обучен на 2 месяцах 04.2024 и 03.2024'
                        },
                    ]
            }

stg_dict = {
                'stg_name': 'ai_cnn',
                'name': 'Ai CNN',
                'desc': 'Стратегия торговли Ai CNN',
                'amount': '1.0',
                'deals': 1,
                'ctg': 'linear',
                'exch': '',
                'x': 1,
                'decimal_part': 2,
                'move': 'two',  # up, down, two
            }

class Strategy_AI_CNN(Api_Trade_Method):
    symbol: str
    uuid: str
    predict_dict: None

    def __init__(self, stg_id, user_id):
        self.stg_id = stg_id
        self.api_session = self.makeSession(stg_id=stg_id)
        self.session = create_session()
        self.user_id = user_id
        self.stg_name = 'ai_cnn'
        self.stg_dict = self.getStgDictFromBD()
        config.getTgId(user_id=user_id)
        self.fee = self.getFeeRate(symbol=self.symbol)

    def Start(self):
        # получаем теущую цену +
        # если цена равна круглой цене в шаге который у нас указан, происходит покупка
        # тут же ставиться стоп на цену шага выше
        # запоминается время покупки в базу и стоп по этой покупке
        # если происходит покупка снова по этой цене, а старый тейкпрофит еще в базе, удаляется старый тейкпрофит и ставится новый двойной
        if self.CheckStopStartStg():
            #self.checkTakeProfit()
            self.predict_dict = self.checkAiPredict()
            current_lastprice_list = self.check_price(self.predict_dict['predict'])
            if self.predict_dict['buy'] and current_lastprice_list:
                print(f'current_lastprice_list {current_lastprice_list}')
                self.tryBuy(current_lastprice_list)
        else:
            print(f"answer {ddict['answer']}")

    def tryBuy(self, current_lastprice_list):
        print(f'покупаем')
        # - тикер берет текущую цену
        # - если timestamp есть в истории покупок в диапазоне 0.5% от текущей цены и времени, тогда не совершаем сделку
        rounded_data = [{**current_lastprice_list, 'close': round(current_lastprice_list['close'], self.stg_dict['decimal_part'])} for current_lastprice_list in data]
        print(rounded_data)
        tradeQ = self.session.query(TradeHistory).order_by(TradeHistory.id.desc())
        priceCountQ = self.session.query(TradeHistory).filter_by(price=str(lastPrice), filled=False)
        if priceCountQ.first():
            lastTX = tradeQ.first()
            if int(self.stg_dict['deals']) >= priceCountQ.count() and lastTX.price == str(lastPrice):
                # print(f'IF {lastPrice} = {lastTX.price}')
                # print(f"deals {self.stg_dict['deals']} >= {priceCountQ.count()}")
                pass
            else:
                # print(f'ELSE {lastPrice} | {lastTX.price}')
                if (self.stg_dict['move'] == 'down' and float(lastPrice) < float(lastTX.price)) or (
                        self.stg_dict['move'] == 'up' and float(lastPrice) > float(lastTX.price)) or self.stg_dict[
                    'move'] == 'two':
                    print(f"price COUNT 2 {priceCountQ.count()} | deals = {self.stg_dict['deals']}\n")
                    if self.stg_dict['fibo']:
                        # fibo_next_num
                        fibo_amount = 1
                        tx = self.BuyMarket(self.symbol, fibo_amount)
                    else:
                        tx = self.BuyMarket(self.symbol, self.stg_dict['amount'])
                    if 'error' not in tx:
                        order_info = self.OrderHistory(tx['result']['orderId'])
                        # print(f'order_info 1 {order_info}')
                        buy_price = order_info['result']['list'][0]['cumExecValue']
                        tx['result']['price'] = buy_price
                        tp_order_dict = {
                            'ctg': self.stg_dict['ctg'],
                            'side': 'Sell',
                            'symbol': self.symbol,
                            'orderType': 'Market',
                            'tp_price': float(buy_price) + float(self.stg_dict['step']),
                            'qty': self.stg_dict['amount'],
                            'uuid': tx['result']['orderId']
                        }
                        tp = self.TakeProfit(order_dict=tp_order_dict)
                        if tp is not None and isinstance(tp, dict) and 'error' not in tp:
                            last_tp = self.LastTakeProfitOrder(symbol=self.symbol, limit=1)
                            tx['result']['price'] = round(float(buy_price), self.decimal_part)
                            tx['result']['tpOrderId'] = last_tp['result']['list'][0]['orderId']
                            self.createTX(tx=tx, tp=last_tp)
                            config.message = emoji.emojize(
                                ":check_mark_button:") + f" Куплено {self.stg_dict['amount']} {self.symbol} повторно по {buy_price} [{self.stg_dict['name']}]"

                        else:
                            config.message = emoji.emojize(
                                ":check_mark_button:") + f" Куплено {self.stg_dict['amount']} {self.symbol} повторно по {buy_price} [{self.stg_dict['name']}]" \
                                                         f"\nTakeProfit не был установлен по причине: {tp}"
                        config.update_message = True
                    else:
                        config.message = tx['error']
                        config.update_message = True
        else:
            tx = self.BuyMarket(self.symbol, self.stg_dict['amount'])
            priceCountQ = self.session.query(TradeHistory).filter_by(price=str(lastPrice), filled=False)
            lastTX = tradeQ.first()
            # print(f'\nlastTX {lastTX.price}')
            print(
                f"else price COUNT - {priceCountQ.count()} | lastprice = {lastPrice}  |  | deals = {self.stg_dict['deals']}\n")
            if 'error' not in tx:
                # print(f"orderid {tx}\n")
                order_info = self.OrderHistory(tx['result']['orderId'])
                # print(f'order_info 2 {order_info}\n')
                buy_price = order_info['result']['list'][0]['cumExecValue']
                tx['result']['price'] = buy_price
                order_dict = {
                    'ctg': self.stg_dict['ctg'],
                    'side': 'Sell',
                    'symbol': self.symbol,
                    'orderType': 'Market',
                    'tp_price': float(buy_price) + float(self.stg_dict['step']),
                    'qty': self.stg_dict['amount'],
                    'uuid': tx['result']['orderId']
                }
                tp = self.TakeProfit(order_dict=order_dict)
                # print(f'tp {tp}')
                if tp is not None and isinstance(tp, dict) and 'error' not in tp:
                    last_tp = self.LastTakeProfitOrder(symbol=self.symbol, limit=1)
                    # print(f"tx buy {tx['result']}")
                    tx['result']['price'] = round(float(buy_price), self.decimal_part)
                    tx['result']['tpOrderId'] = last_tp['result']['list'][0]['orderId']
                    tx_obj = self.createTX(tx=tx, tp=last_tp)
                    # print(f"tp else {tp}")
                    config.message = emoji.emojize(
                        ":check_mark_button:") + f" Куплено {self.stg_dict['amount']} {self.symbol} по {buy_price} [{self.stg_dict['name']}]"
                else:
                    config.message = emoji.emojize(
                        ":check_mark_button:") + f" Куплено {self.stg_dict['amount']} {self.symbol} по {buy_price} [{self.stg_dict['name']}]" \
                                                 f"\nTakeProfit не был установлен по причине: {tp}"
                config.update_message = True
            else:
                config.message = tx['error']
                config.update_message = True
        # - если купили ставим стоп на шаг выше
        # - если это вторая покупка по цене, отменяем первый стоп и и ставим новый умноженный на 2
        self.session.close()

    def check_price(self, last_price_dict):
        # Получаем текущий timestamp
        current_time = datetime.utcnow()
        current_last_price_predict = []
        for item in last_price_dict:
            # Преобразование timestamp из словаря в объект datetime
            record_time = datetime.utcfromtimestamp(int(item['timestamp']) / 1000)
            # Разница времени между текущим моментом и timestamp в словаре
            time_difference = current_time - record_time
            # Проверяем, прошло ли менее 60 секунд
            time_interval = 60 * int(item['interval']) * 2
            if time_difference.total_seconds() <= time_interval:
                current_last_price_predict.append(item['close'])
        return current_last_price_predict

    def createTX(self, tx: dict, tp: dict):
        # print(f"tx {tx['result']}")
        # print(f"tp {tp['result']}")
        tx_dict = {
            'price_clean': tp['result']['list'][0]['lastPriceOnCreated'],
            'tp': tp['result']['list'][0]['price'],
            'side': tp['result']['list'][0]['side'],
            'qty': tp['result']['list'][0]['qty']
        }
        createTx = TradeHistory(price=tx['result']['price'], tx_id=tx['result']['orderId'], tx_dict=tx_dict,
                                stg_id=self.stg_id, user_id=self.user_id,
                                tp_id=tx['result']['tpOrderId']
                                )
        self.session.add(createTx)
        self.session.commit()
        return createTx

    def checkAiPredict(self):
        predict_price = []
        result = {}
        for key, value in cnn_model.items():
            if key == self.symbol:
                for item in value:
                    limit = 0
                    if item['interval'] == '1':
                        limit = 4320
                    if item['interval'] == '5':
                        limit = 2000
                    if item['interval'] == '15':
                        limit = 1500
                    if item['interval'] == '30':
                        limit = 1500
                    klines =self.get_kline(symbol=self.symbol, interval=item['interval'], limit=limit)
                    #print(f'klines: {klines}')
                    predict = CNNPredict(model_dict=item,
                                   klines=klines)
                    predict_price.append(predict.run())
                    result['buy'] = True
                    result['predict'] = predict_price
        print(f'predict list tyt: {predict_price}')
        return result

    def checkTakeProfit(self):
        self.cleanHistory()

    def cleanHistory(self):
        tp = self.LastTakeProfitOrder(symbol=self.symbol, limit=50)
        historyQ = self.session.query(TradeHistory).filter_by(filled=False)
        # print(f'tp cleanHistory {tp} - {self.symbol}')
        for tx in historyQ:
            if not any(tx.tp_id in d.values() for d in tp['result']['list']):
                tx.filled = True
                tx_dict = tx.tx_dict
                fee = None
                earn = None
                percent = None
                try:
                    fee = round(((float(tx_dict['price_clean']) * float(self.fee['takerFeeRate'])) + (
                                float(tx_dict['tp']) * float(self.fee['makerFeeRate']))) * int(tx_dict['qty']), 3)
                    earn = round(((float(tx_dict['tp']) - float(tx_dict['price_clean'])) * int(tx_dict['qty'])) - fee,
                                 3)
                    percent = round((earn / float(tx_dict['price_clean'])) * 100, 3)
                    config.message = emoji.emojize(
                        ":money_with_wings:") + f" Сработал TakeProfit {round(float(tx_dict['tp']), 3)}, чистая прибыль {earn} usdt ({percent}%), комиссия {fee} [{self.stg_dict['name']}, {self.symbol}]"
                    config.update_message = True
                except:
                    pass
                self.session.commit()

        self.session.close()

    '''telegram bot func'''

    # включить или выключить стратегию торговли
    def CheckStopStartStg(self, change: bool = None) -> dict:
        return_dict = {}
        return_dict['answer'] = ''
        with closing(Session(getEngine())) as session:
            query = session.query(Strategy).filter_by(id=self.stg_id).one()
            if query.stg_dict:
                if change is not None:
                    query.start = False if query.start else True
                stg_dict = query.stg_dict
                if not stg_dict['amount']:
                    return_dict[
                        'answer'] = '<b>Нельзя запустить!</b>\n У вас не настроена сумма сделки' + emoji.emojize(
                        ":backhand_index_pointing_left:") + '\n'
                    query.start = False
            else:
                return_dict['answer'] = '<b>Нельзя запустить ' + emoji.emojize(
                    ":backhand_index_pointing_left:") + '</b>, у вас не указана стратегия\n'
            return_dict['start'] = query.start
            # print(f"start {return_dict['start']}")
            session.commit()
        return return_dict


    def returnHelpInfo(self, stg_id: int) -> str:
        return f"Настройка стратегии AI CNN" + emoji.emojize(":robot:") \
               + "\nВводите команды для настройки стратегии\n" \
               + f"\nдля этой стратегии укажите <b>id={stg_id}</b>\n" \
               + f"\n/stgedit ai_cnn id={stg_id} active true - Выбрать эту стратегию\n" \
               + f"/stgedit ai_cnn id={stg_id} ctg spot - spot или linear\n" \
               + f"/stgedit ai_cnn id={stg_id} x 2 - плечо\n" \
               + f"/stgedit ai_cnn id={stg_id} move up - Направление движения покупки up, down, two\n" \
                 f"/stgedit ai_cnn id={stg_id} amount 100 - сколько крипты за одну сделку\n"

    def getCommandValue(self, key: str, value: str) -> str:
        with closing(Session(getEngine())) as session:
            query = session.query(Strategy).filter_by(id=self.stg_id).one()
            query.stg_name = 'ai_cnn'
            if not query:
                result = f'Не создана стратегия торговли\n'
            else:
                if key.lower() == 'active':
                    if value:
                        result = f'<b>Стратегия AI CNN ' + emoji.emojize(
                            ":robot:") + f' активирована для {query.symbol} ({self.stg_id})</b>\nДля ее работы надо указать все настройки и после этого запустить.\n\n'
                        query.stg_dict = stg_dict
                        result += 'Вы обновили статегию, поэтому настройки выставлены <u>по умолчанию</u>, не забудьте их изменить\n\n'

                    else:
                        query.stg_name = Null
                        result = '<b>Стратегия отключена</b>'
                    session.commit()
                if query.stg_dict:

                    if key == 'deals':
                        if value:
                            try:
                                value = int(value)
                                temp_dict = {}
                                temp_dict = query.stg_dict
                                temp_dict['deals'] = value
                                stmt = update(Strategy).where(Strategy.id == self.stg_id).values(stg_dict=temp_dict)
                                session.execute(stmt)
                                session.commit()
                                result = f'Вы указали количество сделок на одну цену - <b>{value}</b>\n\n'
                            except ValueError:
                                result = f'Вы указали не формат, пример: 2\n\n'

                    if key == 'ctg':
                        if value.lower() == 'spot' or value.lower() == 'linear':
                            temp_dict = query.stg_dict
                            temp_dict['ctg'] = value
                            stmt = update(Strategy).where(Strategy.id == self.stg_id).values(stg_dict=temp_dict)
                            session.execute(stmt)
                            session.commit()
                            result = f'Вы указали <b>{value}</b> торговлю\n\n'
                        else:
                            result = f'Вы указали не правильную категорию торговли, можно только spot или linear (бессрочная)\n\n'

                    if key == 'move':
                        if value.lower() == 'up' or value.lower() == 'down' or value.lower() == 'two':
                            temp_dict = query.stg_dict
                            temp_dict['move'] = value
                            stmt = update(Strategy).where(Strategy.id == self.stg_id).values(stg_dict=temp_dict)
                            session.execute(stmt)
                            session.commit()
                            result = f'Вы указали движение торговли <b>{value}</b>\n\n'
                        else:
                            result = f'Вы указали не допустимое движение для торговли\n\n'

                    if key == 'amount':
                        if value:
                            try:
                                value = float(value)
                                temp_dict = {}
                                temp_dict = query.stg_dict
                                temp_dict['amount'] = value
                                stmt = update(Strategy).where(Strategy.id == self.stg_id).values(stg_dict=temp_dict)
                                session.execute(stmt)
                                session.commit()
                                result = f'Вы указали сумму одной сделки - <b>{value}</b>\n\n'
                            except ValueError:
                                result = f'Вы указали не правильную сумму сделки, пример: 100\n\n'
                    self.stg_dict = self.getStgDictFromBD()
                    result += self.getDescriptionStg()
                else:
                    result = f'Сначала активируйте торговую стратегию соотвествующей командой\n'
        return result

    # возврщает описание для телеги бота
    def getDescriptionStg(self) -> str:
        try:
            query = self.session.query(Strategy).filter_by(id=self.stg_id).one()
            answer = f'<b>{self.symbol}</b>:\n'
            answer += f'Торговля запушена ' + emoji.emojize(
                ":check_mark_button:") + '\n' if query.start else f'Торговля остановлена' + emoji.emojize(
                ":stop_sign:") + '\n'


            if self.stg_dict is None:
                answer += f'\n<b>Стратегия торговли не установлена</b>\n'
            else:
                answer += f"\n<b>Описание</b> {self.stg_dict['desc']}\n\n"\
                         f"<b>Текущие настройки</b>\nСумма сделки: {self.stg_dict['amount']}" \
                         f"\nКоличество сделок на одну цену: {self.stg_dict['deals']}\nКатегория: {self.stg_dict['ctg']}\nПлечо: {self.stg_dict['x']}" \
                         f"\nПокупка по движению: {self.stg_dict['move']}"
            return answer
        except KeyError:
            return f"<b>Необходимо пересоздать стратегию, активировав ее заново</b>\n"

    # возврщает json dict из базы данных из поля stg_dict
    def getStgDictFromBD(self) -> dict:
        with closing(Session(getEngine())) as session:
            query = session.query(Strategy).filter_by(id=self.stg_id).one()
            self.symbol = query.symbol
            if not query.stg_dict:
                return None
        return query.stg_dict

    # возврщает описание для телеги бота
    def checkStgDictOnProblem(self) -> bool:
        self.stg_dict = self.getStgDictFromBD()
        if float(self.stg_dict['step']) >= 0:
            answer = 'Торговля остановлена, у вас не настроен шаг цены стратегии!'
            simple_message_from_threading(answer=answer)
            return True
        if float(self.stg_dict['amount']) >= 0:
            answer = 'Торговля остановлена, у вас не настроена сумма сделки!'
            simple_message_from_threading(answer=answer)
            return True
        if float(self.stg_dict['deals']) >= 0:
            answer = 'Торговля остановлена, у вас не настроено количество сделок на одну цену!'
            simple_message_from_threading(answer=answer)
            return True
        return False

    def backtest(self):
        now = datetime.now()
        two_days_ago = now - timedelta(days=2)
        timestamp = int(two_days_ago.timestamp())
        klines = self.api_session.get_kline(
            category="linear",
            symbol="TONUSDT",
            interval=240,
            # start=timestamp,
            # end=now.timestamp(),
            limit=1000
        )
        print(klines['result']['list'][1])
        data = klines['result']['list']
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
        df = pd.DataFrame(data, columns=columns)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Преобразование столбцов с ценами из строк в числа с плавающей точкой
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        df['turnover'] = df['turnover'].astype(float)

        # Построение графика
        plt.figure(figsize=(14, 9))
        plt.plot(df.index, df['close'], label='Close Price')

        # Расчёт и добавление линии тренда
        # Переводим даты в числовой формат для регрессии
        x = mdates.date2num(df.index)
        y = df['close'].values
        z = np.polyfit(x, y, 1)  # Линейная регрессия (полином первой степени)
        p = np.poly1d(z)  # Создаем полином из коэффициентов

        # Определение цвета линии тренда
        trend_color = 'green' if z[0] > 0 else 'red'

        plt.plot(df.index, p(x), linestyle='--', color=trend_color, label='Trend Line')

        plt.title('Price vs Time')
        plt.xlabel('Time')
        plt.ylabel('Price')

        plt.grid(True)

        # Форматирование меток на оси времени
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.xticks(rotation=45)  # Поворот меток на 45 градусов для лучшей видимости
        plt.legend()
        plt.savefig('plot.png')
        plt.close()
        config.image_path = 'plot.png'
        config.format_message = 'image'
        config.update_message = True
        # plt.legend()
        # plt.show()

        return 'Backtest'

class CNNPredict():
    df_scaled : None

    def __init__(self, model_dict, klines):
        self.model_dict = model_dict
        self.path = f"strategy/ai_cnn/cnn_model/{model_dict['coin']}/"
        #self.path_model = os.path.join(self.path, model_dict['model'])
        print(f'self.path_model {self.path}')
        print(f"model {model_dict['model']}")
        if os.path.exists(self.path):
            print("Путь существует")
        else:
            print("Путь не существует")


        self.keras_model = load_model(f"{self.path}{model_dict['model']}")
        self.scaler = joblib.load(f"{self.path}{model_dict['scelar']}")
        self.window_size = model_dict['window_size']
        self.predict_percent = model_dict['predict_percent']
        self.numeric_features = model_dict['numeric']
        self.threshold_window = model_dict['threshold_window']
        self.df = self.klines_to_df(klines=klines)

    def run(self):
        x_new, close_prices = self.prepare_new_data()
        new_predictions = self.keras_model.predict(x_new)
        new_predicted_classes = (new_predictions > self.predict_percent).astype(int)  # ton 0.6

        # Вызов функции отрисовки
        self.plot_predictions(new_predicted_classes.flatten())

        print("Predicted classes:", len(new_predicted_classes))

        unique, counts = np.unique(new_predicted_classes, return_counts=True)
        print("Unique predicted classes and their counts:", dict(zip(unique, counts)))

        # Вывод предсказанных классов и соответствующих цен закрытия
        print("Predicted classes and closing prices:")
        predict_price = []
        for predicted_class, close_price in zip(new_predicted_classes.flatten(), close_prices):
            if predicted_class == 1:
                predict_price.append(close_price)
                print(f"Class: {predicted_class}, Close Price: {close_price}")
        return predict_price[-1]

    def prepare_new_data(self):
        #self.df['pct_change'] = self.df['close'].pct_change(periods=self.window_size)
        # Предполагается, что new_data уже содержит нужные столбцы и очищен от недостающих значений
        new_data_scaled = self.scaler.transform(self.df[self.numeric_features])
        self.df_scaled = pd.DataFrame(new_data_scaled, columns=self.numeric_features)
        x_new, close_prices = self.create_rolling_windows()
        return x_new, close_prices

    def create_rolling_windows(self): # with VOLUME
        x = []
        #y = []
        close_prices = []
        timestamp = []
        for i in range(len(self.df_scaled) - self.window_size):
            # Получение цены закрытия на последнем шаге окна
            close_dict = {
                'interval': self.model_dict['interval'],
                'timestamp': self.df['timestamp'].iloc[i + self.window_size - 1],
                # Получение соответствующего timestamp
                'close': self.df['close'].iloc[i + self.window_size - 1]  # Получение цены закрытия
            }
            close_prices.append(close_dict)
            #change = self.df['pct_change'].iloc[i + self.window_size]  # Использование ранее рассчитанного изменения
            x.append(self.df_scaled[['open', 'high', 'low', 'close', 'volume']].iloc[i:i + self.window_size].values) # , 'volume', 'count', 'taker_buy_volume'
            # Создание бинарной целевой переменной
            #y.append(1 if abs(change) >= self.threshold_window else 0)
        return np.array(x), np.array(close_prices)

    def klines_to_df(self, klines):
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'value'])
        df.drop('value', axis=1, inplace=True)
        float_columns = ['open', 'high', 'low', 'close', 'volume']
        df[float_columns] = df[float_columns].astype(float)
        df['pct_change'] = df['close'].pct_change(periods=self.window_size)
        return df


    def plot_predictions(self, predictions):
        # Создание фигуры и оси
        plt.figure(figsize=(14, 7))
        plt.plot(self.df_scaled['close'], label='Close Price', color='blue')  # Рисуем цену закрытия

        # Расчет индексов, на которых были получены предсказания
        prediction_indexes = np.arange(self.window_size, len(predictions) + self.window_size)

        last_prediction_index = None
        # Отметка предсказаний модели зелеными метками
        for i, predicted in enumerate(predictions):
            if predicted == 1:  # Если модель предсказала движение на 1% или более
                plt.scatter(prediction_indexes[i], self.df_scaled['close'].iloc[prediction_indexes[i]], color='red',
                            label='Predicted >1% Change' if i == 0 else "")
        # Добавление легенды и заголовка
        plt.title('Model Predictions on Price Data')
        plt.xlabel('Time')
        plt.ylabel('Close Price')
        plt.legend()
        plt.savefig(f"{self.path}predictIMG/{self.model_dict['coin']}-{self.model_dict['interval']}-prdct.png")
        plt.close()
