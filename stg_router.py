import asyncio
from contextlib import closing
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import matplotlib
import talib

from strategy.ai_cnn import Strategy_AI_CNN
from trade_classes import Api_Trade_Method

from strategy import ai_cnn

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression


import emoji #https://carpedm20.github.io/emoji/
from pybit import exceptions
from pybit.exceptions import InvalidRequestError
from sqlalchemy import select, Null, update
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import Session, sessionmaker

from pybit.unified_trading import HTTP #https://www.bybit.com/en/help-center/s/webform?language=en_US

from utils_db import getEngine
from models import Strategy, TradeHistory
from config import config
# http://192.168.0.21/pgadmin4/login?next=%2Fpgadmin4%2Fbrowser%2Fpip install --upgrade pip setuptools wheel

# ladder_stg
# - fibo торгует по фибоначчи, если указана только одна сторона, торгует в обратную сторону на указанный amount
# - fibo если указаны две стороны, торгует fibo в каждую сторону начиная с 1

stg_dict = {
    'ladder_stg':
        {
            'stg_name': 'ladder_stg',
            'name': 'Лесенка',
            'desc': 'Стратегия торговли лесенкой',
            'step': '0.02',
            'amount': '1.0',
            'deals': 1,
            'ctg': 'linear',
            'fibo': False,
            'move': 'two', # up, down, two
            'x': 1,
            'exch': ''
        },
    'ai_cnn': ai_cnn.stg_dict
}

def create_session():
    Session = sessionmaker(getEngine())
    return Session()

def splitCommandStg(stgedit: str):
    try:
        stg, stg_id, key, value = stgedit.split(" ", maxsplit=3)
        stg_id = stg_id.split("=", 1)[-1].strip()
    except ValueError:
        return "Ошибка: не указаны все параметры.\n"
    stgObj = getStgObjFromClass(stg_id=stg_id, stg_name=stg)
    return stgObj.getCommandValue(key=key, value=value)

def getStgObjFromClass(stg_id: int = None, stg_name: str = None) -> classmethod:
    session = create_session()
    query = session.query(Strategy).filter_by(id=stg_id).one()
    # вызывается класс с конкретной стратегией
    if stg_id and stg_name is None:
        if query.stg_name == 'ladder_stg':
            createStgObj = Strategy_Step(stg_id=stg_id, user_id=query.user.id)
            session.close()
            return createStgObj
        if query.stg_name == 'ai_cnn':
            createStgObj = Strategy_AI_CNN(stg_id=stg_id, user_id=query.user.id)
            session.close()
            return createStgObj
    if stg_name:
        if stg_name == 'ladder_stg':
            createStgObj = Strategy_Step(stg_id=stg_id, user_id=query.user.id)
            session.close()
            return createStgObj
        if stg_name == 'ai_cnn':
            createStgObj = Strategy_AI_CNN(stg_id=stg_id, user_id=query.user.id)
            session.close()
            return createStgObj
    return None

class Strategy_Step(Api_Trade_Method):
    symbol: str
    decimal_part: int
    uuid: str
    session: None

    def __init__(self, stg_id, user_id):
        self.stg_id = stg_id
        self.api_session = self.makeSession(stg_id=stg_id)
        self.user_id = user_id
        self.stg_name = 'ladder_stg'
        self.stg_dict = self.getStgDictFromBD()
        config.getTgId(user_id=user_id)
        self.fee = self.getFeeRate(symbol=self.symbol)


    async def Start(self):
        # получаем теущую цену +
        # если цена равна круглой цене в шаге который у нас указан, происходит покупка
        # тут же ставиться стоп на цену шага выше
        # запоминается время покупки в базу и стоп по этой покупке
        # если происходит покупка снова по этой цене, а старый тейкпрофит еще в базе, удаляется старый тейкпрофит и ставится новый двойной
        ddict = self.StopStartStg()
        if ddict['start'] == True:
            self.cleanHistory()
            decimal_part = str(self.stg_dict['step'])
            decimal_part = decimal_part.split('.')[1]
            self.decimal_part = len(decimal_part)
            ticker = self.getCurrentPrice(symbol=self.symbol)
            all_price = ticker['result']['list'][0]
            #print(f"lastprice {all_price['lastPrice']}")
            lastPrice = round(float(all_price['lastPrice']), self.decimal_part)
            stepPrice = float(self.stg_dict['step'])
            print(f'lastPrice {lastPrice} - stepPrice {stepPrice}')
            # Вывод количества символов после точки
            dev = str(lastPrice / stepPrice).split('.')[1]
            #dev = round(float(lastPrice) % float(stepPrice),len(decimal_part))
            #print(f'dev {dev} lastprice {lastPrice} steprpice {stepPrice} | / {float(lastPrice / stepPrice)} |decimal part {decimal_part} len {len(decimal_part)}')
            if int(dev[0]) == 0:
                print(dev[0])
                await self.tryBuy(lastPrice)
        else:
            print(f"answer {ddict['answer']}")
            #simple_message_from_threading(answer=ddict['answer'])

    async def tryBuy(self, lastPrice):
        #self.uuid = str(uuid.uuid4())
        self.stg_dict = self.getStgDictFromBD()
        tp = round(lastPrice + float(self.stg_dict['step']), self.decimal_part)
        self.session = create_session()
        tradeQ = self.session.query(TradeHistory).order_by(TradeHistory.id.desc())
        priceCountQ = self.session.query(TradeHistory).filter_by(price=str(lastPrice), filled=False)
        if priceCountQ.first():
            lastTX = tradeQ.first()
            if int(self.stg_dict['deals']) >= priceCountQ.count() and lastTX.price == str(lastPrice):
                #print(f'IF {lastPrice} = {lastTX.price}')
                #print(f"deals {self.stg_dict['deals']} >= {priceCountQ.count()}")
                pass
            else:
                #print(f'ELSE {lastPrice} | {lastTX.price}')
                if (self.stg_dict['move'] == 'down' and float(lastPrice) < float(lastTX.price)) or (self.stg_dict['move'] == 'up' and float(lastPrice) > float(lastTX.price)) or self.stg_dict['move'] == 'two':
                    print(f"price COUNT 2 {priceCountQ.count()} | deals = {self.stg_dict['deals']}\n")
                    if self.stg_dict['fibo']:
                        #fibo_next_num
                        fibo_amount = 1
                        tx = self.BuyMarket(self.symbol, fibo_amount)
                    else:
                        tx = self.BuyMarket(self.symbol, self.stg_dict['amount'])
                    if 'error' not in tx:
                        order_info = await self.OrderHistory(tx['result']['orderId'])
                        #print(f'order_info 1 {order_info}')
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
                            config.message = emoji.emojize(":check_mark_button:") + f" Куплено {self.stg_dict['amount']} {self.symbol} повторно по {buy_price} [{self.stg_dict['name']}]"

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
            #print(f'\nlastTX {lastTX.price}')
            print(f"else price COUNT - {priceCountQ.count()} | lastprice = {lastPrice}  |  | deals = {self.stg_dict['deals']}\n")
            if 'error' not in tx:
                #print(f"orderid {tx}\n")
                order_info = await self.OrderHistory(tx['result']['orderId'])
                #print(f'order_info 2 {order_info}\n')
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
                #print(f'tp {tp}')
                if tp is not None and isinstance(tp, dict) and 'error' not in tp:
                    last_tp = self.LastTakeProfitOrder(symbol=self.symbol, limit=1)
                    #print(f"tx buy {tx['result']}")
                    tx['result']['price'] = round(float(buy_price), self.decimal_part)
                    tx['result']['tpOrderId'] = last_tp['result']['list'][0]['orderId']
                    tx_obj = self.createTX(tx=tx, tp=last_tp)
                    #print(f"tp else {tp}")
                    config.message = emoji.emojize(":check_mark_button:") + f" Куплено {self.stg_dict['amount']} {self.symbol} по {buy_price} [{self.stg_dict['name']}]"
                else:
                    config.message = emoji.emojize(
                        ":check_mark_button:") + f" Куплено {self.stg_dict['amount']} {self.symbol} по {buy_price} [{self.stg_dict['name']}]" \
                                                 f"\nTakeProfit не был установлен по причине: {tp}"
                config.update_message = True
            else:
                config.message = tx['error']
                config.update_message = True
        #- если купили ставим стоп на шаг выше
        #- если это вторая покупка по цене, отменяем первый стоп и и ставим новый умноженный на 2
        self.session.close()

    def createTX(self, tx: dict, tp: dict):
        #print(f"tx {tx['result']}")
        #print(f"tp {tp['result']}")
        tx_dict = {
            'price_clean': tp['result']['list'][0]['lastPriceOnCreated'],
            'tp': tp['result']['list'][0]['price'],
            'side': tp['result']['list'][0]['side'],
            'qty': tp['result']['list'][0]['qty']
        }
        createTx = TradeHistory(price=tx['result']['price'], tx_id=tx['result']['orderId'], tx_dict=tx_dict, stg_id=self.stg_id, user_id=self.user_id,
                                tp_id=tx['result']['tpOrderId']
                                )
        self.session.add(createTx)
        self.session.commit()
        return createTx

    def cleanHistory(self):
        self.session = create_session()
        tp = self.LastTakeProfitOrder(symbol=self.symbol, limit=50)
        historyQ = self.session.query(TradeHistory).filter_by(filled=False)
        #print(f'tp cleanHistory {tp} - {self.symbol}')
        for tx in historyQ:
            if not any(tx.tp_id in d.values() for d in tp['result']['list']):
                tx.filled = True
                tx_dict = tx.tx_dict
                fee = None
                earn = None
                percent = None
                try:
                    fee = round(((float(tx_dict['price_clean']) * float(self.fee['takerFeeRate'])) + (float(tx_dict['tp']) * float(self.fee['makerFeeRate']))) * int(tx_dict['qty']), 3)
                    earn = round(((float(tx_dict['tp']) - float(tx_dict['price_clean'])) * int(tx_dict['qty'])) - fee, 3)
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
    def StopStartStg(self, change: bool = None) -> dict:
        return_dict = {}
        return_dict['answer'] = ''
        with closing(Session(getEngine())) as session:
            query = session.query(Strategy).filter_by(id=self.stg_id).one()
            if query.stg_dict:
                if change is not None:
                    query.start = False if query.start else True
                stg_dict = query.stg_dict
                step = str(stg_dict['step'])
                if not stg_dict['step'] or float(step.replace(",", ".")) <= 0.0:
                    return_dict['answer'] = '<b>Нельзя запустить!</b>\n У вас не настроен шаг стратегии' + emoji.emojize(
                        ":backhand_index_pointing_left:") +'\n'
                    query.start = False
                if not stg_dict['amount'] or float(step.replace(",", ".")) <= 0.0:
                    return_dict['answer'] = '<b>Нельзя запустить!</b>\n У вас не настроена сумма сделки' + emoji.emojize(
                        ":backhand_index_pointing_left:") + '\n'
                    query.start = False
                if not stg_dict['deals'] or int(stg_dict['deals']) <= 0:
                    return_dict['answer'] = '<b>Нельзя запустить!</b>\n У вас не настроено количество сделок на одну цену' + emoji.emojize(
                        ":backhand_index_pointing_left:") + '\n'
                    query.start = False
            else:
                return_dict['answer'] = '<b>Нельзя запустить ' + emoji.emojize(
                    ":backhand_index_pointing_left:") + '</b>, у вас не указана стратегия\n'
            return_dict['start'] = query.start
            #print(f"start {return_dict['start']}")
            session.commit()
        return return_dict

    def returnHelpInfo(self, stg_id: int) -> str:
        return f"Настройка стратегии Лесенка" + emoji.emojize(":chart_increasing:") \
               + "\nВводите команды для настройки стратегии\n" \
               + f"\nдля этой стратегии укажите <b>id={stg_id}</b>\n" \
               + f"\n/stgedit ladder_stg id={stg_id} active true - Выбрать эту стратегию\n" \
               +f"/stgedit ladder_stg id={stg_id} step 0.5 - шаг в USDT (min 20 pips)\n" \
               + f"/stgedit ladder_stg id={stg_id} deals 2 - количество сделок на одну цену\n" \
               + f"/stgedit ladder_stg id={stg_id} ctg spot - spot или linear\n" \
               + f"/stgedit ladder_stg id={stg_id} x 2 - плечо\n" \
               + f"/stgedit ladder_stg id={stg_id} fibo true - включить/выключить Фибоначчи true/false\n" \
               + f"/stgedit ladder_stg id={stg_id} move up - Направление движения покупки up, down, two\n" \
                 f"/stgedit ladder_stg id={stg_id} amount 100 - сколько крипты за одну сделку\n"

    def getCommandValue(self, key: str, value: str) -> str:
        with closing(Session(getEngine())) as session:
            query = session.query(Strategy).filter_by(id=self.stg_id).one()
            query.stg_name = 'ladder_stg'
            if not query:
                result = f'Не создана стратегия торговли\n'
            else:
                if key.lower() == 'active':
                    if value:
                        result = f'<b>Стратегия Лесенка ' + emoji.emojize(":chart_increasing:") + f' активирована для {query.symbol} ({self.stg_id})</b>\nДля ее работы надо указать все настройки и после этого запустить.\n\n'
                        query.stg_dict = stg_dict['ladder_stg']
                        result += 'Вы обновили статегию, поэтому настройки выставлены <u>по умолчанию</u>, не забудьте их изменить\n\n'

                    else:
                        query.stg_name = Null
                        result = '<b>Стратегия отключена</b>'
                    session.commit()
                if query.stg_dict:
                    if key == 'step':
                        if value:
                            try:
                                value = float(value)
                                temp_dict = {}
                                temp_dict = query.stg_dict
                                temp_dict['step'] = value
                                stmt = update(Strategy).where(Strategy.id == self.stg_id).values(stg_dict=temp_dict)
                                session.execute(stmt)
                                session.commit()
                                result = f'Вы указали шаг - <b>{value} USDT</b>\n\n'
                            except ValueError:
                                result = f'Вы указали не правильный шаг, пример: 0.5\n\n'

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

                    if key == 'x':
                        if value:
                            try:
                                x = int(value)
                                temp_dict = {}
                                temp_dict = query.stg_dict
                                temp_dict['x'] = value
                                stmt = update(Strategy).where(Strategy.id == self.stg_id).values(stg_dict=temp_dict)
                                session.execute(stmt)
                                session.commit()
                                result = f'Вы указали плечо <b>{value}</b> для торговли\n\n'
                            except ValueError:
                                result = f'Вы указали не правильное плечо для торговли\n\n'

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

                    if key == 'fibo':
                        if value.lower() == 'false' or value.lower() == 'true':
                            temp_dict = query.stg_dict
                            temp_dict['fibo'] = bool(value)
                            stmt = update(Strategy).where(Strategy.id == self.stg_id).values(stg_dict=temp_dict)
                            session.execute(stmt)
                            session.commit()
                            result = f'Вы указали значение <b>{value}</b> для торговли по Фиббоначи\n\n'
                        else:
                             result = f'Вы указали не правильное значение для торговли Фиббоначи\n\n'

                    if key == 'amount':
                        if value:
                            query.stg_name = 'ladder_stg'
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
            answer = f"<b>Текущие настройки</b>\nШаг цены USDT: {self.stg_dict['step']}\nСумма сделки: {self.stg_dict['amount']}" \
                     f"\nКоличество сделок на одну цену: {self.stg_dict['deals']}\nКатегория: {self.stg_dict['ctg']}\nПлечо: {self.stg_dict['x']}" \
                     f"\nФибоначчи: {self.stg_dict['fibo']}\nПокупка по движению: {self.stg_dict['move']}" \
                     f"\n\n<b>Описание</b> {self.stg_dict['desc']}\n"
            return answer
        except KeyError:
            return f"<b>Необходимо пересоздать стратегию, активировав ее заново</b>\n"

    # возврщает json dict из базы данных из поля stg_dict
    def getStgDictFromBD(self) -> dict:
        with closing(Session(getEngine())) as session:
            query = session.query(Strategy).filter_by(id=self.stg_id).one()
            if not query.stg_dict:
                return None
            self.symbol = query.symbol
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

    def fibo_next_num(self, n: int) -> int:
        # Возвращает список первых n чисел Фибоначчи
        if n <= 0:
            return 1
        feb_prev = 1
        feb_curr = 1
        feb_next = 1
        while feb_next <= n: # 13
            feb_next = feb_prev + feb_curr
            feb_prev = feb_curr
            feb_curr = feb_next
        return feb_next

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

    def backtest_Bollinger_Bands(self):
        print('tyt')
        now = datetime.now()
        two_days_ago = now - timedelta(days=2)
        timestamp = int(two_days_ago.timestamp())
        klines = self.api_session.get_kline(
            category="linear",
            symbol="TONUSDT",
            interval=1,
            #start=timestamp,
            #end=now.timestamp(),
            limit=1000

        )
        print(klines['result']['list'][1])
        data = klines['result']['list']
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
        df = pd.DataFrame(data, columns=columns)
        df['close'] = pd.to_numeric(df['close'])

        # Рассчитываем Bollinger Bands
        upperband, middleband, lowerband = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df['upper_band'] = upperband
        df['lower_band'] = lowerband
        print(f'upperband {upperband}\n')
        print(f'middleband {middleband}\n')
        print(f'lowerband {lowerband}\n')

        # Торговая стратегия
        buys = []
        sells = []
        positions = False
        profit = 0

        for i in range(1, len(df)):
            if df['close'].iloc[i] < df['lower_band'].iloc[i] and not positions:
                # Покупаем, если цена упала ниже нижней границы и у нас нет открытых позиций
                buys.append((i, df['close'].iloc[i]))
                last_buy_price = df['close'].iloc[i]
                positions = True
            elif df['close'].iloc[i] > df['upper_band'].iloc[i] and positions:
                # Продаём, если цена поднялась выше верхней границы и у нас есть открытая покупка
                sells.append((i, df['close'].iloc[i]))
                profit += (df['close'].iloc[i] - last_buy_price)
                positions = False

        # Визуализация цен и Bollinger Bands
        plt.figure(figsize=(14, 7))
        plt.plot(df['close'], label='Close Price', color='gray')
        plt.plot(df['upper_band'], label='Upper Band', linestyle='--', color='red')
        plt.plot(df['lower_band'], label='Lower Band', linestyle='--', color='green')
        plt.scatter(*zip(*buys), color='green', label='Buy Signal', marker='^', s=100)
        plt.scatter(*zip(*sells), color='red', label='Sell Signal', marker='v', s=100)
        plt.title('Price Chart with Bollinger Bands and Trading Signals')
        plt.xlabel('Time')
        plt.ylabel('Price')
        # Добавление текста с статистикой
        textstr = f'Number of Buys: {len(buys)}\nNumber of Sells: {len(sells)}\nTotal Profit: {profit:.2f}'
        plt.gcf().text(0.02, 0.02, textstr, fontsize=12)  # Использование координат фигуры для размещения текста
        '''
        # Инициализация переменных
        last_buy_price = None
        profit = 0
        trades = []
        buy_signals = []
        sell_signals = []

        # Параметры стратегии
        price_movement = 0.02

        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        df['turnover'] = df['turnover'].astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        # Установка timestamp как индекса DataFrame
        df.set_index('timestamp', inplace=True)

        # Проходим по всем строкам данных
        for index, row in df.iterrows():
            current_price = row['close']

            # Проверяем условия для покупки
            if last_buy_price is None or (current_price >= last_buy_price + price_movement) or (
                    current_price <= last_buy_price - price_movement):
                # Регистрируем покупку
                last_buy_price = current_price
                trades.append(('buy', current_price, index))  # добавляем индекс для визуализации
                buy_signals.append((index, current_price))  # сохраняем индексы и цены покупок для графика

            # Проверяем условие для тейк-профита
            for trade in trades:
                if trade[0] == 'buy' and current_price >= trade[1] + price_movement:
                    profit += (current_price - trade[1] - price_movement)  # учитываем цену покупки и продажи
                    sell_signals.append((index, current_price))  # добавляем момент продажи
                    trades.remove(trade)  # закрываем позицию

        # Построение графика
        plt.figure(figsize=(14, 7))
        plt.plot(df['close'], label='Close Price', color='gray')
        plt.scatter(*zip(*buy_signals), color='green', label='Buy Signal', marker='^', s=100)
        plt.scatter(*zip(*sell_signals), color='red', label='Sell Signal', marker='v', s=100)
        plt.title('Price Chart with Buy and Sell Signals')
        plt.xlabel('Time')
        plt.ylabel('Price')


        # Добавление текста с статистикой
        textstr = f'Number of Buys: {len(buy_signals)}\nNumber of Sells: {len(sell_signals)}\nProfit: {profit:.2f}'
        #plt.figtext(0.15, -0.1, textstr, wrap=True, horizontalalignment='left', fontsize=12)
        plt.gcf().text(0.02, 0.02, textstr, fontsize=12)
        
        '''
        plt.legend()
        plt.savefig('plot.png')
        plt.close()
        config.image_path = 'plot.png'
        config.format_message = 'image'
        config.update_message = True
        #plt.legend()
        #plt.show()

        return 'Backtest'







