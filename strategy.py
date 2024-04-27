import math
import time
import uuid
from contextlib import closing
from aiogram import types

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

stg_dict = {
    'ladder_stg':
        {
            'stg_name': 'ladder_stg',
            'name': 'Лесенка',
            'desc': 'Стратегия торговли лесенкой',
            'step': '0.02',
            'amount': 1,
            'deals': 1,
            'ctg': 'linear',
            'fibonachi': False,
            'x': 1,
            'exch': ''
        }}

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

# получаем класс и отдаем объект на основе этого класса
def getStgObjFromClass(stg_id: int, stg_name: str = None) -> classmethod:
    session = create_session()
    query = session.query(Strategy).filter_by(id=stg_id).one()
    if stg_id:
        if query.stg_name == 'ladder_stg':
            createStgObj = Strategy_Step(stg_id=stg_id, user_id=query.user.id)
            session.close()
            return createStgObj
    if stg_name:
        if stg_name == 'ladder_stg':
            createStgObj = Strategy_Step(stg_id=stg_id, user_id=query.user.id)
            session.close()
            return createStgObj
    return None

class Api_Trade_Method():
    api_session: None

        # https://bybit-exchange.github.io/docs/v5/intro
    def makeSession(self, stg_id: int):
        session_api = None
        session = create_session()
        api = session.query(Strategy).filter_by(id=stg_id).one()
        if api.user.api:
            for bybit in api.user.api:
                bybit_key = bybit.bybit_key
                bybit_secret = bybit.bybit_secret
            session_api = HTTP(
                testnet=False,
                api_key=bybit_key,
                api_secret=bybit_secret,
                recv_window="8000"
            )
        session.close()
        return session_api

    def getCurrentPrice(self, symbol: str):
        #self.api_session = HTTP(testnet=False)
        return self.api_session.get_tickers(
            category="spot",
            symbol=symbol,
        )

    def BuyMarket(self, symbol: str, qty: int, tp: str = None, uuid: str = None):
        try:
            return self.api_session.place_order(
                category="linear",
                symbol=symbol,
                side="Buy",
                orderType="Market",
                qty=qty,
                orderLinkId=uuid
            )
        except Exception as api_err:
            if '110007' in api_err.args[0]:
                return {'error': emoji.emojize(":ZZZ:") + " Нет денег на счету для следующей покупки", 'code': api_err.args[0]}
            return {'error': api_err, 'code': api_err.args[0]}


    def TakeProfit(self, order_dict):
        try:
           return self.api_session.set_trading_stop(
                category="linear",
                symbol="TONUSDT",
                takeProfit=str(order_dict['tp_price']),
                tpTriggerBy="MarkPrice",
                tpslMode="Partial",
                tpOrderType="Limit",
                tpSize=str(order_dict['qty']),
                tpLimitPrice=str(order_dict['tp_price']),
                positionIdx=0,
                orderLinkId=order_dict['uuid'],
                )
        except Exception as api_err:
            print(f"takeProfit={str(order_dict['tp_price'])}")
            print(f"tpSize={str(order_dict['qty'])}")
            return {'error': api_err}

    def OrderHistory(self, orderId = None):
        try:
           return self.api_session.get_order_history(
               category="linear",
               orderId=orderId,
               limit=1,
                )
        except Exception as api_err:
            return {'error': api_err}

    def getFeeRate(self, symbol: str):
        #print(f"order_dict {order_dict}\n")
        try:
            fee = self.api_session.get_fee_rates(
                symbol=symbol,
            )
            return {
                'takerFeeRate': fee['result']['list'][0]['takerFeeRate'],
                'makerFeeRate': fee['result']['list'][0]['makerFeeRate']
                }
        except Exception as api_err:
            print(f"\nGet fee EXCEPTION: {api_err.args}\n")


    def LastTakeProfitOrder(self, symbol: str, limit: int):
        try:
            return self.api_session.get_open_orders(
                category="linear",
                symbol=symbol,
                #openOnly=0,
                limit=limit,
                )
        except Exception as api_err:
            print(f"\nLastTakeProfitOrder exception: {api_err.args}\n")
        return {'error': api_err.args}

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


    def Start(self):
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
            # Вывод количества символов после точки
            dev = str(lastPrice / stepPrice).split('.')[1]
            #dev = round(float(lastPrice) % float(stepPrice),len(decimal_part))
            #print(f'dev {dev} lastprice {lastPrice} steprpice {stepPrice} | / {float(lastPrice / stepPrice)} |decimal part {decimal_part} len {len(decimal_part)}')
            if int(dev[0]) == 0:
                self.tryBuySell(lastPrice)
        else:
            print(f"answer {ddict['answer']}")
            #simple_message_from_threading(answer=ddict['answer'])

    def tryBuySell(self, lastPrice):
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
                tx = self.BuyMarket(self.symbol, self.stg_dict['amount'])
                if 'error' not in tx:
                    order_info = self.OrderHistory(tx['result']['orderId'])
                    print(f'order_info 1 {order_info}')
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
            if 'error' not in tx:
                order_info = self.OrderHistory(tx['result']['orderId'])
                print(f'order_info 2 {order_info}')
                print(f"order_info 3 {order_info['result']}")
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
                if not stg_dict['amount'] or int(stg_dict['amount']) <= 0:
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
               + f"\n/stgedit ladder_stg id={stg_id} active True - Выбрать эту стратегию\n" \
               +f"/stgedit ladder_stg id={stg_id} step 0.5 - шаг в USDT (min 20 pips)\n" \
               + f"/stgedit ladder_stg id={stg_id} deals 2 - количество сделок на одну цену\n" \
               + f"/stgedit ladder_stg id={stg_id} ctg spot - spot или linear\n" \
               + f"/stgedit ladder_stg id={stg_id} x 2 - плечо\n" \
               + f"/stgedit ladder_stg id={stg_id} fibo True - включить/выключить Фибоначчи True/False\n" \
                 f"/stgedit ladder_stg id={stg_id} amount 100 - сколько USDT за одну сделку\n"

    def getCommandValue(self, key: str, value: str) -> str:
        with closing(Session(getEngine())) as session:
            query = session.query(Strategy).filter_by(id=self.stg_id).one()
            query.stg_name = 'ladder_stg'
            if not query:
                result = f'Не создана стратегия торговли\n'
            else:
                if key == 'active':
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
                        if value == 'spot' or value == 'linear':
                            try:
                                temp_dict = {}
                                temp_dict = query.stg_dict
                                temp_dict['ctg'] = value
                                stmt = update(Strategy).where(Strategy.id == self.stg_id).values(stg_dict=temp_dict)
                                session.execute(stmt)
                                session.commit()
                                result = f'Вы указали <b>{value}</b> торговлю\n\n'
                            except ValueError:
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



                    if key == 'amount':
                        if value:
                            query.stg_name = 'ladder_stg'
                            try:
                                value = int(value)
                                temp_dict = {}
                                temp_dict = query.stg_dict
                                temp_dict['amount'] = value
                                stmt = update(Strategy).where(Strategy.id == self.stg_id).values(stg_dict=temp_dict)
                                session.execute(stmt)
                                session.commit()
                                result = f'Вы указали сумму одной сделки - <b>{value}</b>\n\n'
                            except ValueError:
                                result = f'Вы указали не правильную сумму сделки, пример: 100\n\n'
                    ddcit = query.stg_dict
                    result += f"<b>Текущие настройки</b>\nШаг цены USDT: {ddcit['step']}\nСУмма сделки в токене: {ddcit['amount']}" \
                                      f"\nКоличество сделок на одну цену: {ddcit['deals']}\nКатегория: {ddcit['ctg']}\nПлечо: {ddcit['x']}" \
                              f"\n\n<b>Описание</b>\n{ddcit['desc']}\n"
                else:
                    result = f'Сначала активируйте торговую стратегию соотвествующей командой\n'
        return result

    # возврщает описание для телеги бота
    def getDescriptionStg(self) -> str:
        try:
            answer = f"<b>Текущие настройки</b>\nШаг цены USDT: {self.stg_dict['step']}\nСумма сделки: {self.stg_dict['amount']}" \
                     f"\nКоличество сделок на одну цену: {self.stg_dict['deals']}\nКатегория: {self.stg_dict['ctg']}\nПлечо: {self.stg_dict['x']}" \
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






