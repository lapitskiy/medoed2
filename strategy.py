import math
import time
import uuid
from contextlib import closing
from aiogram import types

import emoji #https://carpedm20.github.io/emoji/
from pybit.exceptions import InvalidRequestError
from sqlalchemy import select, Null, update
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import Session, sessionmaker

from pybit.unified_trading import HTTP #https://www.bybit.com/en/help-center/s/webform?language=en_US

from utils_db import getEngine
from models import Strategy, TradeHistory
from config import config

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
        session = create_session()
        api = session.query(Strategy).filter_by(id=stg_id).one()
        for bybit in api.user.api:
            bybit_key = bybit.bybit_key
            bybit_secret = bybit.bybit_secret
        session_api = HTTP(
            testnet=False,
            api_key=bybit_key,
            api_secret=bybit_secret,
            recv_window="8000"
        )
        return session_api

    def getCurrentPrice(self, symbol: str):
        #self.api_session = HTTP(testnet=False)
        return self.api_session.get_tickers(
            category="spot",
            symbol=symbol,
        )

    def BuyMarket(self, symbol: str, qty: int, tp: str, uuid: str = None):
        return self.api_session.place_order(
            category="linear",
            symbol=symbol,
            side="Buy",
            orderType="Market",
            qty=qty,
            orderLinkId=uuid
        )

    def TakeProfit(self, order_dict):
        #print(f"order_dict {order_dict}\n")
        try:
            self.api_session.set_trading_stop(
                category="linear",
                symbol="TONUSDT",
                takeProfit=str(order_dict['tp_price']),
                tpTriggerBy="MarkPrice",
                tpslMode="Partial",
                tpOrderType="Limit",
                tpSize="2",
                tpLimitPrice=str(order_dict['tp_price']),
                positionIdx=0,
                orderLinkId=order_dict['uuid'],
                )
        except InvalidRequestError as e:
            print(f"tp exc {e}\n")


    '''
    def BuyMarket(self, symbol: str, qty: int, tp: str, uuid: str):
        return self.api_session.place_order(
            category="linear",
            symbol=symbol,
            side="Buy",
            orderType="Market",
            qty=qty,
            takeProfit=tp,
            tpslMode='Partial',
            orderLinkId=uuid
        )
    '''

    def LastTakeProfitOrder(self, symbol: str, limit: int):
        return self.api_session.get_open_orders(
            category="linear",
            symbol=symbol,
            #openOnly=0,
            limit=limit,
            )

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


    def Start(self):
        # получаем теущую цену +
        # если цена равна круглой цене в шаге который у нас указан, происходит покупка
        # тут же ставиться стоп на цену шага выше
        # запоминается время покупки в базу и стоп по этой покупке
        # если происходит покупка снова по этой цене, а старый тейкпрофит еще в базе, удаляется старый тейкпрофит и ставится новый двойной
        ddict = self.StopStartStg()
        if ddict['start'] == True:
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
        time.sleep(5)
        tp = round(lastPrice + float(self.stg_dict['step']), self.decimal_part)
        self.session = create_session()
        self.cleanHistory()
        tradeQ = self.session.query(TradeHistory).order_by(TradeHistory.id.desc()).filter_by(price=str(lastPrice))
        if tradeQ.first():
            lastTX = tradeQ.first()
            if tradeQ and self.stg_dict['deals'] >= tradeQ.count() or lastTX.price == str(lastPrice):
                pass
            else:
                print(f"tx if 2 deals = {tx}")
                tx = self.BuyMarket(tradeQ.stg.symbol, stg_dict['amount'], tp=tp)
                tx['result']['price'] = lastPrice
                tx_obj = self.createTX(tx=tx, session=session)
                print(f'[{lastTX.id}] покупка по цене {lastPrice} {tx}')
                session.commit()
                #self.TakeProfit(symbol, tx)
                config.message = f"[" + emoji.emojize(":check_mark_button:")+ f"] Куплено {lastPrice}"
                config.update_message = True
        else:
            tx = self.BuyMarket(self.symbol, self.stg_dict['amount'], tp=tp)
            print(f"tx else {tx}")
            order_dict = {
                'ctg': self.stg_dict['ctg'],
                'side': 'Sell',
                'symbol': self.symbol,
                'orderType': 'Market',
                'tp_price': round(float(lastPrice) + float(self.stg_dict['step']),self.decimal_part),
                'qty': self.stg_dict['amount'],
                'uuid': tx['result']['orderId']
            }
            self.TakeProfit(order_dict=order_dict)
            tp = self.LastTakeProfitOrder(symbol=self.symbol, limit=1)
            tx['result']['price'] = lastPrice
            tx['result']['tpOrderId'] = tp['result']['list'][0]['orderId']
            tx_obj = self.createTX(tx=tx)
            print(f"tp else {tp}")
            config.message = f"[" + emoji.emojize(":check_mark_button:")+ f"] Куплено {lastPrice}"
            config.update_message = True
        #- если купили ставим стоп на шаг выше
        #- если это вторая покупка по цене, отменяем первый стоп и и ставим новый умноженный на 2
        self.session.close()

    def createTX(self, tx: dict):
        #print(f"tx {tx['result']['orderId']}")
        createTx = TradeHistory(price=tx['result']['price'], tx_id=tx['result']['orderId'], tx_dict=tx['result'], stg_id=self.stg_id, user_id=self.user_id, tp_id=tx['result']['tpOrderId'])
        self.session.add(createTx)
        self.session.commit()
        return createTx

    def cleanHistory(self):
        tp = self.LastTakeProfitOrder(symbol=self.symbol, limit=50)
        historyQ = self.session.query(TradeHistory).all()
        for tx in historyQ:
            if not any(tx.tp_id in d.values() for d in tp['result']['list']):
                self.session.delete(tx)  # Удаляем пользователя
                self.session.commit()

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
               +f"/stgedit ladder_stg id={stg_id} step 0.5 - шаг в USDT\n" \
               + f"/stgedit ladder_stg id={stg_id} deals 2 - количество сделок на одну цену\n" \
               + f"/stgedit ladder_stg id={stg_id} ctg spot - spot или linear\n" \
               + f"/stgedit ladder_stg id={stg_id} x 2 - плечо\n" \
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







