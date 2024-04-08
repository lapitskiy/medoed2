import time

import emoji #https://carpedm20.github.io/emoji/
from sqlalchemy import select, Null, update
from sqlalchemy.orm import Session

from pybit.unified_trading import HTTP

from utils_db import *
from models import User, Api, Strategy

stg_dict = {
    'ladder_stg':
        {
            'name': 'Лесенка',
            'desc': 'Стратегия торговли лесенкой',
            'step': '0,0',
            'amount': 0,
            'exch': ''
        }}

def splitCommandStg(stgedit: str):
    try:
        stg, stg_id, key, value = stgedit.split(" ", maxsplit=3)
    except ValueError:
        return "Ошибка: не указаны все параметры.\n"
    if stg == 'ladder_stg':
        stg_id = stg_id.split("=", 1)[-1].strip()
        createStgObj = Strategy_Step(stg_id=stg_id)
        return createStgObj.getCommandValue(key=key, value=value)

# получаем класс и отдаем объект на основе этого класса
def getStgObjFromClass(stg_id: int) -> classmethod:
    if stg_id:
        with Session(getEngine()) as session:
            query = session.query(Strategy).filter_by(id=stg_id).one()
            if query.stg_name == 'ladder_stg':
                createStgObj = Strategy_Step(stg_id=stg_id)
                createStgObj.stg_id = stg_id
                return createStgObj
    return None

class Api_Trade_Method():
    # https://bybit-exchange.github.io/docs/v5/intro
    def getCurrentPrice(self, symbol: str):
        session = HTTP(testnet=False)
        return session.get_tickers(
            category="spot",
            symbol=symbol,
        )


class Strategy_Step(Api_Trade_Method):
    symbol: str

    def __init__(self, stg_id):
        self.stg_id = stg_id
        self.stg_dict = self.getStgDictFromBD()

    def Start(self):
        # получаем теущую цену +
        # если цена равна круглой цене в шаге который у нас указан, происходит покупка
        # тут же ставиться стоп на цену шага выше
        # запоминается время покупки в базу и стоп по этой покупке
        # если происходит покупка снова по этой цене, а старый тейкпрофит еще в базе, удаляется старый тейкпрофит и ставится новый двойной
        ticker = self.getCurrentPrice(symbol=self.symbol)
        all_price = ticker['result']['list'][0]
        lastPrice = round(float(all_price['lastPrice']),2)
        stepPrice = float(self.stg_dict['step'])
        dev = round(float(lastPrice % stepPrice),2)
        print(f'step {dev}')
        if dev == 0:
            print(f'{lastPrice} - step {stepPrice}')





    '''telegram bot func'''
    # включить или выключить стратегию торговли
    def StopStartStg(self, start: bool = None) -> dict:
        return_txt = ''
        with Session(getEngine()) as session:
            query = session.query(Strategy).filter_by(id=self.stg_id).one()
            if query.stg_dict:
                if start:
                    query.start = start
                else:
                    query.start = False if query.start else True
                stg_dict = query.stg_dict
                if not stg_dict['step'] or float(stg_dict['step']) <= 0.0:
                    return_txt = '<b>Нельзя запустить!</b>\n У вас не настроен шаг стратегии' + emoji.emojize(
                        ":backhand_index_pointing_left:") +'\n'
                    query.start = False
                if not stg_dict['amount'] or int(stg_dict['amount']) <= 0:
                    return_txt = '<b>Нельзя запустить!</b>\n У вас не настроена сумма сделки' + emoji.emojize(
                        ":backhand_index_pointing_left:") + '\n'
                    query.start = False
            else:
                return_txt = '<b>Нельзя запустить ' + emoji.emojize(
                    ":backhand_index_pointing_left:") + '</b>, у вас не указана стратегия\n'
            session.commit()
        return return_txt

    def returnHelpInfo(self) -> str:
        return f"Настройка стратегии Лесенка" + emoji.emojize(":chart_increasing:") \
               + "\nВводите команды для настройки стратегии\n" \
               + f"\nдля этой стратегии укажите <b>id={self.stg_id}</b>\n" \
               + f"\n/stgedit ladder_stg id={self.stg_id} active True - Выбрать эту стратегию\n" \
               +f"/stgedit ladder_stg id={self.stg_id} step 0.5 - шаг в USDT\n" \
                f"/stgedit ladder_stg id={self.stg_id} amount 100 - сколько USDT за одну сделку\n"

    def getCommandValue(self, key: str, value: str) -> str:
        with Session(getEngine()) as session:
            query = session.query(Strategy).filter_by(id=self.stg_id).one()
            if key == 'active':
                if value:
                    query.stg_name = 'ladder_stg'
                    result = f'<b>Стратегия Лесенка ' + emoji.emojize(":chart_increasing:") + f' активирована для {query.symbol} ({self.stg_id})</b>\nДля ее работы надо указать все настройки и после этого запустить.'
                else:
                    query.stg_name = Null
                    result = '<b>Стратегия отключена</b>'
                session.commit()

            if key == 'step':
                if value:
                    query.stg_name = 'ladder_stg'
                    try:
                        value = float(value)
                        temp_dict = {}
                        temp_dict = query.stg_dict
                        temp_dict['step'] = value
                        print(f"temp {temp_dict}")
                        stmt = update(Strategy).where(Strategy.id == self.stg_id).values(stg_dict=temp_dict)
                        session.execute(stmt)
                        session.commit()
                        print(f"commit {query.stg_dict}")
                        result = f'Вы указали шаг - <b>{value} USDT</b>\n'
                    except ValueError:
                        result = f'Вы указали не правильный шаг, пример: 0.5\n'

            if key == 'amount':
                if value:
                    query.stg_name = 'ladder_stg'
                    try:
                        value = int(value)
                        temp_dict = {}
                        temp_dict = query.stg_dict
                        temp_dict['amount'] = value
                        print(f"temp {temp_dict}")
                        stmt = update(Strategy).where(Strategy.id == self.stg_id).values(stg_dict=temp_dict)
                        session.execute(stmt)
                        session.commit()
                        print(f"commit {query.stg_dict}")
                        result = f'Вы указали сумму одной сделки - <b>{value} USDT</b>\n'
                    except ValueError:
                        result = f'Вы указали не правильную сумму сделки, пример: 100\n'

            if query.stg_dict is None:

                query.stg_dict = stg_dict['ladder_stg']
                session.commit()
                print(f"commit {query.stg_dict}")
            ddcit = query.stg_dict
            result = result + f"\nОписание: {ddcit['desc']}\nШаг цены USDT: {ddcit['step']}\nСумма сделки USDT: {ddcit['amount']}"
        return result

    # возврщает описание для телеги бота
    def getDescriptionStg(self) -> str:
        return f"\nОписание: {self.stg_dict['desc']}\nШаг цены USDT: {self.stg_dict['step']}\nСумма сделки USDT: {self.stg_dict['amount']}"

    # возврщает json dict из базы данных из поля stg_dict
    def getStgDictFromBD(self) -> dict:
        with Session(getEngine()) as session:
            query = session.query(Strategy).filter_by(id=self.stg_id).one()
            if not query.stg_dict:
                return None
            self.symbol = query.symbol
        return query.stg_dict





