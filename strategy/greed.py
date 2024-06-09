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

from stg_router import Api_Trade_Method
from utils_db import getEngine
from models import Strategy, TradeHistory
from config import config

stg_dict = {
                'stg_name': 'greed_stg',
                'name': 'Greed stg',
                'desc': 'Стратегия торговли Greed',
                'ctg': 'linear',
                'exch': '',
                'decimal_part': 2,
                'tp_percent': 1,
                'move': 'two',  # up, down, two
            }

def create_session():
    Session = sessionmaker(getEngine())
    return Session()

class Strategy_Greed(Api_Trade_Method):
    symbol: str
    decimal_part: int
    uuid: str
    session: None

    def __init__(self, stg_id, user_id):
        self.stg_id = stg_id
        self.api_session = self.makeSession(stg_id=stg_id)
        self.user_id = user_id
        self.stg_name = 'greed_stg'
        self.stg_dict = self.getStgDictFromBD()
        config.getTgId(user_id=user_id)


    def Start(self):
        ddict = self.StopStartStg()
        if ddict['start'] == True:
            self.tryBuySell(lastPrice)
        else:
            print(f"answer {ddict['answer']}")
            #simple_message_from_threading(answer=ddict['answer'])

    def tryBuySell(self, lastPrice):

        # покупаю 0.2 и продаю сразу 0.2
        # если покупка выросла на 1% закрываю
        # продаю 0.4
        pass

    def createTX(self, tx: dict, tp: dict):
        print(f"tx {tx['result']}")
        tx_dict = {
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
                self.session.commit()
                config.message = emoji.emojize(":money_with_wings:") + f" Сработал TakeProfit\n"
                config.update_message = True
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
        return f"Настройка стратегии Лесенка фиббоначи" + emoji.emojize(":chart_increasing:") \
               + "\nВводите команды для настройки стратегии\n" \
               + f"\nдля этой стратегии укажите <b>id={stg_id}</b>\n" \
               + f"\n/stgedit ladder_stg id={stg_id} active True - Выбрать эту стратегию\n" \
               +f"/stgedit ladder_stg id={stg_id} step 0.5 - шаг в USDT\n" \
               + f"/stgedit ladder_stg id={stg_id} deals 2 - количество сделок на одну цену\n" \
               + f"/stgedit ladder_stg id={stg_id} ctg spot - spot или linear\n" \
               + f"/stgedit ladder_stg id={stg_id} x 2 - плечо\n" \
                 f"/stgedit ladder_stg id={stg_id} amount 10 - сколько крипты за одну сделку\n"

    def getCommandValue(self, key: str, value: str) -> str:
        with closing(Session(getEngine())) as session:
            query = session.query(Strategy).filter_by(id=self.stg_id).one()
            query.stg_name = 'ladder_stg'
            if not query:
                result = f'Не создана стратегия торговли\n'
            else:
                if key == 'active':
                    if value:
                        result = f'<b>Стратегия Лесенка фиббоначи ' + emoji.emojize(":chart_increasing:") + f' активирована для {query.symbol} ({self.stg_id})</b>\nДля ее работы надо указать все настройки и после этого запустить.\n\n'
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





