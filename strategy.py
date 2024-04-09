import time
from contextlib import closing
from aiogram import types

import emoji #https://carpedm20.github.io/emoji/
from sqlalchemy import select, Null, update
from sqlalchemy.orm import Session, sessionmaker

from pybit.unified_trading import HTTP

from utils_db import getEngine
from models import Strategy


stg_dict = {
    'ladder_stg':
        {
            'stg_name': 'ladder_stg',
            'name': 'Лесенка',
            'desc': 'Стратегия торговли лесенкой',
            'step': '0.0',
            'amount': 0,
            'deals': 1,
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
def getStgObjFromClass(stg_id: int, stg_name: str = None) -> classmethod:
    if stg_id:
        session = create_session()
        query = session.query(Strategy).filter_by(id=stg_id).one()
        if query.stg_name == 'ladder_stg':
            createStgObj = Strategy_Step(stg_id=stg_id)
            createStgObj.stg_id = stg_id
            session.close()
            return createStgObj
    if stg_name:
        if stg_name == 'ladder_stg':
            createStgObj = Strategy_Step(stg_id=stg_id)
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

def create_session():
    Session = sessionmaker(getEngine())
    return Session()


class Strategy_Step(Api_Trade_Method):
    symbol: str

    def __init__(self, stg_id):
        self.stg_id = stg_id
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
            ticker = self.getCurrentPrice(symbol=self.symbol)
            all_price = ticker['result']['list'][0]
            lastPrice = round(float(all_price['lastPrice']),2)
            stepPrice = float(self.stg_dict['step'])
            dev = round(float(lastPrice % stepPrice),2)
            print(f'tyt:')
            if dev == 0:
                self.trySellBuy()

                print(f'{lastPrice} - step {stepPrice}')
        else:
            print(f"answer {ddict['answer']}")
            #simple_message_from_threading(answer=ddict['answer'])

    def tryBuySell(self):
        pass



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
                    print(f"tyt1 {query.start}")
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
            print(f"start {return_dict['start']}")
            session.commit()
        return return_dict

    def returnHelpInfo(self, stg_id: int) -> str:
        return f"Настройка стратегии Лесенка" + emoji.emojize(":chart_increasing:") \
               + "\nВводите команды для настройки стратегии\n" \
               + f"\nдля этой стратегии укажите <b>id={stg_id}</b>\n" \
               + f"\n/stgedit ladder_stg id={stg_id} active True - Выбрать эту стратегию\n" \
               +f"/stgedit ladder_stg id={stg_id} step 0.5 - шаг в USDT\n" \
               + f"/stgedit ladder_stg id={stg_id} deals 2 - количество сделок на одну цену\n" \
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
                                result = f'Вы указали шаг - <b>{value} USDT</b>\n\n'
                            except ValueError:
                                result = f'Вы указали не правильный шаг, пример: 2\n\n'

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
                                result = f'Вы указали сумму одной сделки - <b>{value} USDT</b>\n\n'
                            except ValueError:
                                result = f'Вы указали не правильную сумму сделки, пример: 100\n\n'
                    ddcit = query.stg_dict
                    result += f"<b>Текущие настройки</b>\nШаг цены USDT: {ddcit['step']}\nСумма сделки USDT: {ddcit['amount']}" \
                                      f"\nКоличество сделок на одну цену: {ddcit['deals']}\n\n<b>Описание</b>\n{ddcit['desc']}\n"
                else:
                    result = f'Сначала активируйте торговую стратегию соотвествующей командой\n'
        return result

    # возврщает описание для телеги бота
    def getDescriptionStg(self) -> str:
        try:
            answer = f"<b>Текущие настройки</b>\nШаг цены USDT: {self.stg_dict['step']}\nСумма сделки USDT: {self.stg_dict['amount']}" \
                                      f"\nКоличество сделок на одну цену: {self.stg_dict['deals']}\n\n<b>Описание</b> {self.stg_dict['desc']}\n"
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






