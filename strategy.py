import emoji #https://carpedm20.github.io/emoji/
from sqlalchemy import select, Null, update
from sqlalchemy.orm import Session

from utils_db import *
from models import User, Api, Strategy

stg_dict = {
    'ladder_stg':
        {
            'name': 'Лесенка',
            'desc': 'Стратегия торговли лесенкой',
            'step': '0,0',
            'amount': 0
        }}

def runStg(stg_name: str, stg_id: int):
    if stg_name == 'ladder_stg':
        createStg = Strategy_Step()
        return createStg.returnHelpInfo(stg_id=stg_id)

def splitCommandStg(stgedit: str):
    ddict = {'answer' : ''}
    try:
        stg, stg_id, key, value = stgedit.split(" ", maxsplit=3)
    except ValueError:
        ddict['answer'] = "Ошибка: не указаны все параметры.\n"
        return ddict
    if stg == 'ladder_stg':
        ddict['answer'] = Strategy_Step.getCommandValue(key=key, value=value, stg_id=stg_id)
        return ddict


class Strategy_Step():
    step: float

    def returnStgDict(self, step: str) -> dict:
        self.step = float(step)
        ddict = {
            'step': step,
        }
        return ddict

    def returnHelpInfo(self, stg_id: int) -> str:
        return f"Настройка стратегии Лесенка" + emoji.emojize(":chart_increasing:") \
               + "\nВводите команды для настройки стратегии\n" \
               + f"\nдля этой стратегии укажите <b>id={stg_id}</b>\n" \
               + f"\n/stgedit ladder_stg id={stg_id} active True - Выбрать эту стратегию\n" \
               +f"/stgedit ladder_stg id={stg_id} step 0.5 - шаг в USDT\n" \
                f"/stgedit ladder_stg id={stg_id} amount 100 - сколько USDT за одну сделку\n"

    @staticmethod
    def getCommandValue(key: str, value: str, stg_id: int) -> str:
        stg_id = stg_id.split("=", 1)[-1].strip()
        print(f"stg_id {stg_id}")
        with Session(getEngine()) as session:
            query = session.query(Strategy).filter_by(id=stg_id).one()
            if key == 'active':
                if value:
                    query.stg_name = 'ladder_stg'
                    result = f'<b>Стратегия Лесенка ' + emoji.emojize(":chart_increasing:") + f' активирована для {query.symbol} ({stg_id})</b>\nДля ее работы надо указать все настройки и после этого запустить.'
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
                        #query.stg_dict = None
                        #query.stg_dict = temp_dict
                        stmt = update(Strategy).where(Strategy.id == stg_id).values(stg_dict=temp_dict)
                        session.execute(stmt)
                        # Вызываем session.commit(), чтобы зафиксировать изменения в базе данных
                        session.commit()
                        print(f"commit {query.stg_dict}")
                        result = f'Вы указали шаг - <b>{value} USDT</b>\n'
                    except ValueError:
                        result = f'Вы указали не правильный шаг, пример: 0.5\n'

            if query.stg_dict is None:

                query.stg_dict = stg_dict['ladder_stg']
                session.commit()
                print(f"commit {query.stg_dict}")
            ddcit = query.stg_dict
            result = result + f"\nОписание: {ddcit['desc']}\nШаг цены USDT: {ddcit['step']}\nСумма сделки USDT: {ddcit['amount']}"
        return result




