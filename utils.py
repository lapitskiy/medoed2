from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, select, insert
from sqlalchemy import text

from pybit.unified_trading import HTTP
from pybit.exceptions import InvalidRequestError
from sqlalchemy.orm import selectinload, Session

from utils_db import *
from models import User, Api, Strategy

from config import config


def dbAccCheck(id):
    ddict = {}

    print(f"id {id}")
    print(f"user {config.user_db.get_secret_value()}")
    print(f"pass {config.pass_db.get_secret_value()}")
    print(f"database_name {config.name_db.get_secret_value()}")
    print(f"host {config.host_db.get_secret_value()}")

    with Session(getEngine()) as session:
        statement = select(User).filter_by(user=id)
        user_obj = session.scalars(statement).all()
        #query = session.query(User).filter_by(user=id)
        print(f" query {user_obj}")
        if not user_obj:
            print(f" добавляем акк")
            # Вставляем новую запись в таблицу
            # Создаем объект MetaData
            session.add(User(user=id))
            session.commit()
            ddict = {'answer': 'Ваш аккаунт создан'}
        else:
        # Печатаем результат запроса
            ddict = {'answer': 'Ваш аккаунт уже активирован'}
    return ddict

def dbCheck():
    connection = dbConnect()
    print(f"exc {type(connection)}")
    if type(connection) == tuple:
        return connection

    connection.close()
    return 'Соедение с базой успешное, целостность базы не нарушена'

def AddCoin2(exchange, coin, user_id):
    ddict = {}
    connection = dbConnect()
    current_user = connection.execute(select(user).where(user.c.user == user_id)).first()
    current_user_api = connection.execute(select(api).where(api.c.id == current_user.api)).first()
    if not current_user:
        ddict = {'answer': 'пользовате не создан, запустите команду start'}
    else:
        if current_user.api is None:
            ddict = {'answer': 'нет api записи, укажите его в настройках'}
        else:
            session = HTTP(
                testnet=False,
                api_key=current_user_api.bybit_key,
                api_secret=current_user_api.bybit_secret,
            )
            try:
                get_coin = session.get_instruments_info(category='spot', symbol=coin,)
                print(f"instr {get_coin}")
                if not get_coin['result']['list']:
                    ddict['answer'] = f'Пары нет в споте ByBit, укажите правильно название пары (пример: BTCUSDT)\nМонета не добавлена в базу'
                else:
                    if get_coin['result']['list'][0]['symbol'] == coin:

                        ddict['answer'] = f'Пара есть в споте ByBit\nМонета добавлена в базу бота'
            except InvalidRequestError as e:
                return {'answer': e}
    return ddict

def AddCoin(exchange, coin, id):
    ddict = {'answer': ''}
    with Session(getEngine()) as session:
        query = session.query(User).options(selectinload(User.api)).options(selectinload(User.stg)).filter_by(user=id).one()
        if not query.api:
            ddict['answer'] = ddict['answer'] + f'<b>Нет api, добавьте его в настройках</b>'
        else:
            for key in query.api:
                ddict['answer'] = ddict['answer'] + f'ByBit api: {key.bybit_key}\n'
                exchange_session = HTTP(
                    testnet=False,
                    api_key=key.bybit_key,
                    api_secret=key.bybit_secret
                )
                try:
                    get_coin = exchange_session.get_instruments_info(category='spot', symbol=coin, )
                    print(f"instr {get_coin}")
                    if not get_coin['result']['list']:
                        ddict['answer'] = ddict['answer'] + f'<b>Монета не добавлена в базу</b>\nПары нет в споте ByBit, укажите правильно название (пример: BTCUSDT)\n'
                    else:
                        if get_coin['result']['list'][0]['symbol'] == coin:
                            query.stg = [Strategy(symbol=coin, limit=0, start=False)]
                            session.add(query)
                            session.commit()
                            ddict['answer'] = f'Пара есть в споте ByBit\n<b>{coin} добавлен в базу бота</b>\n\nДля запуска торговли отредактируйте стратегию этой пары'
                except InvalidRequestError as e:
                    return {'answer': e}
    return ddict

def AddApiByBit(api_key, api_secret, id):
    ddict = {}
    session = HTTP(
        testnet=False,
        api_key=api_key,
        api_secret=api_secret,
    )
    try:
        ddict['result'] = session.get_wallet_balance(
            accountType="UNIFIED",
            coin="BTC",
        )
        #print(f"res {ddict['result']}")
        if ddict['result']['retMsg'] == 'OK' and ddict['result']['retCode'] == 0:
            print(f"ok")
            with Session(getEngine()) as session:
                query = session.query(User).options(selectinload(User.api)).filter_by(user=id).one()
                if not query.api:
                    print(f"query {query.user}")
                    print(f"api_secret {api_secret}")
                    query.api = [Api(bybit_secret=api_secret, bybit_key=api_key)]
                    session.add(query)
                    session.commit()
                    ddict = {'answer': f'Api bybit для вашего логина добавлен'}
                else:
                    for key in query.api:
                        if key.bybit_key == api_key:
                            ddict = {'answer': f'Такой Api bybit уже установлен'}
    except InvalidRequestError as e:
        return {'answer': e}
    return ddict

def CheckApiEx(user_id, company=None):
    ddict = {'answer' : '<b>Установленные api</b>\n'}
    with Session(getEngine()) as session:
        query = session.query(User).options(selectinload(User.api)).filter_by(user=user_id).one()
    if not query.api:
        ddict['answer'] = ddict['answer'] + f'Сейчас у вас нет добавленных API\n'
    else:
        for key in query.api:
            ddict['answer'] = ddict['answer'] + f'ByBit api: {key.bybit_key}\n'
    return ddict

def CheckExistCoin(user_id):
    ddict = {'answer' : '<b>Торгуемые пары:</b>\n'}
    with Session(getEngine()) as session:
        query = session.query(User).options(selectinload(User.stg)).filter_by(user=user_id).one()
    if not query.stg:
        ddict['answer'] = ddict['answer'] + f'У вас не добавлены пары для торговли\n'
    else:
        for key in query.stg:
            ddict['answer'] = ddict['answer'] + f'{key.symbol}\n'
    return ddict
