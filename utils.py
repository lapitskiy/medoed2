from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, select, insert
from sqlalchemy import text

from pybit.unified_trading import HTTP
from pybit.exceptions import InvalidRequestError
from utils_db import *
from models import api, user


def dbAccCheck(id):
    ddict = {}

    print(f"id {id}")
    print(f"user {config.user_db.get_secret_value()}")
    print(f"pass {config.pass_db.get_secret_value()}")
    print(f"database_name {config.name_db.get_secret_value()}")
    print(f"host {config.host_db.get_secret_value()}")

    connection = dbConnect()
    query = select(user).where(user.c.user == id)
    result = connection.execute(query)
    result = result.first()
    if not result:
        print(f" добавляем акк")
        # Вставляем новую запись в таблицу
        # Создаем объект MetaData
        insert_query = insert(user).values(user=id)
        print(insert_query)
        compiled = insert_query.compile()
        print(compiled.params)
        add = connection.execute(insert_query)
        connection.commit()
        print(f" add {add}")
        ddict = {'answer': 'Ваш аккаунт создан'}
    else:
    # Печатаем результат запроса
        for row in result:
            if row == id:
                ddict = {'answer': 'Ваш аккаунт уже активирован'}
            print(row)

    # Закрываем соединение
    connection.close()
    return ddict

def dbCheck():
    connection = dbConnect()
    print(f"exc {type(connection)}")
    if type(connection) == tuple:
        return connection

    connection.close()
    return 'Соедение с базой успешное, целостность базы не нарушена'

def dbAddCoin(exchange, coin, user_id):
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




def TestApiByBit(api_key, api_secret, user_id):
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
            connection = dbConnect()
            query = select(user).where(user.c.user == user_id)
            result = connection.execute(query)
            result = result.first()
            if not result:
                ddict = {'answer': 'пользовате не создан, запустите команду start'}
            else:
                ddict = {'answer': f'Api bybit для вашего логина обновлен и работает'}
                if result.api is None:
                    connection.execute(api.insert().values(bybit_key=api_key, bybit_secret=api_secret))
                    fk = connection.execute(select(api).where(api.c.bybit_secret == api_secret)).first()
                    connection.execute(user.update().where(user.c.user == user_id).values(api=fk.id))
                    connection.commit()
                    print(f"fk {fk.bybit_key}")
                    #result.api = fk.
                    print(f"ins {result}")
                else:
                    connection.execute(api.update().where(api.c.id == result.api).values(bybit_key=api_key, bybit_secret=api_secret))
                    connection.commit()
            # Закрываем соединение
            connection.close()

    except InvalidRequestError as e:
        return {'answer': e}
    return ddict
