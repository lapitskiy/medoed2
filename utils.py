from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, select, insert
from sqlalchemy import text

from pybit.unified_trading import HTTP
from pybit.exceptions import InvalidRequestError
from utils_db import *


def dbAccCheck(id):
    ddict = {}

    print(f"id {id}")
    print(f"user {config.user_db.get_secret_value()}")
    print(f"pass {config.pass_db.get_secret_value()}")
    print(f"database_name {config.name_db.get_secret_value()}")
    print(f"host {config.host_db.get_secret_value()}")

    connection = dbConnect()

    metadata = MetaData()

    table = Table('user', metadata,
                       Column('user', Integer)
                       )

    print(f" table {table}")
    query = select(table).where(table.c.user == id)
    result = connection.execute(query)
    result = result.first()
    if not result:
        print(f" добавляем акк")
        # Вставляем новую запись в таблицу
        # Создаем объект MetaData
        insert_query = insert(table).values(user=id)
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

def dbAddApi(api):
    ddict = {}
    params = ({"user": id})



def TestApiByBit(api_key, api_secret):
    ddict = {}
    session = HTTP(
        testnet=False,
        api_key='G3sGs9GrEZQtoTsQTQ',
        api_secret='K8KZwvF3YToRtX0tUeBaYLmspudqm36TQ51S',
    )
    try:

        ddict['result'] = session.get_wallet_balance(
            accountType="UNIFIED",
            coin="BTC",
        )
    except InvalidRequestError as e:
        return {'error': e}
    return ddict
