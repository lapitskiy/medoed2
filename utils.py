import emoji #https://carpedm20.github.io/emoji/

from pybit.unified_trading import HTTP
from pybit.exceptions import InvalidRequestError
from sqlalchemy import select
from sqlalchemy.orm import selectinload, Session, sessionmaker

from models import User, Api, Strategy

from config import config, secret_config
from strategy import getStgObjFromClass
from utils_db import getEngine

def create_session():
    Session = sessionmaker(getEngine())
    return Session()

def dbAccCheck(id):
    with Session(getEngine()) as session:
        statement = select(User).filter_by(user=id)
        user_obj = session.scalars(statement).all()
        if not user_obj:
            session.add(User(user=id))
            session.commit()
            ddict = {'answer': 'Ваш аккаунт создан'}
        else:
            ddict = {'answer': 'Ваш аккаунт уже активирован'}
    return ddict

def dbCheck():
    connection = dbConnect()
    print(f"exc {type(connection)}")
    if type(connection) == tuple:
        return connection

    connection.close()
    return 'Соедение с базой успешное, целостность базы не нарушена'

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
                            print(f"coin {coin} {query.id}")
                            stg = Strategy(symbol=coin, limit=0, start=False, user_id=query.id)
                            print(f"query.stg {query.stg}")
                            session.add(stg)
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
    ddict = {'answer' : '<b>Подключенные пары:</b>\n'}
    with Session(getEngine()) as session:
        query = session.query(User).options(selectinload(User.stg)).filter_by(user=user_id).one()
    if not query.stg:
        ddict['answer'] = ddict['answer'] + f'У вас не добавлены пары для торговли\n'
    else:
        for key in query.stg:
            emj = emoji.emojize(":check_mark_button:") if key.start else emoji.emojize(":stop_sign:")
            ddict['answer'] = ddict['answer'] + f'{key.symbol} {emj}\n'
    return ddict

def getStgData(stg_id):
    with Session(getEngine()) as session:
        query = session.query(Strategy).filter_by(id=stg_id).one()
        ddict = {'answer': f'<b>{query.symbol}</b>:\n'}
        if not query:
            ddict['answer'] = ddict['answer'] + f'Нет\n'
        else:
            start_txt = f'Торговля запушена '+ emoji.emojize(":check_mark_button:")+'\n' if query.start else f'Торговля остановлена' + emoji.emojize(":stop_sign:")+'\n'
            ddict['answer'] = ddict['answer'] + start_txt

            if query.stg_dict is None:
                ddict['answer'] = ddict['answer'] + f'Стратегия торговли не установлена\n'
            else:
                stgObj = getStgObjFromClass(stg_id=stg_id)
                ddict['answer'] = ddict['answer'] + stgObj.getDescriptionStg()
    return ddict['answer']
