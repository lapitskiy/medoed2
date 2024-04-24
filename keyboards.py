from aiogram.filters.callback_data import CallbackData
from aiogram.utils.keyboard import ReplyKeyboardBuilder, InlineKeyboardBuilder
from sqlalchemy.orm import Session, selectinload

from strategy import stg_dict
from utils import create_session
from utils_db import getEngine
from models import User, Strategy


class MyCallback(CallbackData, prefix="main"):
    foo: str

class TradeCallback(CallbackData, prefix="trade_menu"):
    foo: str
    id: int

class EditStgCallback(CallbackData, prefix="trade_menu"):
    foo: str
    id: int
    action: str

class ChooseStgCallback(CallbackData, prefix="trade_menu"):
    foo: str
    id: int
    stg_name: str

class ReportCallback(CallbackData, prefix="report_menu"):
    foo: str
    id: int
    action: str

def main_keybord():
    builder = InlineKeyboardBuilder()
    builder.button(
        text="Сводка",
        callback_data=MyCallback(foo="report")  # Value can be not packed to string inplace, because builder knows what to do with callback instance
    )
    builder.button(
        text="Торговля",
        callback_data=MyCallback(foo="trade")  # Value can be not packed to string inplace, because builder knows what to do with callback instance
    )
    builder.button(
        text="Настройки",
        callback_data=MyCallback(foo="settings")  # Value can be not packed to string inplace, because builder knows what to do with callback instance
    )
    return builder.as_markup()

''' REPORT KEYBORD '''

def report_keybord(id):
    builder = InlineKeyboardBuilder()
    session = create_session()
    print(id)
    query = session.query(User).filter_by(user=id).one()
    txt = 'Телетайп ВКЛ' if query.teletaip else 'Телетайп ВЫКЛ'
    session.close()
    builder.button(
        text=txt,
        callback_data=ReportCallback(foo='report', id=id, action='teletipe')
        # Value can be not packed to string inplace, because builder knows what to do with callback instance
    )
    builder.button(
        text="Скачать отчет",
        callback_data=ReportCallback(foo="report", id=id, action='reportpdf')  # Value can be not packed to string inplace, because builder knows what to do with callback instance
    )
    builder.button(
        text="Назад",
        callback_data=MyCallback(foo="main_menu")  # Value can be not packed to string inplace, because builder knows what to do with callback instance
    )
    return builder.as_markup()

''' STG KEYBORD '''

def stg_keybord(stg_id: int):
    builder = InlineKeyboardBuilder()
    ddict = {}
    with Session(getEngine()) as session:
        query = session.query(Strategy).filter_by(id=stg_id).one()
        txt = 'Остановить' if query.start else 'Запустить'
        builder.button(
            text=txt,
            callback_data=EditStgCallback(foo='stg_edit', id=stg_id, action='start')
            # Value can be not packed to string inplace, because builder knows what to do with callback instance
        )
    builder.button(
        text="Стратегия",
        callback_data=ChooseStgCallback(foo="stg_choose", id=stg_id, stg_name='')  # Value can be not packed to string inplace, because builder knows what to do with callback instance
    )
    builder.button(
        text="Назад",
        callback_data=MyCallback(foo="trade")  # Value can be not packed to string inplace, because builder knows what to do with callback instance
    )
    builder.adjust(2)
    return builder.as_markup()

def stg_choose_keybord(stg_id: int):
    builder = InlineKeyboardBuilder()
    ddict = {}
    for key, value in stg_dict.items():  # from strategy import *
        builder.button(
            text=stg_dict[key]['name'],
            callback_data=ChooseStgCallback(foo='stg_choose', id=stg_id, stg_name=key)
            # Value can be not packed to string inplace, because builder knows what to do with callback instance
        )

    builder.button(
        text="Назад",
        callback_data=EditStgCallback(foo='stg_edit', id=stg_id, action='')
        # Value can be not packed to string inplace, because builder knows what to do with callback instance
    )
    builder.adjust(2)
    return builder.as_markup()

def trade_keybord(user_id: int):
    ddict = {'stg':''}
    builder = InlineKeyboardBuilder()
    '''
    TRADE ORM BUTTON
    '''
    with Session(getEngine()) as session:
        query = session.query(User).options(selectinload(User.stg)).filter_by(user=user_id).one()
        if query.stg:
            for key in query.stg:
                builder.button(
                    text=key.symbol,
                    callback_data=TradeCallback(foo='stg', id=key.id)
                    # Value can be not packed to string inplace, because builder knows what to do with callback instance
                )

    '''
    SIMPLE BUTTON
    '''
    builder.button(
        text="Добавить крипту",
        callback_data=MyCallback(foo="addcoin")  # Value can be not packed to string inplace, because builder knows what to do with callback instance
    )
    builder.button(
        text="Назад",
        callback_data=MyCallback(foo="main_menu")  # Value can be not packed to string inplace, because builder knows what to do with callback instance
    )
    builder.adjust(2)
    return builder.as_markup()

def settings_keybord():
    builder = InlineKeyboardBuilder()
    builder.button(
        text="Настройки ByBit API",
        callback_data=MyCallback(foo="api_bybit")  # Value can be not packed to string inplace, because builder knows what to do with callback instance
    )
    builder.button(
        text="Назад",
        callback_data=MyCallback(foo="main_menu")  # Value can be not packed to string inplace, because builder knows what to do with callback instance
    )
    return builder.as_markup()

def keybord_crypto_replay():
    builder = ReplyKeyboardBuilder()
    builder.button(text="Добавить крипту")
    builder.button(text="Назад в меню")
    builder.adjust(3, 3)
    return builder.as_markup(resize_keyboard=True, input_field_placeholder='Выберите пункт меню')
