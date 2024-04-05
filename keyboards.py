from aiogram.filters.callback_data import CallbackData
from aiogram.utils.keyboard import ReplyKeyboardBuilder, InlineKeyboardBuilder



class MyCallback(CallbackData, prefix="my"):
    foo: str
    bar: int

def main_keybord():
    builder = InlineKeyboardBuilder()
    builder.button(
        text="Сводка",
        callback_data=MyCallback(foo="report", bar="1")  # Value can be not packed to string inplace, because builder knows what to do with callback instance
    )
    builder.button(
        text="Торговля",
        callback_data=MyCallback(foo="trade", bar="2")  # Value can be not packed to string inplace, because builder knows what to do with callback instance
    )
    builder.button(
        text="Настройки",
        callback_data=MyCallback(foo="settings", bar="2")  # Value can be not packed to string inplace, because builder knows what to do with callback instance
    )
    return builder.as_markup()

def trade_keybord():
    builder = InlineKeyboardBuilder()
    builder.button(
        text="Добавить крипту",
        callback_data=MyCallback(foo="addcoin", bar="1")  # Value can be not packed to string inplace, because builder knows what to do with callback instance
    )
    builder.button(
        text="Назад",
        callback_data=MyCallback(foo="main_menu", bar="2")  # Value can be not packed to string inplace, because builder knows what to do with callback instance
    )
    return builder.as_markup()

def settings_keybord():
    builder = InlineKeyboardBuilder()
    builder.button(
        text="Ввести api ByBit",
        callback_data=MyCallback(foo="api_bybit", bar="1")  # Value can be not packed to string inplace, because builder knows what to do with callback instance
    )
    builder.button(
        text="Назад",
        callback_data=MyCallback(foo="main_menu", bar="2")  # Value can be not packed to string inplace, because builder knows what to do with callback instance
    )
    return builder.as_markup()

def keybord_crypto_replay():
    builder = ReplyKeyboardBuilder()
    builder.button(text="Добавить крипту")
    builder.button(text="Назад в меню")
    builder.adjust(3, 3)
    return builder.as_markup(resize_keyboard=True, input_field_placeholder='Выберите пункт меню')
