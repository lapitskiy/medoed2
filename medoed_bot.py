import asyncio
import logging
from aiogram import Bot, Dispatcher

from aiogram import F
from aiogram.types import Message, CallbackQuery
from aiogram.filters import Command, CommandObject, CommandStart

from utils import *
from keyboards import *

from keyboards import MyCallback

# Включаем логирование, чтобы не пропустить важные сообщения
logging.basicConfig(level=logging.INFO)
# Объект бота
bot = Bot(token=config.bot_token.get_secret_value(), parse_mode="HTML")
# Диспетчер
dp = Dispatcher()

''' INLINE MAIN'''
''' INLINE MAIN'''
''' INLINE MAIN'''

@dp.message(Command("start"))
async def add_bybit_api_command(message: Message, command: CommandObject):
        result = dbAccCheck(message.from_user.id)
        await message.delete()
        await message.answer(f"Привет, <b>{message.from_user.username}</b>!",
                             reply_markup=main_keybord())
        return

# Filter callback by type and value of field :code:`foo`
@dp.callback_query(MyCallback.filter(F.foo == "main_menu"))
async def callback_main_menu(query: CallbackQuery, callback_data: MyCallback):
    #await query.message.delete()
    await query.message.edit_text(f"Главное меню", reply_markup=main_keybord())

# Filter callback by type and value of field :code:`foo`
@dp.callback_query(MyCallback.filter(F.foo == "report"))
async def callback_report(query: CallbackQuery, callback_data: MyCallback):
    await query.message.edit_text(f"Сводка будет тут", reply_markup=main_keybord())

# Filter callback by type and value of field :code:`foo`
@dp.callback_query(MyCallback.filter(F.foo == "settings"))
async def callback_settings(query: CallbackQuery, callback_data: MyCallback):
    #await query.message.delete()
    await query.message.edit_text(f"Настройки", reply_markup=settings_keybord())

''' INLINE TRADE'''
''' INLINE TRADE'''
''' INLINE TRADE'''

# Filter callback by type and value of field :code:`foo`
@dp.callback_query(MyCallback.filter(F.foo == "trade"))
async def callback_trade_settings(query: CallbackQuery, callback_data: MyCallback):
    result = CheckExistCoin(query.from_user.id)
    await query.message.edit_text(f"{result['answer']}\n", reply_markup=trade_keybord())

# Filter callback by type and value of field :code:`foo`
@dp.callback_query(MyCallback.filter(F.foo == "addcoin"))
async def callback_addcoin(query: CallbackQuery, callback_data: MyCallback):
    await query.message.edit_text(f"Введите название торгуемой пары в формате\n<b>/addcoin bybit TONUSDT</b>", reply_markup=trade_keybord())

''' TRADE COMMAND'''

@dp.message(Command("addcoin"))
async def add_coin_command(
        message: Message,
        command: CommandObject):
    # Если не переданы никакие аргументы, то
    # command.args будет None
    if command.args is None:
        await message.answer(
            "Ошибка: неправильный формат команды. Пример:\n"
            "/addcoin bybit TONUSDT"
        )
        return
    try:

        exchange, coin = command.args.split(" ", maxsplit=1)
        # Если получилось меньше двух частей, вылетит ValueError
    except ValueError:
        await message.answer(
            "Ошибка: неправильный формат команды. Пример:\n"
            "/addcoin bybit TONUSDT"
        )
        return
    result = AddCoin(exchange, coin, message.from_user.id)
    await message.answer(f"{result['answer']}")


''' INLINE SETTINGS'''
''' INLINE SETTINGS'''
''' INLINE SETTINGS'''

# Filter callback by type and value of field :code:`foo`
@dp.callback_query(MyCallback.filter(F.foo == "api_bybit"))
async def callback_api_bybit(query: CallbackQuery, callback_data: MyCallback):
    #await query.message.delete()
    result = CheckApiEx(query.from_user.id)
    await query.message.edit_text(f"{result['answer']}\nВведите ключ api ByBit в формате \n/api_bybit api_key api_secret", reply_markup=settings_keybord())

''' COMMAND '''
''' COMMAND '''
''' COMMAND '''

@dp.message(Command("api_bybit"))
async def add_bybit_api_command(
        message: Message,
        command: CommandObject
):
    # Если не переданы никакие аргументы, то
    # command.args будет None
    if command.args is None:
        await message.answer(
            "Ошибка: неправильный формат команды. Пример:\n"
            "/api_bybit X3sGs9GrEZQtoTsQTQ Z8KZwvF3YToRtX0tUeBaYLmspudqm36TQ51X"
        )
        return
    try:

        api_key, api_secret = command.args.split(" ", maxsplit=1)
        # Если получилось меньше двух частей, вылетит ValueError
    except ValueError:
        await message.answer(
            "Ошибка: неправильный формат команды. Пример:\n"
            "/api_bybit X3sGs9GrEZQtoTsQTQ Z8KZwvF3YToRtX0tUeBaYLmspudqm36TQ51X"
        )
        return
    # Пробуем разделить аргументы на две части по первому встречному пробелу

    result = AddApiByBit(api_key, api_secret, message.from_user.id)
    await message.delete()
    await message.answer(f"{result['answer']}")



# Запуск процесса поллинга новых апдейтов
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    #trade()
    asyncio.run(main())
