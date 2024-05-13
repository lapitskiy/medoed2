import asyncio
import logging
import time
import threading

from aiogram import Bot, Dispatcher, types

from aiogram import F
from aiogram.types import Message, CallbackQuery, InputFile, FSInputFile
from aiogram.filters import Command, CommandObject, CommandStart
from sqlalchemy.exc import NoResultFound

from keyboards import MyCallback, TradeCallback, EditStgCallback, ChooseStgCallback, main_keybord, settings_keybord, \
    trade_keybord, stg_keybord, ReportCallback, report_keybord, stg_choose_keybord
from models import User, Strategy

from stg_router import getStgObjFromClass, stg_dict, splitCommandStg
from config import config, secret_config
from cl_thread import StartTrade

# print(f"query {query.user}")

# Включаем логирование, чтобы не пропустить важные сообщения
from utils import dbAccCheck, CheckExistCoin, create_session, AddCoin, CheckApiEx, AddApiByBit, DelApiByBit

logging.basicConfig(level=logging.INFO)
# Объект бота
bot = Bot(token=secret_config.bot_token.get_secret_value(), parse_mode="HTML")
# Диспетчер
dp = Dispatcher()

# Функция для бесконечного цикла в асинхронном потоке
async def async_infinite_loop():
    while True:
        await asyncio.sleep(0)
        try:
            if config.update_message:
                config.update_message = False
                if config.message != config.last_message and config.format_message == 'text':
                    config.last_message = config.message
                    await bot.send_message(config.chat_id, str(config.message))
                if config.format_message == 'image':
                    #img_byte_arr = BytesIO()
                    #config.image.save(img_byte_arr, format='PNG')  # Сохраняем изображение в BytesIO объект
                    #img_byte_arr.seek(0)
                    #print(img_byte_arr)
                    #photo = InputFile("plot.png")
                    config.format_message = 'text'
                    await bot.send_photo(config.chat_id, photo=FSInputFile(config.image_path))
        except Exception as e:
            print(f'msg exc {e}')
            continue

''' INLINE MAIN'''
''' INLINE MAIN'''
''' INLINE MAIN'''

@dp.message(Command("start"))
async def start_command(message: Message, command: CommandObject):
        dbAccCheck(message.from_user.id)
        config.chat_id = message.from_user.id

        #await message.delete()
        await message.answer(f"Привет, <b>{message.from_user.username}</b>!",
                             reply_markup=main_keybord())

# Filter callback by type and value of field :code:`foo`
@dp.callback_query(MyCallback.filter(F.foo == "main_menu"))
async def callback_main_menu(query: CallbackQuery, callback_data: MyCallback):
    #await query.message.delete()
    await query.message.edit_text(f"Главное меню", reply_markup=main_keybord())

# Filter callback by type and value of field :code:`foo`
@dp.callback_query(MyCallback.filter(F.foo == "report"))
async def callback_report(query: CallbackQuery, callback_data: MyCallback):
    await query.message.edit_text(f"Телетайп и отчеты по торговле", reply_markup=report_keybord(query.from_user.id))

# Filter callback by type and value of field :code:`foo`
@dp.callback_query(MyCallback.filter(F.foo == "settings"))
async def callback_settings(query: CallbackQuery, callback_data: MyCallback):
    #await query.message.delete()
    await query.message.edit_text(f"Настройки", reply_markup=settings_keybord())

''' INLINE REPORT'''
''' INLINE REPORT'''
''' INLINE REPORT'''

# Filter callback by type and value of field :code:`foo`
@dp.callback_query(ReportCallback.filter(F.foo == "report"))
async def callback_report(query: CallbackQuery, callback_data: MyCallback):
    if callback_data.action == 'teletipe':
        session = create_session()
        print(f"callback_data.id {callback_data.id}")
        query = session.query(Strategy).filter(Strategy.user.user == callback_data.id).one()
        answer = 'Телетайп ВКЛ' if query.user.teletaip else 'Телетайп ВЫКЛ'
        session.close()
    if callback_data.action == 'reportpdf':
        pass
    await query.message.edit_text(f"{result['answer']}\n", reply_markup=trade_keybord(user_id=query.from_user.id))


''' INLINE TRADE'''
''' INLINE TRADE'''
''' INLINE TRADE'''

# меню монет

@dp.callback_query(MyCallback.filter(F.foo == "trade"))
async def callback_trade_settings(query: CallbackQuery, callback_data: MyCallback):
    result = CheckExistCoin(query.from_user.id)
    await query.message.edit_text(f"{result['answer']}\n", reply_markup=trade_keybord(user_id=query.from_user.id))


# Filter callback by type and value of field :code:`foo`
@dp.callback_query(MyCallback.filter(F.foo == "addcoin"))
async def callback_addcoin(query: CallbackQuery, callback_data: MyCallback):
    await query.message.edit_text(f"Введите название торгуемой пары в формате\n<b>/addcoin bybit TONUSDT</b>", reply_markup=trade_keybord(user_id=query.from_user.id))

''' TRADE ORM KEYBORD'''

# меню выбранной монеты!
@dp.callback_query(TradeCallback.filter(F.foo == 'stg'))
async def callback_trade(query: CallbackQuery, callback_data: TradeCallback):
    #answer = getStgData(stg_id=callback_data.id)
    print(f'{callback_data.id}')
    stgObj = getStgObjFromClass(stg_id=callback_data.id)
    if stgObj is None:
        answer = 'Стратегия для пары не добавлена'
    else:
        answer = stgObj.getDescriptionStg()
    await query.message.edit_text(f"{answer}", reply_markup=stg_keybord(stg_id=callback_data.id))

''' Меню конкретной монеты, запуск и выбор стратегии '''
@dp.callback_query(EditStgCallback.filter(F.foo == 'stg_edit'))
async def callback_edit_stg(query: CallbackQuery, callback_data: EditStgCallback):
    answer = {'answer':''}
    stgObj = getStgObjFromClass(stg_id=callback_data.id)
    if stgObj:
        if callback_data.action == 'start':
            answer = stgObj.CheckStopStartStg(change=True)
            answer['answer'] = stgObj.getDescriptionStg()
        else:
            answer['answer'] = stgObj.getDescriptionStg()
    else:
        answer = {'answer': 'Создайте сначала <u>стратегию</u> для запуска!\n\n'}
    await query.message.edit_text(f"\n{answer['answer']}", reply_markup=stg_keybord(stg_id=callback_data.id))

@dp.callback_query(ChooseStgCallback.filter(F.foo == 'stg_choose'))
async def callback_choose_stg(query: CallbackQuery, callback_data: ChooseStgCallback):
    answer = 'Выберите стратегию для получения помощи по настройке\n'
    if callback_data.stg_name in stg_dict:
        print(f'stg_name call {callback_data.stg_name}')
        print(f'id call {callback_data.id}')
        stgObj = getStgObjFromClass(stg_id=callback_data.id, stg_name=callback_data.stg_name)
        answer = stgObj.returnHelpInfo(stg_id=callback_data.id)
    await query.message.edit_text(f"{answer}\n", reply_markup=stg_choose_keybord(stg_id=callback_data.id))

@dp.callback_query(ChooseStgCallback.filter(F.foo == 'stg_backtest'))
async def callback_choose_stg(query: CallbackQuery, callback_data: ChooseStgCallback):
    answer = 'Backtest текущей стратегии\n'
    stgObj = getStgObjFromClass(stg_id=callback_data.id)
    answer = stgObj.backtest()
    await query.message.edit_text(f"{answer}\n", reply_markup=stg_choose_keybord(stg_id=callback_data.id))


''' TRADE STRATEGY COMMAND'''

@dp.message(Command('stgedit'))
async def stgedit_command(message: Message, command: CommandObject):

    # Если не переданы никакие аргументы, то
    # command.args будет None
    if command.args is None:
        await message.answer(
            "Ошибка: неправильный формат команды\n"
        )
        return
    result = splitCommandStg(stgedit=command.args)
    await message.answer(f"{result}")


''' TRADE COMMAND'''

@dp.message(Command("addcoin"))
async def add_coin_command(message: Message, command: CommandObject):
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
    await query.message.edit_text(f"{result['answer']}\nВведите ключ api ByBit в формате \n/api_bybit api_key api_secret"
                                  f"\n\nУдалить api bybit\n/api_bybit del api_key", reply_markup=settings_keybord())

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

        value1, value2 = command.args.split(" ", maxsplit=1)
        # Если получилось меньше двух частей, вылетит ValueError
    except ValueError:
        await message.answer(
            "Ошибка: неправильный формат команды. Пример:\n"
            "/api_bybit X3sGs9GrEZQtoTsQTQ Z8KZwvF3YToRtX0tUeBaYLmspudqm36TQ51X"
        )
        return
    # Пробуем разделить аргументы на две части по первому встречному пробелу
    if value1 == 'del':
        result = DelApiByBit(value2, message.from_user.id)
    else:
        result = AddApiByBit(value1, value2, message.from_user.id)
    await message.delete()
    print(f"answer {result['answer']}")
    await message.answer(f"{result['answer']}")


# Запуск процесса поллинга новых апдейтов
async def main():
    trade = StartTrade()
    trade.start()
    loop = asyncio.get_running_loop()
    loop.create_task(async_infinite_loop())
    await dp.start_polling(bot, skip_updates=True)


if __name__ == "__main__":
    asyncio.run(main())


