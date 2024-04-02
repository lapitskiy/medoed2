import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from config import config
from trade import trade

from aiogram import F
from aiogram.types import Message
from aiogram.filters import Command, CommandObject
from aiogram.enums import ParseMode

from utils import *
from keyboards import *

# Включаем логирование, чтобы не пропустить важные сообщения
logging.basicConfig(level=logging.INFO)
# Объект бота
bot = Bot(token=config.bot_token.get_secret_value(), parse_mode="HTML")
# Диспетчер
dp = Dispatcher()




@dp.message(F.text, Command("start"))
async def any_message(message: Message):
    result = dbAccCheck(message.from_user.id)
    #await bot.delete_message(message.from_user.id, message.message_id)
    await message.answer(
        f"Привет, <b>{message.from_user.username}</b>! {result['answer']}", reply_markup=keybord_main_replay()
    )

''' Настройки '''
@dp.message(F.text.lower() == "настройки")
async def settings_message(message: Message):
    await bot.delete_message(message.from_user.id, message.message_id)
    await message.answer(
        f"Меню настроек", reply_markup=keybord_settings_replay()
    )

@dp.message(F.text.lower() == "ввести api bybit")
async def settings_message(message: Message):
    await bot.delete_message(message.from_user.id, message.message_id)
    await message.answer(
        f"Введите ключ api ByBit в формате /api_bybit api_key api_secret", reply_markup=keybord_settings_replay()
    )


@dp.message(Command("api_bybit"))
async def cmd_settimer(
        message: Message,
        command: CommandObject
):
    # Если не переданы никакие аргументы, то
    # command.args будет None
    if command.args is None:
        await message.answer(
            "Ошибка: вы не указали api"
        )
        return
    try:
        api_key, api_secret = command.args.split(" ", maxsplit=1)
        # Если получилось меньше двух частей, вылетит ValueError
    except ValueError:
        await message.answer(
            "Ошибка: неправильный формат команды. Пример:\n"
            "/api_bybit X3sGs9GrEZQtoTsQTQ Z8KZwvF3YToRtX0tUeBaYLmspudqm36TQ51S"
        )
        return
    # Пробуем разделить аргументы на две части по первому встречному пробелу

    result = TestApiByBit(api_key, api_secret)
    if 'error' in result:
        await message.answer(
            f"Ошибка: {result['error']}")

    if 'result' in result:
        await message.answer(
            f"Api подлючен. Ваш счет: {result['result']}")

@dp.message(F.text.lower() == "назад в меню")
async def settings_message(message: Message):
    await bot.delete_message(message.from_user.id, message.message_id)
    await message.answer(
        f"Основное меню", reply_markup=keybord_main_replay()
    )

# Запуск процесса поллинга новых апдейтов
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    #trade()
    asyncio.run(main())
