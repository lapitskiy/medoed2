import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from config import config
from trade import trade

from aiogram import F
from aiogram.types import Message
from aiogram.filters import Command
from aiogram.enums import ParseMode

from utils import *

# Включаем логирование, чтобы не пропустить важные сообщения
logging.basicConfig(level=logging.INFO)
# Объект бота
bot = Bot(token=config.bot_token.get_secret_value(), parse_mode="HTML")
# Диспетчер
dp = Dispatcher()




@dp.message(F.text, Command("start"))
async def any_message(message: Message):
    dbAccCheck(message.from_user.id)
    await message.answer(
        f"Hello, <b>{message.from_user.username}</b>!"
    )

# Запуск процесса поллинга новых апдейтов
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    #trade()
    asyncio.run(main())
