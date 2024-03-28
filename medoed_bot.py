import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command
from config import config

# Включаем логирование, чтобы не пропустить важные сообщения
logging.basicConfig(level=logging.INFO)
# Объект бота
bot = Bot(token=config.bot_token.get_secret_value())
bot = Bot(token="1149862512:AAFHaUq_sy_ucvENdke7lOOh9CxVWVnm9Qs")
# Диспетчер
dp = Dispatcher(bot)





class GetTokenState(StatesGroup):
    token_state = State()  # Will be represented in storage as 'Mydialog:otvet'


# Хэндлер на команду /start
@dp.message_handler(commands=["start"])
async def cmd_start(message: types.Message):
    #keyboard_markup = types.InlineKeyboardMarkup()
    #user_id_btn = types.InlineKeyboardButton('Проверить токен бинанс', callback_data='user_id')
    #keyboard_markup.row(user_id_btn)

    await GetTokenState.token_state.set()

    kb = [
        [
            types.KeyboardButton(text="Аккаунт"),
            types.KeyboardButton(text="Сделки"),
        ],
    ]

    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True, keyboard=kb)

    await message.answer(f"Ваш ID: {message.from_user.id}. Укажите токен с бинанс для создания вашего аккаунта", reply_markup=keyboard)


# А здесь получаем ответ, указывая состояние и передавая сообщение пользователя
@dp.message_handler(state=GetTokenState.token_state)
async def process_message(message: types.Message, state: FSMContext):

    async with state.proxy() as data:
        data['text'] = message.text
        user_message = data['text']

        # дальше ты обрабатываешь сообщение, ведешь рассчеты и выдаешь ему ответ.

        answer = 'ваш токен '

        await bot.send_message(
            message.from_user.id,
            answer,
            parse_mode='HTML',
        )

    # Finish conversation
    await state.finish()  # закончили работать с сотояниями

@dp.message_handler(commands=["token"])
async def token(message: types.Message):
    await message.reply("Введите ваш токен Binance")


@dp.callback_query_handler(text='user_id')
async def user_id_inline_callback(callback_query: types.CallbackQuery):
    await callback_query.answer(f"Ваш токен бинанс: xxxx", True)

# Запуск процесса поллинга новых апдейтов
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())