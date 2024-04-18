from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr
from sqlalchemy.orm import sessionmaker
from sqlalchemy import exc, create_engine

from models import User


def getEngine():    # Создаем строку подключения
    connection_string = f'postgresql+psycopg2://{secret_config.user_db.get_secret_value()}:{secret_config.pass_db.get_secret_value()}@{secret_config.host_db.get_secret_value()}/{secret_config.name_db.get_secret_value()}'
    try:
        engine = create_engine(connection_string)
    except exc.SQLAlchemyError as e:
        return e.args
    return engine


class SecretSettings(BaseSettings):
    # Желательно вместо str использовать SecretStr
    # для конфиденциальных данных, например, токена бота
    bot_token: SecretStr
    user_db: SecretStr
    pass_db: SecretStr
    host_db: SecretStr
    name_db: SecretStr

    # Начиная со второй версии pydantic, настройки класса настроек задаются
    # через model_config
    # В данном случае будет использоваться файла .env, который будет прочитан
    # с кодировкой UTF-8
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

class Settings():
    chat_id: int = None
    message: str = None

    def __init__(self):
        self.update_message = False

    def getTgId(self, user_id: int):
        Session = sessionmaker(getEngine())
        session = Session()
        query = session.query(User).filter_by(id=user_id).one()
        if query:
            self.chat_id = query.user
        session.close()


# При импорте файла сразу создастся
# и провалидируется объект конфига,
# который можно далее импортировать из разных мест
config = Settings()
secret_config = SecretSettings()