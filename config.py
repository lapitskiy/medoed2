from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr


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
    chat_id = None

# При импорте файла сразу создастся
# и провалидируется объект конфига,
# который можно далее импортировать из разных мест
config = Settings()
secret_config = SecretSettings()