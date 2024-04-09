from sqlalchemy import exc
from sqlalchemy import create_engine
from config import config, secret_config

def getEngine():    # Создаем строку подключения
    connection_string = f'postgresql+psycopg2://{secret_config.user_db.get_secret_value()}:{secret_config.pass_db.get_secret_value()}@{secret_config.host_db.get_secret_value()}/{secret_config.name_db.get_secret_value()}'
    try:
        engine = create_engine(connection_string)
    except exc.SQLAlchemyError as e:
        return e.args
    return engine