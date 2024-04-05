from sqlalchemy import exc
from sqlalchemy import create_engine
from config import config

def getEngine():    # Создаем строку подключения
    connection_string = f'postgresql+psycopg2://{config.user_db.get_secret_value()}:{config.pass_db.get_secret_value()}@{config.host_db.get_secret_value()}/{config.name_db.get_secret_value()}'
    try:
        engine = create_engine(connection_string)
    except exc.SQLAlchemyError as e:
        return e.args
    return engine