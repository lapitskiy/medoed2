from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, select
from sqlalchemy import text
from config import config


def insert_data(db, table, rows):
    db.execute(text(f"INSERT INTO {table} VALUES (:id, :fname, :lname, :mail, :age)"), params)
    db.commit()

def dbAccCheck(id):
    ddict = {}
    params = ({"user": id})

    print(f"id {id}")
    print(f"user {config.user_db.get_secret_value()}")
    print(f"pass {config.pass_db.get_secret_value()}")
    print(f"database_name {config.name_db.get_secret_value()}")
    print(f"host {config.host_db.get_secret_value()}")

    # Создаем строку подключения
    connection_string = f'postgresql+psycopg2://{config.user_db.get_secret_value()}:{config.pass_db.get_secret_value()}@{config.host_db.get_secret_value()}/{config.name_db.get_secret_value()}'

    # Создаем объект Engine, который представляет собой интерфейс к базе данных
    engine = create_engine(connection_string)

    # Устанавливаем соединение с базой данных
    connection = engine.connect()

    metadata = MetaData()
    metadata.reflect(bind=engine)

    table = metadata.tables['user']
    print(f" table {table}")
    # Выполняем SQL-запрос
    #result = connection.execute(f"SELECT user FROM user WHERE user = :user", params)

    # Выполняем SELECT запрос с параметрами
    query = select(table).where(table.c.user == id)
    #query = select(table).where(table.c.user == id)
    result = connection.execute(query)
    if result:
        print(f" есть запись {result}")

    if not result:
        print(f" добавляем акк")
        # Вставляем новую запись в таблицу
        # Создаем объект MetaData
        insert_query = table.insert().values(user=id)
        connection.execute(insert_query)
        ddict = {'answer': 'Ваш аккаунт добавлен в бота'}

    # Печатаем результат запроса
    for row in result:
        print(f" result2 {result}")
        print(row)

    # Закрываем соединение
    connection.close()
    return ddict
