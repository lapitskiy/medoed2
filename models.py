from sqlalchemy import MetaData, Table, Column, Integer, String, ForeignKey

metadata = MetaData()

api = Table('api', metadata,
    Column('bybit_secret', String(200), nullable=False, unique=True),
    Column('bybit_key', String(200),  nullable=False, unique=True)
)

user = Table('user', metadata,
                       Column('user', Integer, unique=True)
                       )



