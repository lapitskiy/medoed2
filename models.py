from sqlalchemy import MetaData, Table, Column, Integer, String, ForeignKey, Identity, JSON
from sqlalchemy.orm import relationship, declarative_base

#alembic revision --message="Initial" --autogenerate
#>alembic upgrade head

metadata = MetaData()

api = Table('api', metadata,
    Column('id', Integer, Identity(always=True), primary_key=True),
    Column('bybit_secret', String(200), nullable=False, unique=True),
    Column('bybit_key', String(200),  nullable=False, unique=True)
)

coin = Table('coin', metadata,
    Column('id', Integer, Identity(always=True), primary_key=True),
    Column('symbol', String(200), nullable=False),
    Column('deposit', String(200), nullable=False),
    Column('strategy', String(200), nullable=False),
    Column('stg_dict', JSON),
    Column('user_id', Integer, ForeignKey('user.id'))
)

user = Table('user', metadata,
                       Column('id', Integer, Identity(always=True), primary_key=True),
                       Column('user', Integer, unique=True),
                       Column('api', ForeignKey('api.id', ondelete='SET NULL')),
                       Column('stg', relationship('coin')),
                       )



