from sqlalchemy import MetaData, Table, Column, Integer, String, ForeignKey, Identity

#alembic revision --message="Initial" --autogenerate
#>alembic upgrade head

metadata = MetaData()

api = Table('api', metadata,
    Column('id', Integer, Identity(always=True), primary_key=True),
    Column('bybit_secret', String(200), nullable=False, unique=True),
    Column('bybit_key', String(200),  nullable=False, unique=True)
)

user = Table('user', metadata,
                       Column('id', Integer, Identity(always=True), primary_key=True),
                       Column('user', Integer, unique=True),
                       Column('api', ForeignKey('api.id', ondelete='SET NULL')),
                       )



