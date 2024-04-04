from __future__ import annotations
from typing import List, Optional

from sqlalchemy import Column, Integer, String, ForeignKey, Identity, JSON, MetaData, Any
from sqlalchemy.ext.declarative import declarative_base

from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import relationship

#alembic revision --message="Initial" --autogenerate
#>alembic upgrade head

#metadata = MetaData()

class Base(DeclarativeBase):
    type_annotation_map = {
        dict[str, Any]: JSON
    }
    #metadata = metadata

class User(Base):
    __tablename__ = 'users'
    id: Mapped[int] = mapped_column(Identity(always=True), primary_key=True)
    user: Mapped[int] = mapped_column(unique=True)
    api: Mapped[int] = mapped_column(ForeignKey('api.id', ondelete='SET NULL'))
    stg: Mapped[List[Strategy]] = relationship()

class Api(Base):
    __tablename__ = 'api'
    id: Mapped[int] = mapped_column(Identity(always=True), primary_key=True)
    bybit_secret: Mapped[str] = mapped_column(nullable=False, unique=True)
    bybit_key: Mapped[str] = mapped_column(nullable=False, unique=True)

class Strategy(Base):
    __tablename__ = 'strategy'
    id: Mapped[int] = mapped_column(Identity(always=True), primary_key=True)
    symbol: Mapped[str] = mapped_column(nullable=False)
    limit: Mapped[int] = mapped_column(nullable=False)
    stg_dict: Mapped[dict[str, Any]] = mapped_column(type_=JSON)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))



'''
class Api(Base):
    __tablename__ = 'api'
    id = Column(Integer(), Identity(always=True), primary_key=True)
    bybit_secret = Column(String(200), nullable=False, unique=True)
    bybit_key = Column(String(200),  nullable=False, unique=True)

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer(), Identity(always=True), primary_key=True)
    user = Column(Integer(), unique=True)
    api = Column(Integer(), ForeignKey('api.id', ondelete='SET NULL'))
    stg = relationship('Strategy', backref=backref('user'))

class Strategy(Base):
    __tablename__ = 'strategy'
    id = Column(Integer(), Identity(always=True), primary_key=True)
    symbol = Column(String(200), nullable=False)
    limit = Column(String(200), nullable=False)
    stg_dict = Column(JSON)
    user_id = Column(Integer(), ForeignKey('users.id'))
'''

