from __future__ import annotations
from typing import List, Optional
import sqlalchemy as sa

from sqlalchemy import Column, Integer, String, ForeignKey, Identity, JSON, MetaData, Any, BigInteger
from sqlalchemy.ext.declarative import declarative_base

from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import relationship

#alembic revision --message="Initial" --autogenerate
#>alembic upgrade head

class Base(DeclarativeBase):
    type_annotation_map = {
        dict[str, Any]: JSON
    }

class User(Base):
    __tablename__ = 'users'
    id: Mapped[int] = mapped_column(Identity(always=True), primary_key=True)
    user: Mapped[int] = mapped_column(sa.BigInteger, unique=True)
    api: Mapped[List[Api]] = relationship(back_populates="user")
    stg: Mapped[List[Strategy]] = relationship(back_populates="user")
    teletaip: Mapped[bool] = mapped_column(default=False, nullable=False)

class Api(Base):
    __tablename__ = 'api'
    id: Mapped[int] = mapped_column(Identity(always=True), primary_key=True)
    bybit_secret: Mapped[str] = mapped_column(nullable=False, unique=True)
    bybit_key: Mapped[str] = mapped_column(nullable=False, unique=True)
    user: Mapped['User'] = relationship(back_populates="api")
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))

class Strategy(Base):
    __tablename__ = 'strategy'
    id: Mapped[int] = mapped_column(Identity(always=True), primary_key=True)
    symbol: Mapped[str] = mapped_column(nullable=False)
    limit: Mapped[int] = mapped_column(nullable=False)
    start: Mapped[bool] = mapped_column(nullable=False)
    stg_name: Mapped[str] = mapped_column(nullable=True)
    stg_dict: Mapped[dict[str, Any]] = mapped_column(type_=JSON, nullable=True)
    user: Mapped['User'] = relationship(back_populates="stg")
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))

class TradeHistory(Base):
    __tablename__ = 'history'
    id: Mapped[int] = mapped_column(Identity(always=True), primary_key=True)
    price: Mapped[str] = mapped_column(nullable=False)
    price_tx: Mapped[int] = mapped_column(sa.BigInteger, unique=True)

