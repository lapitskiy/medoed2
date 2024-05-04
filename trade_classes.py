from sqlalchemy.orm import sessionmaker

from models import Strategy
from utils_db import getEngine

import asyncio

import emoji #https://carpedm20.github.io/emoji/
from pybit import exceptions
from pybit.exceptions import InvalidRequestError
from sqlalchemy import select, Null, update
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import Session, sessionmaker

from pybit.unified_trading import HTTP #https://www.bybit.com/en/help-center/s/webform?language=en_US

from utils_db import getEngine
from models import Strategy, TradeHistory
from config import config


def create_session():
    Session = sessionmaker(getEngine())
    return Session()

class Api_Trade_Method():
    api_session: None

        # https://bybit-exchange.github.io/docs/v5/intro
    def makeSession(self, stg_id: int):
        session_api = None
        session = create_session()
        api = session.query(Strategy).filter_by(id=stg_id).one()
        if api.user.api:
            for bybit in api.user.api:
                bybit_key = bybit.bybit_key
                bybit_secret = bybit.bybit_secret
            session_api = HTTP(
                testnet=False,
                api_key=bybit_key,
                api_secret=bybit_secret,
                recv_window=8000
            )
        session.close()
        return session_api

    def getCurrentPrice(self, symbol: str):
        #self.api_session = HTTP(testnet=False)
        return self.api_session.get_tickers(
            category="spot",
            symbol=symbol,
        )

    def BuyMarket(self, symbol: str, qty: int, tp: str = None, uuid: str = None):
        try:
            return self.api_session.place_order(
                category="linear",
                symbol=symbol,
                side="Buy",
                orderType="Market",
                qty=qty,
                orderLinkId=uuid
            )
        except Exception as api_err:
            err_split = api_err.args[0]
            err_split = err_split.split(")", 1)[0]
            if '110007' in err_split:
                return {'error': emoji.emojize(":ZZZ:") + " Нет денег на счету для следующей покупки", 'code': api_err.args[0]}
            return {'error': err_split, 'code': api_err.args[0]}


    def TakeProfit(self, order_dict):
        try:
           return self.api_session.set_trading_stop(
                category="linear",
                symbol="TONUSDT",
                takeProfit=str(order_dict['tp_price']),
                tpTriggerBy="MarkPrice",
                tpslMode="Partial",
                tpOrderType="Limit",
                tpSize=str(order_dict['qty']),
                tpLimitPrice=str(order_dict['tp_price']),
                positionIdx=0,
                orderLinkId=order_dict['uuid'],
                )
        except Exception as api_err:
            print(f"takeProfit={str(order_dict['tp_price'])}")
            print(f"tpSize={str(order_dict['qty'])}")
            return {'error': api_err}

    async def OrderHistory(self, orderId = None):
        try:
            get_history = self.api_session.get_order_history(
               category="linear",
               orderId=orderId,
               limit=1,
                )
            if not get_history['result']['list']:
                await asyncio.sleep(0)
                get_history = self.api_session.get_order_history(
                    category="linear",
                    orderId=orderId,
                    limit=1,
                )
            return get_history
        except Exception as api_err:
            return {'error': api_err}

    def getFeeRate(self, symbol: str):
        #print(f"order_dict {order_dict}\n")
        try:
            fee = self.api_session.get_fee_rates(
                symbol=str(symbol),
            )
            return {
                'takerFeeRate': fee['result']['list'][0]['takerFeeRate'],
                'makerFeeRate': fee['result']['list'][0]['makerFeeRate']
                }
        except Exception as api_err:
            print(f"\nGet fee EXCEPTION: {api_err.args}\n")


    def LastTakeProfitOrder(self, symbol: str, limit: int):
        #try:
        last_tp = self.api_session.get_open_orders(
            category="linear",
            symbol=symbol,
            limit=limit,
            )
        return last_tp
        #except Exception as api_err:
        #    print(f"\nLastTakeProfitOrder exception: {api_err.args}\n")
        #    return {'error': api_err.args}

    def get_kline(self, symbol: str, interval:str, limit: int):
        #try:
        kline = self.api_session.get_kline(
            category="linear",
            symbol=symbol,
            interval=interval,
            limit=limit
            )
        return kline
        #except Exception as api_err:
        #    print(f"\nLastTakeProfitOrder exception: {api_err.args}\n")
        #    return {'error': api_err.args}