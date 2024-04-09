import time
import threading
from contextlib import closing
from sqlalchemy.orm import Session
from models import Strategy


#
# поток связанный с торговлей на бинанс
#
from strategy import getStgObjFromClass
from utils_db import getEngine


class StartTrade(threading.Thread):
    """StartTrade"""
    def __init__(self):
        threading.Thread.__init__(self)
        #self.trade = trade

    def run(self):
        self.start_trade()

    def start_trade(self):
        while True:
            time.sleep(1)
            self.checkTradeStg()

    def checkTradeStg(self):
        with closing(Session(getEngine())) as session:
            query = session.query(Strategy).all()
            for stg in query:
                if stg.start and stg.stg_name:
                    stgObj = getStgObjFromClass(stg_id=stg.id)
                    stgObj.Start()







