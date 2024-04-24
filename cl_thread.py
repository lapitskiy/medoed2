import time
import threading
from contextlib import closing
from sqlalchemy.orm import Session, sessionmaker
from models import Strategy


#
# поток связанный с торговлей на бинанс
#
from strategy import getStgObjFromClass
from utils_db import getEngine



def create_session():
    Session = sessionmaker(getEngine())
    return Session()

class StartTrade(threading.Thread):

    """StartTrade"""
    def __init__(self):
        threading.Thread.__init__(self)
        #self.trade = trade

    def run(self):
        self.start_trade()

    def start_trade(self):
        while True:
            self.checkTradeStg()

    def checkTradeStg(self):
        session = create_session()
        query = session.query(Strategy).all()
        for stg in query:
            if stg.start and stg.stg_name:
                pass
                stgObj = getStgObjFromClass(stg_id=stg.id)
                stgObj.Start()
                print('tut')
        session.close()







