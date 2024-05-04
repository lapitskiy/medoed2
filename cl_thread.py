import asyncio
import time
import threading
from contextlib import closing
from sqlalchemy.orm import Session, sessionmaker
from models import Strategy


#
# поток связанный с торговлей на бинанс
#
from stg_router import getStgObjFromClass
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
        obj_state = {}  # Словарь для отслеживания объектов
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.start_trade(obj_state))
        finally:
            loop.close()

    async def start_trade(self, obj_state):
        while True:
            # start all strategy in loop
            session = create_session()
            query = session.query(Strategy).all()
            for stg in query:
                if stg.start and stg.stg_name and stg.id not in obj_state:
                    stgObj = getStgObjFromClass(stg_id=stg.id)
                    obj_state[stg.id] = stgObj
                    asyncio.create_task(self.process_and_cleanup(stgObj, obj_state, stg.id))
                else:
                    pass
                    #print(f"Object {stg.id}  - {obj_state} is already being processed.")
                await asyncio.sleep(0)
            session.close()

    async def process_and_cleanup(self, stgObj, obj_state, stg_id):
        await stgObj.Start()
        del obj_state[stg_id]








