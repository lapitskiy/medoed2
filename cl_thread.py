import threading
from fu_tools import *

#
# поток связанный с торговлей на бинанс
#

class StartTrade(threading.Thread):
    """StartTrade"""
    def __init__(self):
        threading.Thread.__init__(self)
        #self.trade = trade

    def run(self):
        self.start_trade()

    def start_trade(self):
        #print('==========GO Trade Thread==========')
        while True:
            start_trade_stg(stg='10point')




