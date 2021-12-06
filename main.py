import os

from ib_insync import *

from models.hft_model_1 import HftModel1
from models.stock_model_1 import HftStockModel1

if __name__ == '__main__':
	TWS_HOST = os.environ.get('TWS_HOST', '127.0.0.1')
	TWS_PORT = os.environ.get('TWS_PORT', 7497)

	print('Connecting on host:', TWS_HOST, 'port:', TWS_PORT)

	'''model = HftModel1(
		host=TWS_HOST,
		port=TWS_PORT,
		client_id=2,
	)'''
	model = HftStockModel1(
		host=TWS_HOST,
		port=TWS_PORT,
		client_id=2,
	)

	to_trade = [
		('SNAP', Stock('SNAP'))
	]

	'''to_trade = [
		('EURUSD', Forex('EURUSD')),
		('USDJPY', Forex('USDJPY'))
	]'''

	model.run(to_trade=to_trade, trade_qty=10)
