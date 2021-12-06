import os

from ib_insync import Forex, Stock

from models.hft_model_1 import HftModel1
from models.stock_model_1 import HftStockModel1

if __name__ == '__main__':
	TWS_HOST = os.environ.get('TWS_HOST', '127.0.0.1')
	TWS_PORT = os.environ.get('TWS_PORT', 7497)

	print('Connecting on host:', TWS_HOST, 'port:', TWS_PORT)

	model = HftStockModel1(
		host=TWS_HOST,
		port=TWS_PORT,
	)

	to_trade = [
		('SNAP', Stock('SNAP', 'SMART', 'USD'))
	]

	model.run(to_trade=to_trade, trade_qty=100)
