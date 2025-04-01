from typing import Any
import json
import jsonpickle
from datamodel import Order, TradingState
from logger import Logger

logger = Logger()

def createMarketOrders(product, order_depth):
    orders = []

    buy_orders = order_depth.buy_orders
    sell_orders = order_depth.sell_orders


    maxBuy = max(buy_orders.keys())
    minSell = min(sell_orders.keys())

    midPrice = 10000

    #I used kinda aggressive buying and selling quantities at 15 but it never reaches the position limit so IDK
    if midPrice - maxBuy > 2:
        orders.append(Order(product, maxBuy+2, 15))
    
    if minSell - midPrice > 2:
        orders.append(Order(product, minSell-2, -15))

    return orders


class Trader:
    def run(self, state):
        result = {}

        product = 'RAINFOREST_RESIN'
        order_depth = state.order_depths[product]
        result[product] = createMarketOrders(product, order_depth)

        logger.flush(state, result, None, "")
        return result, None, ""