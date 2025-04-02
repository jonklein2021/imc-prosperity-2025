from typing import Any
import json
import jsonpickle
from datamodel import Order, TradingState
from logger import Logger

logger = Logger()

class Trader:
    def __init__(self):
        self.mid_prices = {
            "RAINFOREST_RESIN": 10000,
            "KELP": []
        }


    def sma(self, state: TradingState, product, window_size):
        order_depth = state.order_depths[product]
        
        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders
        
        # calculate best bid and ask
        best_bid = max(buy_orders.keys()) if buy_orders else 0
        best_ask = min(sell_orders.keys()) if sell_orders else float("inf")
        
        logger.print(f"Best Bid: {best_bid}, Best Ask: {best_ask}")
        
        # calculate mid price
        if buy_orders and sell_orders:
            mid_price = (best_bid + best_ask) / 2
        elif buy_orders:
            mid_price = best_bid  # use bid price if no asks available
        elif sell_orders:
            mid_price = best_ask  # use ask price if no bids available
        else:
            logger.print("No available prices.")
            return 
        
        # add mid price to list
        self.mid_prices[product].append(mid_price)
        if len(self.mid_prices[product]) > window_size:
            self.mid_prices[product].pop(0)
            
        if state.timestamp // 100 < window_size:
            logger.print(f"{state.timestamp // 100} < {window_size}; waiting...")
            return 0
        
        # calculate sma
        if len(self.mid_prices[product]) == window_size:
            sma = sum(self.mid_prices[product]) / window_size
            logger.print(f"Current SMA: {sma}, Mid-Price: {mid_price}")
            return sma
        else:
            # not enough prices to calculate SMA, wait until we have some
            logger.print(self.mid_prices)
            logger.print("Not enough prices to calculate SMA, waiting...")
            return 0

    def createResinOrders(self, state: TradingState):
        product = 'RAINFOREST_RESIN'
        orders = []

        order_depth = state.order_depths[product]
        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders

        maxBuy = max(buy_orders.keys())
        minSell = min(sell_orders.keys())

        #I used kinda aggressive buying and selling quantities at 15 but it never reaches the position limit so IDK
        midPrice = self.mid_prices[product]

        if midPrice - maxBuy > 2:
            orders.append(Order(product, maxBuy+2, 15))
        
        if minSell - midPrice > 2:
            orders.append(Order(product, minSell-2, -15))

        return orders

    def createKelpOrders(self, state: TradingState):
        product = 'KELP'
        orders = []

        order_depth = state.order_depths[product]
        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders

        maxBuy = max(buy_orders.keys())
        minSell = min(sell_orders.keys())

        sma = self.sma(state, product, 6)
        if sma == 0:
            return []

        if sma - maxBuy > 2:
            orders.append(Order(product, maxBuy+1, 4))
        
        if minSell - sma > 2:
            orders.append(Order(product, minSell-1, -4))
        return orders
        

    def run(self, state):
        result = {}

        result['RAINFOREST_RESIN'] = self.createResinOrders(state)
        result['KELP'] = self.createKelpOrders(state)

        logger.flush(state, result, None, "")
        return result, None, ""