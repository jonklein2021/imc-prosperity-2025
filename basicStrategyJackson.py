from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

class Trader:
    def run(self, state: TradingState):
        
        """
        print(f"state : {state}")
        print(f"traderData : {state.traderData}")
        print(f"timestamp : {state.timestamp}")
        print(f"listings : {state.listings}")
        print(f"own_trades : {state.own_trades}")
        print(f"market_trades : {state.market_trades}")
        print(f"position : {state.position}")
        print(f"observations : {state.observations}")
        """

        #print(f"order_depths : {state.order_depths['RAINFOREST_RESIN']}")
        #print(state.toJSON())
        result = {}
        product = 'RAINFOREST_RESIN'
        result[product] = []

        midPrice = 10000

        order_depth = state.order_depths[product]
        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders

        for buy_order in buy_orders:
            if buy_order > midPrice:
                result[product].append(Order(product, buy_order, -buy_orders[buy_order]))

        for sell_order in sell_orders:
            if sell_order < midPrice:
                result[product].append(Order(product, sell_order, -sell_orders[sell_order]))

        
        conversions = 0   
        traderData = "Sample"
        return result, conversions, traderData
