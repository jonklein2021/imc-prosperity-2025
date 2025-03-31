from typing import Any
import json
import jsonpickle
from datamodel import Order, TradingState
from logger import Logger

logger = Logger()

class Trader:
    def __init__(self):
        self.window_size = 5  # Number of past prices to consider for SMA
        
        # buffer factor
        # < 1 to allow for more trades
        # > 1 to allow for less trades
        self.flex = 1
        
        self.mid_prices = {
            "RAINFOREST_RESIN": [],
            "KELP": []
        }
        
    def sma(self, state: TradingState, product):
            orders = []
            order_depth = state.order_depths[product]
            
            logger.print(f"=== {product} ===")
            
            # calculate best bid and ask
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else float("inf")
            
            logger.print(f"Best Bid: {best_bid}, Best Ask: {best_ask}")
            
            # calculate mid price
            if order_depth.buy_orders and order_depth.sell_orders:
                mid_price = (best_bid + best_ask) / 2
            elif order_depth.buy_orders:
                mid_price = best_bid  # use bid price if no asks available
            elif order_depth.sell_orders:
                mid_price = best_ask  # use ask price if no bids available
            else:
                logger.print("No available prices.")
                return []
            
            # add mid price to list
            self.mid_prices[product].append(mid_price)
            if len(self.mid_prices[product]) > self.window_size:
                self.mid_prices[product].pop(0)
                
            if state.timestamp // 100 < self.window_size:
                logger.print(f"{state.timestamp // 100} < {self.window_size}; waiting...")
                return []
            
            # calculate sma
            if len(self.mid_prices[product]) == self.window_size:
                sma = sum(self.mid_prices[product]) / self.window_size
                logger.print(f"Current SMA: {sma}, Mid-Price: {mid_price}")
            else:
                # not enough prices to calculate SMA, wait until we have some
                logger.print(self.mid_prices)
                logger.print("Not enough prices to calculate SMA, waiting...")
                return []
            
            # price below SMA → Buy signal
            # find sell orders that are profitable to buy
            if mid_price * self.flex < sma:
                logger.print(f"[Buy signal] Mid-Price: {mid_price}")
                for ask, ask_quantity in order_depth.sell_orders.items():
                    logger.print(f"Ask: {ask}, Ask Quantity: {ask_quantity}")
                    # ensure good entry price
                    if ask * self.flex < mid_price:
                        logger.print(f"BUY {ask_quantity} @ {ask}")
                        orders.append(Order(product, ask, -ask_quantity))

            # price above SMA → Sell signal
            # find buy orders that are profitable to sell
            elif mid_price > sma * self.flex:
                logger.print(f"[Sell signal] Mid-Price: {mid_price}")
                for bid, bid_quantity in order_depth.buy_orders.items():
                    logger.print(f"Bid: {bid}, Bid Quantity: {bid_quantity}")
                    # ensure profitable exit
                    if bid > mid_price * self.flex:
                        logger.print(f"SELL {bid_quantity} @ {bid}")
                        orders.append(Order(product, bid, -bid_quantity))
            
            return orders
    
    def run(self, state: TradingState):
        traderData = jsonpickle.encode({
            "RAINFOREST_RESIN": [],
            "KELP": []
        })
        
        conversions = None
        result = {}
        
        # get past prices from traderData
        if len(state.traderData) != 0:
            self.mid_prices = jsonpickle.decode(state.traderData)
            logger.print("mid_prices: " + str(self.mid_prices))
        
        # apply SMA to each product
        for product in state.listings.keys():
            result[product] = self.sma(state, product)
        
        # log mid prices for next iteration
        traderData = jsonpickle.encode(self.mid_prices)
        
        logger.flush(state, result, conversions, traderData)
        
        return result, conversions, traderData
