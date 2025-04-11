from typing import Any
import json
import jsonpickle
import math
import statistics
import numpy as np
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."

logger = Logger()

class Product:
    RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    INK = "SQUID_INK"
    DJEMBES = "DJEMBES"
    JAMS = "JAMS"
    CROISSANTS = "CROISSANTS"
    BASKET1 = "PICNIC_BASKET1" # = 6 CROISSANTS, 3 JAMS, 1 DJEMBES
    BASKET2 = "PICNIC_BASKET2" # = 4 CROISSANTS, 2 JAMS

products = [Product.RESIN, Product.KELP, Product.INK, Product.DJEMBES, Product.JAMS, Product.CROISSANTS, Product.BASKET1, Product.BASKET2]

class Trader:
    def __init__(self):        
        self.LIMIT = {
            Product.RESIN: 50,
            Product.KELP: 50,
            Product.INK: 50,
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
            Product.BASKET1: 60,
            Product.BASKET2: 100
        }
        
        self.window_size = 20
        
        self.prices = {
            Product.RESIN: [10000],
            Product.KELP: [2033],
            Product.INK: [1904],
            Product.CROISSANTS: [13436],
            Product.JAMS: [4298],
            Product.DJEMBES: [6593],
            Product.BASKET1: [59052],
            Product.BASKET2: [30408]
        }
        
        self.prev_ema = { p: 0.0 for p in products }
        
        self.spread_history = []
    
    def calculate_mid_price(self, buy_orders, sell_orders):
        if not buy_orders or not sell_orders:
            return 0.0
        
        # get best bid and ask prices
        best_bid = max(buy_orders.keys())
        best_ask = min(sell_orders.keys())
        
        # calculate mid price
        mid_price = (best_bid + best_ask) / 2
        logger.print(f"Best Bid: {best_bid}, Best Ask: {best_ask}, Mid Price: {mid_price}")
        
        return mid_price        
    
    def calculate_fair_price(self, buy_orders, sell_orders):
        avg_buy = statistics.mean(buy_orders.keys()) if buy_orders else 0.0
        avg_sell = statistics.mean(sell_orders.keys()) if sell_orders else 0.0
        
        # take midpoint of buy and sell averages
        fair_price = (avg_buy + avg_sell) / 2
        logger.print(f"Avg Buy: {avg_buy}, Avg Sell: {avg_sell}, Fair Price: {fair_price}")
        
        return fair_price
    
    def calculate_vwap_price(self, buy_orders, sell_orders):
        total_volume = 0
        total_value = 0
        
        for price, volume in sell_orders.items():
            total_volume += volume
            total_value += price * volume
        
        for price, volume in buy_orders.items():
            total_volume -= volume
            total_value -= price * volume
        
        if total_volume == 0:
            return 0
        
        vwap = total_value / total_volume
        logger.print(f"VWAP: {vwap}")
        
        return vwap

    def calculate_volatility(self, prices):
        if len(prices) < 2:
            return 0.0
        
        mean_price = sum(prices) / len(prices)
        variance = sum((price - mean_price) ** 2 for price in prices) / (len(prices) - 1)
        
        return math.sqrt(variance)

    # gamma is the risk aversion coefficient; < 0.1 for less volatile, > 0.5 for more volatile
    def calculate_rpf(self, product, mid_price, position, gamma, timestamp):
        time_left = 1000000 - (timestamp % 1000000) # T-t
        variance = statistics.variance(self.prices[product]) if len(self.prices[product]) >= 2 else 0 # variance
        
        rpf = mid_price - position*gamma*variance*time_left
        logger.print(f"rpf: {rpf}")
        
        return rpf

    def take_best_orders(self, product, orders, order_depth, fair_price, position, buy_order_volume, sell_order_volume):
            logger.print("-- Taking best orders --")
            
            if len(order_depth.sell_orders) != 0:
                best_ask = min(order_depth.sell_orders.keys())
                best_ask_amount = -order_depth.sell_orders[best_ask]
                
                if best_ask <= fair_price:
                    quantity = min(best_ask_amount, self.LIMIT[product] - position) # max amount to buy 
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity)) 
                        logger.print(f"BUY {quantity} @ {best_ask}")
                        buy_order_volume += quantity

            if len(order_depth.buy_orders) != 0:
                best_bid = max(order_depth.buy_orders.keys())
                best_bid_amount = order_depth.buy_orders[best_bid]
                if best_bid >= fair_price:
                    quantity = min(best_bid_amount, self.LIMIT[product] + position) # max we can sell 
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        logger.print(f"SELL {quantity} @ {best_bid}")
                        sell_order_volume += quantity
            
            return buy_order_volume, sell_order_volume
    
    def mean_reversion(self, product, orders, order_depth, fair_price, position, buy_order_volume, sell_order_volume):
        logger.print("-- Mean Reversion --")
        mean = statistics.mean(self.prices[product])
        std = statistics.stdev(self.prices[product])
        z = (fair_price - mean) / std if std > 0 else 0

        if z < -1:
            # Price too low → buy
            for ask, ask_qty in sorted(order_depth.sell_orders.items()):
                qty = min(-ask_qty, self.LIMIT[product] - position)
                if qty > 0:
                    orders.append(Order(product, ask, qty))
                    logger.print(f"BUY {qty} @ {ask}")
                    buy_order_volume += qty
                    break
        elif z > 1:
            # Price too high → sell
            for bid, bid_qty in sorted(order_depth.buy_orders.items(), reverse=True):
                qty = -min(bid_qty, self.LIMIT[product] + position)
                if qty > 0:
                    orders.append(Order(product, bid, qty))
                    logger.print(f"SELL {qty} @ {bid}")
                    sell_order_volume += qty
                    break

        return buy_order_volume, sell_order_volume
    
    # basic univariate linear regression model between price and time
    def basic_lr(self, product, threshold, orders, order_depth, position, buy_order_volume, sell_order_volume):
        logger.print("-- LR: Price vs Time --")
        
        # estalish regression between the following
        prices = self.prices[product]
        times = np.arange(len(prices))

        # compute LR
        slope, intercept = np.polyfit(times, prices, 1)
        predicted = slope * times + intercept
        residual = prices[-1] - predicted[-1] # current deviation from trend
        
        logger.print(predicted)
        
        # buy signal
        if residual < -threshold:
            for ask, ask_qty in sorted(order_depth.sell_orders.items()):
                buy_qty = min(-ask_qty, self.LIMIT[product] - position)
                if buy_qty > 0:
                    orders.append(Order(product, ask, buy_qty))
                    logger.print(f"BUY {buy_qty} @ {ask}")
                    buy_order_volume += buy_qty
                    break
        
        # sell signal
        elif residual > threshold:
            for bid, bid_qty in sorted(order_depth.buy_orders.items(), reverse=True):
                ask_qty = min(bid_qty, self.LIMIT[product] + position)
                if ask_qty > 0:
                    orders.append(Order(product, bid, -ask_qty))
                    logger.print(f"SELL {ask_qty} @ {bid}")
                    sell_order_volume += ask_qty
                    break
        
        return buy_order_volume, sell_order_volume
    
    # pairwise linear regression model between two products
    # if product1 is overpriced vs product2, short product1 and long product2
    # if product1 is underpriced vs product2, long product1 and short product2
    def pairwise_lr(self, state: TradingState, threshold, product1, product2, orders1, orders2, pos1, pos2, buy_order_volume, sell_order_volume):
        logger.print("-- Pairwise LR --")
        
        prices1 = self.prices[product1]
        prices2 = self.prices[product2]
        
        max_pos1 = self.LIMIT[product1] + pos1
        max_pos2 = self.LIMIT[product2] + pos2
        
        order_depth1 = state.order_depths[product1]
        order_depth2 = state.order_depths[product2]

        slope, intercept = np.polyfit(prices1, prices2, 1)
        expected_p2 = slope * prices1[-1] + intercept
        residual = prices2[-1] - expected_p2
        
        logger.print(f"Expected {product2}: {expected_p2}, Actual: {prices2[-1]}, Residual: {residual}")        
        
        if residual > threshold:
            # Product2 is OVERVALUED → sell product2, buy product1
            logger.print(f"[Spread Trade] {product2} OVERVALUED → SELL {product2}, BUY {product1}")

            # Sell product2 (overpriced)
            for bid, bid_qty in sorted(order_depth2.buy_orders.items(), reverse=True):
                sell_qty = min(bid_qty, max_pos2 + pos2 - sell_order_volume)
                if sell_qty > 0:
                    orders1.append(Order(product2, bid, -sell_qty))
                    logger.print(f"SELL {sell_qty} @ {bid}")
                    sell_order_volume += sell_qty
                    break

            # Buy product1 (undervalued)
            for ask, ask_qty in sorted(order_depth1.sell_orders.items()):
                buy_qty = min(-ask_qty, max_pos1 - pos1 - buy_order_volume)
                if buy_qty > 0:
                    orders1.append(Order(product1, ask, buy_qty))
                    logger.print(f"BUY {buy_qty} @ {ask}")
                    buy_order_volume += buy_qty
                    break

        elif residual < -threshold:
            # Product2 is UNDERVALUED → buy product2, sell product1
            logger.print(f"[Spread Trade] {product2} UNDERVALUED → BUY {product2}, SELL {product1}")

            # Buy product2
            for ask, ask_qty in sorted(order_depth2.sell_orders.items()):
                buy_qty = min(-ask_qty, max_pos2 - pos2 - buy_order_volume)
                if buy_qty > 0:
                    orders2.append(Order(product2, ask, buy_qty))
                    buy_order_volume += buy_qty
                    break

            # Sell product1
            for bid, bid_qty in sorted(order_depth1.buy_orders.items(), reverse=True):
                sell_qty = min(bid_qty, max_pos1 + pos1 - sell_order_volume)
                if sell_qty > 0:
                    orders2.append(Order(product1, bid, -sell_qty))
                    sell_order_volume += sell_qty
                    break

        return buy_order_volume, sell_order_volume
    
    def stat_arb(self, state, threshold, product1, product2, orders1, orders2, pos1, pos2, buy_order_volume, sell_order_volume):
        logger.print("-- Statistical Arbitrage --")
        
        price1 = self.prices[product1]
        price2 = self.prices[product2]

        # Calculate spread
        # slope, intercept = np.polyfit(price1[-30:], price2[-30:], 1)
        # expected_p2 = slope * price1[-1] + intercept
        # spread = price2[-1] - expected_p2
        spread = price1[-1] - price2[-1]
        
        # Add spread price to list
        self.spread_history.append(spread)
            
        spread_window = 30
        if len(self.spread_history) < spread_window:
            return buy_order_volume, sell_order_volume

        # Calculate mean and standard deviation of the spread
        mean = np.mean(self.spread_history[-30:])
        std = np.std(self.spread_history[-30:])

        # Calculate z-score
        z = (spread - mean) / std if std > 0 else 0
        logger.print(f"Spread: {spread}, Mean: {mean}, Std: {std}, Z-Score: {z}")

        order_depth1 = state.order_depths[product1]
        order_depth2 = state.order_depths[product2]

        # If z-score exceeds threshold, execute trades
        if z > threshold:
            # product1 is too expensive → sell product1, buy product2
            logger.print(f"[Stat Arb] {product1} OVERVALUED → SELL {product1}, BUY {product2}")

            # Sell product1
            for bid, bid_qty in sorted(order_depth1.buy_orders.items(), reverse=True):
                sell_qty = min(bid_qty, self.LIMIT[product1] + pos1 - sell_order_volume)
                if sell_qty > 0:
                    orders1.append(Order(product1, bid, -sell_qty))
                    logger.print(f"SELL {sell_qty} @ {bid}")
                    sell_order_volume += sell_qty
                    break

            # Buy product2
            for ask, ask_qty in sorted(order_depth2.sell_orders.items()):
                buy_qty = min(-ask_qty, self.LIMIT[product2] - pos2 - buy_order_volume)
                if buy_qty > 0:
                    orders2.append(Order(product2, ask, buy_qty))
                    logger.print(f"BUY {buy_qty} @ {ask}")
                    buy_order_volume += buy_qty
                    break

        elif z < -threshold:
            # product2 is too expensive → sell product2, buy product1
            logger.print(f"[Stat Arb] {product2} OVERVALUED → SELL {product2}, BUY {product1}")

            # Sell product2
            for bid, bid_qty in sorted(order_depth2.buy_orders.items(), reverse=True):
                sell_qty = min(bid_qty, self.LIMIT[product2] + pos2 - sell_order_volume)
                if sell_qty > 0:
                    orders2.append(Order(product2, bid, -sell_qty))
                    logger.print(f"SELL {sell_qty} @ {bid}")
                    sell_order_volume += sell_qty
                    break

            # Buy product1
            for ask, ask_qty in sorted(order_depth1.sell_orders.items()):
                buy_qty = min(-ask_qty, self.LIMIT[product1] - pos1 - buy_order_volume)
                if buy_qty > 0:
                    orders1.append(Order(product1, ask, buy_qty))
                    logger.print(f"BUY {buy_qty} @ {ask}")
                    buy_order_volume += buy_qty
                    break

        return buy_order_volume, sell_order_volume

    def obi(self, product, threshold, orders, order_depth, position, buy_order_volume, sell_order_volume):
        logger.print("-- OBI --")
        
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return buy_order_volume, sell_order_volume

        # compute buy/sell volume at all levels (or limit to top N)
        buy_volume = sum(order_depth.buy_orders.values())
        sell_volume = -sum(order_depth.sell_orders.values())

        obi = (buy_volume - sell_volume) / (buy_volume + sell_volume) if (buy_volume + sell_volume) > 0 else 0
        logger.print(f"{product} OBI: {obi:.2f} (Buy Vol: {buy_volume}, Sell Vol: {sell_volume})")

        # buy signal
        if obi > threshold:
            for ask, ask_qty in sorted(order_depth.sell_orders.items()):
                buy_qty = min(-ask_qty, self.LIMIT[product] - position)
                if buy_qty > 0:
                    orders.append(Order(product, ask, buy_qty))
                    buy_order_volume += buy_qty
                    logger.print(f"BUY {buy_qty} @ {ask}")
                    break

        # sell signal
        elif obi < -threshold:
            for bid, bid_qty in sorted(order_depth.buy_orders.items(), reverse=True):
                ask_qty = min(bid_qty, self.LIMIT[product] + position)
                if ask_qty > 0:
                    orders.append(Order(product, bid, -ask_qty))
                    sell_order_volume += ask_qty
                    logger.print(f"SELL {ask_qty} @ {bid}")
                    break

        return buy_order_volume, sell_order_volume
    
    def ema(self, product, alpha, orders, order_depth, fair_price, position, buy_order_volume, sell_order_volume):
        logger.print("-- EMA --")

        # calculate EMA
        if len(self.prices[product]) == 1:
            ema = fair_price # initialize that jawn
        else:
            if len(self.prices[product]) > self.window_size:
                self.prices[product].pop(0)
            
            # calculate EMA using the previous EMA value
            ema = alpha * fair_price + (1 - alpha) * self.prev_ema[product]

        self.prev_ema[product] = ema

        logger.print(f"Current EMA: {ema}, Mid-Price: {fair_price}")

        # price below EMA → Buy signal
        if fair_price < ema:
            logger.print(f"[Buy signal] Mid-Price: {fair_price}")
            # find sell orders that are profitable to buy
            for ask, ask_quantity in sorted(order_depth.sell_orders.items()):
                logger.print(f"Ask: {ask}, Ask Quantity: {ask_quantity}")
                # ensure good entry price
                buy_qty = min(abs(ask_quantity), self.LIMIT[product] - position - buy_order_volume)
                if ask < fair_price and buy_qty > 0:
                    orders.append(Order(product, ask, buy_qty))
                    logger.print(f"BUY {-ask_quantity} @ {ask}")
                    buy_order_volume += abs(ask_quantity)

        # price above EMA → Sell signal
        elif fair_price > ema:
            logger.print(f"[Sell signal] Mid-Price: {fair_price}")
            # find buy orders that are profitable to sell
            for bid, bid_quantity in sorted(order_depth.buy_orders.items(), reverse=True):
                logger.print(f"Bid: {bid}, Bid Quantity: {bid_quantity}")
                # ensure profitable exit
                sell_qty = min(bid_quantity, self.LIMIT[product] + position - sell_order_volume)
                if bid > fair_price and sell_qty > 0:
                    logger.print(f"SELL {sell_qty} @ {bid}")
                    orders.append(Order(product, bid, -sell_qty))
                    sell_order_volume += sell_qty

        return buy_order_volume, sell_order_volume
    
    # a risk management function that neutralizes or reduces the current position 
    def clear_position_order(self, product, orders, order_depth, fair_value, position, buy_order_volume, sell_order_volume):
        logger.print("-- Clearing position --")
        new_position = position + buy_order_volume - sell_order_volume
        
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)

        # how much we can buy/sell without breaching position limits
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if new_position > 0 and fair_for_ask in order_depth.buy_orders.keys():
            clear_quantity = min(order_depth.buy_orders[fair_for_ask], new_position)
            sent_quantity = min(sell_quantity, clear_quantity)
            orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
            logger.print(f"SELL {sent_quantity} @ {fair_for_ask}")
            sell_order_volume += abs(sent_quantity)

        if new_position < 0 and fair_for_bid in order_depth.sell_orders.keys():
            clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(new_position))
            sent_quantity = min(buy_quantity, clear_quantity)
            orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
            logger.print(f"BUY {sent_quantity} @ {fair_for_bid}")
            buy_order_volume += abs(sent_quantity)
    
        return buy_order_volume, sell_order_volume
    
    def market_make(self, product, orders, order_depth, fair_price, position, buy_order_volume, sell_order_volume):
        logger.print("-- Market making --")
        
        # calculate best bid and ask
        filtered_bids = [p for p in order_depth.buy_orders.keys() if p < fair_price - 1]
        filtered_asks = [p for p in order_depth.sell_orders.keys() if p > fair_price + 1]
        
        bid = max(filtered_bids) + 1 if len(filtered_bids) > 0 else max(order_depth.buy_orders.keys())
        ask = min(filtered_asks) - 1 if len(filtered_asks) > 0 else min(order_depth.sell_orders.keys())

        # make buy order
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, bid, buy_quantity))
            logger.print(f"BUY {buy_quantity} @ {bid}")

        # make sell order
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, ask, -sell_quantity))
            logger.print(f"SELL {sell_quantity} @ {ask}")
        
        return buy_order_volume, sell_order_volume
    
    def run(self, state: TradingState):        
        conversions = None
        result = { p: [] for p in products }
        
        # load trader data
        if state.traderData:
            trader_data = jsonpickle.decode(state.traderData)
            self.prices = trader_data["prices"]
            self.prev_ema = trader_data["prev_ema"]
            self.spread_history = trader_data["spread_history"]
        
        djembes_price = self.calculate_mid_price(
            state.order_depths[Product.DJEMBES].buy_orders,
            state.order_depths[Product.DJEMBES].sell_orders
        ) if Product.DJEMBES in state.order_depths else self.prices[Product.DJEMBES][-1]
        
        croissants_price = self.calculate_mid_price(
            state.order_depths[Product.CROISSANTS].buy_orders,
            state.order_depths[Product.CROISSANTS].sell_orders
        ) if Product.CROISSANTS in state.order_depths else self.prices[Product.CROISSANTS][-1]
        
        jams_price = self.calculate_mid_price(
            state.order_depths[Product.JAMS].buy_orders,
            state.order_depths[Product.JAMS].sell_orders
        ) if Product.JAMS in state.order_depths else self.prices[Product.JAMS][-1]
        
        basket1_price = self.calculate_mid_price(
            state.order_depths[Product.BASKET1].buy_orders,
            state.order_depths[Product.BASKET1].sell_orders
        )
        
        basket2_price = self.calculate_mid_price(
            state.order_depths[Product.BASKET2].buy_orders,
            state.order_depths[Product.BASKET2].sell_orders
        ) if Product.BASKET2 in state.order_depths else self.prices[Product.BASKET2][-1]
        
        self.prices[Product.DJEMBES].append(djembes_price)
        self.prices[Product.CROISSANTS].append(croissants_price)
        self.prices[Product.JAMS].append(jams_price)
        self.prices[Product.BASKET1].append(basket1_price)
        self.prices[Product.BASKET2].append(basket2_price)
        
        # ensure available liquidity
        if (
            Product.CROISSANTS in state.order_depths
            and Product.JAMS in state.order_depths
            and Product.DJEMBES in state.order_depths
        ):
            ### BASKET 1 ARBITRAGE ###
            if Product.BASKET1 in state.order_depths:
                threshold = 100
                
                # compute price of synthetic basket
                synthetic_basket1_price = 6 * croissants_price + 3 * jams_price + djembes_price
                spread = basket1_price - synthetic_basket1_price
                logger.print(f"Basket1 Price: {basket1_price}, Synthetic Basket1 Price: {synthetic_basket1_price}, Spread: {spread}")
                
                # +spread -> basket1 is overpriced
                if spread > threshold:
                    # compute bids and asks
                    basket1_bid = max(state.order_depths[Product.BASKET1].buy_orders.keys())
                    croissants_ask = min(state.order_depths[Product.CROISSANTS].sell_orders.keys())
                    jams_ask = min(state.order_depths[Product.JAMS].sell_orders.keys())
                    djembes_ask = min(state.order_depths[Product.DJEMBES].sell_orders.keys())
                    
                    # sell basket1 at bid
                    result[Product.BASKET1].append(Order(Product.BASKET1, basket1_bid, -1))
                    
                    # and buy components at ask
                    result[Product.CROISSANTS].append(Order(Product.CROISSANTS, croissants_ask, 6))
                    result[Product.JAMS].append(Order(Product.JAMS, jams_ask, 3))
                    result[Product.DJEMBES].append(Order(Product.DJEMBES, djembes_ask, 1))
                
                # -spread -> basket1 is underpriced
                elif spread < -threshold:
                    # compute bids and asks
                    basket1_ask = min(state.order_depths[Product.BASKET1].sell_orders.keys())
                    croissants_bid = max(state.order_depths[Product.CROISSANTS].buy_orders.keys())
                    jams_bid = max(state.order_depths[Product.JAMS].buy_orders.keys())
                    djembes_bid = max(state.order_depths[Product.DJEMBES].buy_orders.keys())
                    
                    # buy basket1 at ask
                    result[Product.BASKET1].append(Order(Product.BASKET1, basket1_ask, 1))
                    
                    # sell components at bid
                    result[Product.CROISSANTS].append(Order(Product.CROISSANTS, croissants_bid, -6))
                    result[Product.JAMS].append(Order(Product.JAMS, jams_bid, -3))
                    result[Product.DJEMBES].append(Order(Product.DJEMBES, djembes_bid, -1))
            
            ### BASKET 2 ARBITRAGE ###
            if Product.BASKET2 in state.order_depths:
                threshold = 75
                
                synthetic_basket2_price = 4 * croissants_price + 2 * jams_price
                spread = basket2_price - synthetic_basket2_price
                logger.print(f"Basket2 Price: {basket2_price}, Synthetic Basket2 Price: {synthetic_basket2_price}, Spread: {spread}")
                
                # +spread -> basket2 is overpriced
                if spread > threshold:
                    # compute bids and asks
                    basket2_bid = max(state.order_depths[Product.BASKET2].buy_orders.keys())
                    croissants_ask = min(state.order_depths[Product.CROISSANTS].sell_orders.keys())
                    jams_ask = min(state.order_depths[Product.JAMS].sell_orders.keys())
                    
                    # sell basket2 at bid 
                    result[Product.BASKET2].append(Order(Product.BASKET2, basket2_bid, -1))
                    
                    # buy synthetic basket at ask
                    result[Product.CROISSANTS].append(Order(Product.CROISSANTS, croissants_ask, 4))
                    result[Product.JAMS].append(Order(Product.JAMS, jams_ask, 2))
                
                # -spread -> basket2 is underpriced
                elif spread < -threshold:
                    # compute bids and asks
                    basket2_ask = min(state.order_depths[Product.BASKET2].sell_orders.keys())
                    croissants_bid = max(state.order_depths[Product.CROISSANTS].buy_orders.keys())
                    jams_bid = max(state.order_depths[Product.JAMS].buy_orders.keys())
                    
                    # buy basket2 at ask
                    result[Product.BASKET2].append(Order(Product.BASKET2, basket2_ask, 1))
                    
                    # sell components at bid
                    result[Product.CROISSANTS].append(Order(Product.CROISSANTS, croissants_bid, -4))
                    result[Product.JAMS].append(Order(Product.JAMS, jams_bid, -2))
        
        # update trader data
        trader_data = jsonpickle.encode({
            "prices": self.prices,
            "prev_ema": self.prev_ema,
            "spread_history": self.spread_history
        })
        
        logger.flush(state, result, conversions, trader_data)
        
        return result, conversions, trader_data
