from typing import Any
import json
import jsonpickle
import math
import statistics
import numpy as np
from datamodel import Listing, ConversionObservation, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

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

class BlackScholes:
    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        d1 = (math.log(spot) - math.log(strike) + (0.5 * volatility * volatility) * time_to_expiry) / (volatility * math.sqrt(time_to_expiry))
        d2 = d1 - volatility * math.sqrt(time_to_expiry)
        call_price = spot * statistics.NormalDist().cdf(d1) - strike * statistics.NormalDist().cdf(d2)
        return call_price

    @staticmethod
    def black_scholes_put(spot, strike, time_to_expiry, volatility):
        d1 = (math.log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry) / (volatility * math.sqrt(time_to_expiry))
        d2 = d1 - volatility * math.sqrt(time_to_expiry)
        put_price = strike * statistics.NormalDist().cdf(-d2) - spot * statistics.NormalDist().cdf(-d1)
        return put_price

    @staticmethod
    def delta(spot, strike, time_to_expiry, volatility):
        d1 = (math.log(spot) - math.log(strike) + (0.5 * volatility * volatility) * time_to_expiry) / (volatility * math.sqrt(time_to_expiry))
        return statistics.NormalDist().cdf(d1)

    @staticmethod
    def gamma(spot, strike, time_to_expiry, volatility):
        d1 = (math.log(spot) - math.log(strike) + (0.5 * volatility * volatility) * time_to_expiry) / (volatility * math.sqrt(time_to_expiry))
        return statistics.NormalDist().pdf(d1) / (spot * volatility * math.sqrt(time_to_expiry))

    @staticmethod
    def vega(spot, strike, time_to_expiry, volatility):
        d1 = (math.log(spot) - math.log(strike) + (0.5 * volatility * volatility) * time_to_expiry) / (volatility * math.sqrt(time_to_expiry))
        return statistics.NormalDist().pdf(d1) * (spot * math.sqrt(time_to_expiry)) / 100

    @staticmethod
    def implied_volatility(call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10):
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0  # Initial guess as the midpoint
        # binary search ts
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(spot, strike, time_to_expiry, volatility)
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility

class Product:
    RESIN = 0
    KELP = 1
    INK = 2
    DJEMBES = 3
    JAMS = 4
    CROISSANTS = 5
    BASKET1 = 6 # = 6 CROISSANTS, 3 JAMS, 1 DJEMBES
    BASKET2 = 7 # = 4 CROISSANTS, 2 JAMS
    VOLCANIC_ROCK = 8
    VOUCHER_9500 = 9
    VOUCHER_9750 = 10
    VOUCHER_10000 = 11
    VOUCHER_10250 = 12
    VOUCHER_10500 = 13
    MACARON = 14
    
products = [
    # round 1
    Product.RESIN, Product.KELP, Product.INK,
    
    # round 2
    Product.DJEMBES, Product.JAMS, Product.CROISSANTS, Product.BASKET1, Product.BASKET2,
    
    # round 3
    Product.VOLCANIC_ROCK, Product.VOUCHER_9500, Product.VOUCHER_9750, Product.VOUCHER_10000, Product.VOUCHER_10250, Product.VOUCHER_10500,
    
    # round 4
    Product.MACARON
]

product_strings = [
    "RAINFOREST_RESIN", "KELP", "SQUID_INK",
    "DJEMBES", "JAMS", "CROISSANTS", "PICNIC_BASKET1", "PICNIC_BASKET2",
    "VOLCANIC_ROCK", "VOLCANIC_ROCK_VOUCHER_9500", "VOLCANIC_ROCK_VOUCHER_9750", "VOLCANIC_ROCK_VOUCHER_10000", "VOLCANIC_ROCK_VOUCHER_10250", "VOLCANIC_ROCK_VOUCHER_10500",
    "MAGNIFICENT_MACARONS"
]

class Trader:
    def __init__(self):
        self.LIMIT = [
            50, # Product.RESIN
            50, # Product.KELP
            50, # Product.INK
            250, # Product.CROISSANTS
            350, # Product.JAMS
            60, # Product.DJEMBES
            60, # Product.BASKET1
            100, # Product.BASKET2
            400, # Product.VOLCANIC_ROCK
            200, # Product.VOUCHER_9500
            200, # Product.VOUCHER_9750
            200, # Product.VOUCHER_10000
            200, # Product.VOUCHER_10250
            200, # Product.VOUCHER_10500
            75, # Product.MACARON
        ]
        
        self.historical_avgs = [
            10000, # Product.RESIN
            2033, # Product.KELP
            1904, # Product.INK
            13436, # Product.CROISSANTS
            4298, # Product.JAMS
            6593, # Product.DJEMBES
            59052, # Product.BASKET1
            30408, # Product.BASKET2
            10332, # Product.VOLCANIC_ROCK
            832, # Product.VOUCHER_9500
            583, # Product.VOUCHER_9750
            343, # Product.VOUCHER_10000
            148, # Product.VOUCHER_10250
            41, # Product.VOUCHER_10500
            664, # Product.MACARON
        ]
        
        self.mp_window_size = 2
        self.mid_prices = [
            [], # Product.RESIN
            [], # Product.KELP
            [], # Product.INK
            [], # Product.CROISSANTS
            [], # Product.JAMS
            [], # Product.DJEMBES
            [], # Product.BASKET1
            [], # Product.BASKET2
            [], # Product.VOLCANIC_ROCK
            [], # Product.VOUCHER_9500
            [], # Product.VOUCHER_9750
            [], # Product.VOUCHER_10000
            [], # Product.VOUCHER_10250
            [], # Product.VOUCHER_10500
            [], # Product.MACARON
        ]
        
        # stores spread of basket1 and its components
        self.spread_history = []
        self.spread_history_size = 50
        
        # store spread of options
        self.option_spread_history = {
            'VOLCANIC_ROCK_VOUCHER_9500': [],
            'VOLCANIC_ROCK_VOUCHER_9750': [],
            'VOLCANIC_ROCK_VOUCHER_10000': [],
            'VOLCANIC_ROCK_VOUCHER_10250': [],
            'VOLCANIC_ROCK_VOUCHER_10500': []
        }
        self.option_spread_history_size = 50
        
        self.option_spread_means = {
            Product.VOUCHER_9500: -0.007616528715159741,
            Product.VOUCHER_9750: 0.00024144919130100485,
            Product.VOUCHER_10000: 0.0038512838011738647,
            Product.VOUCHER_10250: 0.002831169571447253,
            Product.VOUCHER_10500: 0.000569217926167293
        }
        
        self.buy_order_volume = 0
        self.sell_order_volume = 0
    
    def calculate_mid_price(self, buy_orders, sell_orders):
        if not buy_orders or not sell_orders:
            return 0.0
        
        # get best bid and ask prices
        best_bid = max(buy_orders.keys())
        best_ask = min(sell_orders.keys())
        
        # calculate mid price
        mid_price = (best_bid + best_ask) / 2
        
        return mid_price        
    
    def calculate_fair_price(self, buy_orders, sell_orders):
        avg_buy = statistics.mean(buy_orders.keys()) if buy_orders else 0.0
        avg_sell = statistics.mean(sell_orders.keys()) if sell_orders else 0.0
        
        # take midpoint of buy and sell averages
        fair_price = (avg_buy + avg_sell) / 2
        
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
        variance = statistics.variance(self.mid_prices[product]) if len(self.mid_prices[product]) >= 2 else 0 # variance
        
        rpf = mid_price - position*gamma*variance*time_left
        
        return rpf

    # takes the best orders from the order depth and places them in the orders list
    # if the order is within the limits and the price is better than the fair price
    def take_best_orders(self, product, orders, order_depth, fair_price, width, position):
            logger.print("-- Taking best orders --")
            
            if len(order_depth.sell_orders) != 0:
                best_ask = min(order_depth.sell_orders.keys())
                best_ask_amount = -order_depth.sell_orders[best_ask]
                
                if best_ask <= fair_price - width:
                    quantity = min(best_ask_amount, self.LIMIT[product] - position) # max amount to buy 
                    if quantity > 0:
                        orders.append(Order(product_strings[product], best_ask, quantity)) 
                        logger.print(f"BUY {quantity} @ {best_ask}")
                        self.buy_order_volume += quantity

            if len(order_depth.buy_orders) != 0:
                best_bid = max(order_depth.buy_orders.keys())
                best_bid_amount = order_depth.buy_orders[best_bid]
                if best_bid >= fair_price + width:
                    quantity = min(best_bid_amount, self.LIMIT[product] + position) # max we can sell 
                    if quantity > 0:
                        orders.append(Order(product_strings[product], best_bid, -quantity))
                        logger.print(f"SELL {quantity} @ {best_bid}")
                        self.sell_order_volume += quantity
    
    # a risk management function that neutralizes or reduces the current position 
    def clear_position_order(self, product, orders, order_depth, fair_value, position):
        logger.print("-- Clearing position --")
        new_position = position + self.buy_order_volume - self.sell_order_volume
        
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)

        # how much we can buy/sell without breaching position limits
        buy_quantity = self.LIMIT[product] - (position + self.buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - self.sell_order_volume)

        if new_position > 0 and fair_for_ask in order_depth.buy_orders.keys():
            clear_quantity = min(order_depth.buy_orders[fair_for_ask], new_position)
            sent_quantity = min(sell_quantity, clear_quantity)
            orders.append(Order(product_strings[product], fair_for_ask, -abs(sent_quantity)))
            logger.print(f"SELL {sent_quantity} @ {fair_for_ask}")
            self.sell_order_volume += abs(sent_quantity)

        if new_position < 0 and fair_for_bid in order_depth.sell_orders.keys():
            clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(new_position))
            sent_quantity = min(buy_quantity, clear_quantity)
            orders.append(Order(product_strings[product], fair_for_bid, abs(sent_quantity)))
            logger.print(f"BUY {sent_quantity} @ {fair_for_bid}")
            self.buy_order_volume += abs(sent_quantity)
    
    def market_make(self, product, orders, order_depth, fair_price, position):
        logger.print("-- Market making --")
        
        # calculate best bid and ask
        filtered_bids = [p for p in order_depth.buy_orders.keys() if p < fair_price - 1]
        filtered_asks = [p for p in order_depth.sell_orders.keys() if p > fair_price + 1]
        
        bid = max(filtered_bids) + 1 if len(filtered_bids) > 0 else max(order_depth.buy_orders.keys())
        ask = min(filtered_asks) - 1 if len(filtered_asks) > 0 else min(order_depth.sell_orders.keys())

        # make buy order
        buy_quantity = self.LIMIT[product] - (position + self.buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product_strings[product], bid, buy_quantity))
            # self.buy_order_volume += buy_quantity
            logger.print(f"BUY {buy_quantity} @ {bid}")

        # make sell order
        sell_quantity = self.LIMIT[product] + (position - self.sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product_strings[product], ask, -sell_quantity))
            # self.sell_order_volume += sell_quantity
            logger.print(f"SELL {sell_quantity} @ {ask}")
    
    # safely places orders without exceeding position limits
    def safe_order(self, product, price, quantity, position, orders):
        # buy
        if quantity > 0:
            buy_quantity = self.LIMIT[product] - (position + self.buy_order_volume)
            if buy_quantity <= 0:
                return # Can't buy more
            qty = min(quantity, buy_quantity)
            self.buy_order_volume += qty
        
        # sell
        else:
            sell_quantity = self.LIMIT[product] + (position - self.sell_order_volume)
            if sell_quantity <= 0:
                return # Can't sell more
            qty = -min(-quantity, sell_quantity)
            self.sell_order_volume += abs(qty)

        if qty != 0:
            orders.append(Order(product_strings[product], price, qty))
    
    def basket_arb(self, threshold, mid_prices, best_bids, best_asks, positions, result):
        logger.print("-- Basket 1 Arbitrage --")
        # compute price of synthetic basket
        synthetic_basket1_price = 6 * mid_prices[Product.CROISSANTS] + 3 * mid_prices[Product.JAMS] + mid_prices[Product.DJEMBES]
        
        # compute spread
        spread = mid_prices[Product.BASKET1] - synthetic_basket1_price
        
        # log spread in history
        self.spread_history.append(spread)
        if len(self.spread_history) < self.spread_history_size:
            return # not enough data to compute stddev
        elif len(self.spread_history) > self.spread_history_size:
            self.spread_history.pop(0)
        
        # compute stddev and z-score
        z_score = (spread - 48.762433333333334) / np.std(self.spread_history)
        
        logger.print(f"Basket1 Price: {mid_prices[Product.BASKET1]}, Synthetic Basket1 Price: {synthetic_basket1_price}, Spread: {spread}")
        
        target_position = self.LIMIT[Product.BASKET1] - 2
        
        # +spread -> basket1 is overpriced
        if z_score >= threshold and positions[Product.BASKET1] != -target_position:
            # sell basket1 at bid
            self.safe_order(Product.BASKET1, best_bids[Product.BASKET1], -1, positions[Product.BASKET1], result[product_strings[Product.BASKET1]])
            
            # buy components at ask
            self.safe_order(Product.CROISSANTS, best_asks[Product.CROISSANTS], 6, positions[Product.CROISSANTS], result[product_strings[Product.CROISSANTS]])
            self.safe_order(Product.JAMS, best_asks[Product.JAMS], 3, positions[Product.JAMS], result[product_strings[Product.JAMS]])
            self.safe_order(Product.DJEMBES, best_asks[Product.DJEMBES], 1, positions[Product.DJEMBES], result[product_strings[Product.DJEMBES]])
        
        # -spread -> basket1 is underpriced
        if z_score <= -threshold and positions[Product.BASKET1] != target_position:
            # buy basket1 at ask
            self.safe_order(Product.BASKET1, best_asks[Product.BASKET1], 1, positions[Product.BASKET1], result[product_strings[Product.BASKET1]])
            
            # sell components at bid
            self.safe_order(Product.CROISSANTS, best_bids[Product.CROISSANTS], -6, positions[Product.CROISSANTS], result[product_strings[Product.CROISSANTS]])
            self.safe_order(Product.JAMS, best_bids[Product.JAMS], -3, positions[Product.JAMS], result[product_strings[Product.JAMS]])
            self.safe_order(Product.DJEMBES, best_bids[Product.DJEMBES], -1, positions[Product.DJEMBES], result[product_strings[Product.DJEMBES]])
    
    # returns bid, ask
    def get_macaron_bid_ask(self, observation: ConversionObservation):
        storage_cost = 0.1
        bid = observation.bidPrice - (observation.exportTariff + observation.transportFees + storage_cost)
        ask = observation.askPrice + observation.importTariff + observation.transportFees + storage_cost
        return bid, ask
    
    def macaron_arb_clear_pos(self, position):
        return -position
    
    def macaron_arb_take(self, result, order_depth: OrderDepth, observation: ConversionObservation, edge: float, position: int):
        logger.print('-- MACARON ARB TAKE --')
        
        implied_bid, implied_ask = self.get_macaron_bid_ask(observation)
        logger.print(f"Implied Bid: {implied_bid}, Implied Ask: {implied_ask}")
        
        buy_quantity = self.LIMIT[Product.MACARON] - position
        sell_quantity = self.LIMIT[Product.MACARON] + position

        for price in sorted(list(order_depth.sell_orders.keys())):
            if price > implied_bid - edge:
                break
            
            if price < implied_bid - edge:
                # max amount to buy
                quantity = min(abs(order_depth.sell_orders[price]), buy_quantity)
                if quantity > 0:
                    result[product_strings[Product.MACARON]].append(Order(product_strings[Product.MACARON], price, quantity))
                    logger.print(f"BUY {quantity} @ {price}")
                    self.buy_order_volume += quantity

        for price in sorted(list(order_depth.buy_orders.keys()), reverse=True):
            if price < implied_ask + edge:
                break

            if price > implied_ask + edge:
                # max amount to sell
                quantity = min(abs(order_depth.buy_orders[price]), sell_quantity)
                if quantity > 0:
                    result[product_strings[Product.MACARON]].append(Order(product_strings[Product.MACARON], price, -quantity))
                    logger.print(f"SELL {quantity} @ {price}")
                    self.sell_order_volume += quantity

    def macaron_arb_make(self, result, order_depth: OrderDepth, observation: ConversionObservation, edge: float, position: int):
        logger.print('-- MACARON ARB MAKE --')
        
        implied_bid, implied_ask = self.get_macaron_bid_ask(observation)
        logger.print(f"Implied Bid: {implied_bid}, Implied Ask: {implied_ask}")
        
        bid = implied_bid - edge
        ask = implied_ask + edge
        
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
        
        buy_quantity = min(self.LIMIT[Product.MACARON] - (position + self.buy_order_volume), self.LIMIT[Product.MACARON])
        if buy_quantity > 0:
            if ask > best_bid:
                result[product_strings[Product.MACARON]].append(Order(product_strings[Product.MACARON], best_bid, buy_quantity))  # Buy order
            else:
                result[product_strings[Product.MACARON]].append(Order(product_strings[Product.MACARON], round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[Product.MACARON] + (position - self.sell_order_volume)
        if sell_quantity > 0:
            result[product_strings[Product.MACARON]].append(Order(product_strings[Product.MACARON], round(ask), -sell_quantity))  # Sell order
        
    def run(self, state: TradingState):
        conversions = None
        result = { p: [] for p in product_strings }
        
        self.buy_order_volume = 0
        self.sell_order_volume = 0
        
        # load trader data
        if len(state.traderData) > 0:
            trader_data = jsonpickle.decode(state.traderData)
            self.mid_prices = trader_data["mid_prices"]
            self.spread_history = trader_data["spread_history"]
            self.option_spread_history = trader_data["option_spread_history"]
        
        mid_prices = []
        # best_bids = []
        # best_asks = []
        positions = []
        
        # initialize mid_prices, bids, asks, and positions for convenience
        for p_i, p in enumerate(product_strings):
            if p in state.order_depths and state.order_depths[p].buy_orders and state.order_depths[p].sell_orders:
                mid_price = self.calculate_mid_price(state.order_depths[p].buy_orders, state.order_depths[p].sell_orders)
                # best_bid = max(state.order_depths[p].buy_orders.keys()) if state.order_depths[p].buy_orders else 0
                # best_ask = min(state.order_depths[p].sell_orders.keys()) if state.order_depths[p].sell_orders else float("inf")
            else:
                mid_price = self.mid_prices[p_i][-1] if self.mid_prices[p_i] else self.historical_avgs[p_i]
                # best_bid = 0
                # best_ask = float("inf")
            
            mid_prices.append(mid_price)
            # best_bids.append(best_bid)
            # best_asks.append(best_ask)
            positions.append(state.position.get(p, 0))
        
        # update mid_prices for next iteration
        for p_i in products:
            self.mid_prices[p_i].append(mid_prices[p_i])
            if len(self.mid_prices[p_i]) > self.mp_window_size:
                self.mid_prices[p_i].pop(0)
        
        ### TODO: ADD OPTION TRADING HERE ###
        
        option = Product.VOUCHER_9500
        option_str = product_strings[option]
        order_depth = state.order_depths[option_str]
        
        round_number = 5
        call_price = round(mid_prices[option])
        strike = 9500 + 250*(option - Product.VOUCHER_9500)
        spot = round(mid_prices[Product.VOLCANIC_ROCK])
        tte = ((8 - round_number) / 7) - (state.timestamp / 1000000 / 7)
        
        iv = BlackScholes.implied_volatility(call_price, spot, strike, tte)
        fair = round(BlackScholes.black_scholes_call(spot, strike, tte, iv))
        logger.print(f"Option: {option_str}, Strike: {strike}, Spot: {spot}, TTE: {tte}, IV: {iv}")
        logger.print(f"Fair Price: {fair}, Call Price: {call_price}")
        
        # trade around fair price
        spread = fair - mid_prices[option]
        self.option_spread_history[option_str].append(spread)
        if len(self.option_spread_history[option_str]) > self.option_spread_history_size:
            self.option_spread_history[option_str].pop(0)
        
        if option_str in state.order_depths and len(self.option_spread_history[option_str]) == self.option_spread_history_size:
            z_score = (spread - self.option_spread_means[option]) / np.std(self.option_spread_history[option_str]) if len(self.option_spread_history[option_str]) > 1 else 0
            logger.print(f"Spread: {spread}, Z-Score: {z_score}")
            
            threshold = 0 # number of standard deviations needed to trigger a trade
            # buy signal
            if z_score > threshold and order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders.keys())
                best_ask_qty = order_depth.sell_orders[best_ask]
                
                self.safe_order(option, best_ask, -best_ask_qty, positions[option], result[option_str])
            
            # sell signal
            elif z_score < -threshold and order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_bid_qty = order_depth.buy_orders[best_bid]
                
                self.safe_order(option, best_bid, -best_bid_qty, positions[option], result[option_str])

        # update trader data
        trader_data = jsonpickle.encode({
            "mid_prices": self.mid_prices,
            "spread_history": self.spread_history,
            "option_spread_history": self.option_spread_history
        })

        logger.flush(state, result, conversions, trader_data)
        
        return result, conversions, trader_data
