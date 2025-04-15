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
    
products = [
    Product.RESIN, Product.KELP, Product.INK, 
    Product.DJEMBES, Product.JAMS, Product.CROISSANTS, Product.BASKET1, Product.BASKET2,
    Product.VOLCANIC_ROCK, Product.VOUCHER_9500, Product.VOUCHER_9750, Product.VOUCHER_10000, Product.VOUCHER_10250, Product.VOUCHER_10500
]

product_strings = [
    "RAINFOREST_RESIN", "KELP", "SQUID_INK",
    "DJEMBES", "JAMS", "CROISSANTS", "PICNIC_BASKET1", "PICNIC_BASKET2",
    "VOLCANIC_ROCK", "VOLCANIC_ROCK_VOUCHER_9500", "VOLCANIC_ROCK_VOUCHER_9750", "VOLCANIC_ROCK_VOUCHER_10000", "VOLCANIC_ROCK_VOUCHER_10250", "VOLCANIC_ROCK_VOUCHER_10500"
]

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
        ]
        
        # stores spread of basket1 and its components
        self.spread_history_size = 50
        self.spread_history = []
        
        self.mean_vol = {
            Product.VOUCHER_9500: 0.03336990208786825,
            Product.VOUCHER_9750: 0.0344296202495087,
            Product.VOUCHER_10000: 0.03129081943938605,
            Product.VOUCHER_10250: 0.029347176353511733,
            Product.VOUCHER_10500: 0.03001790981783722
        }
        
        self.vol_history_size = 25
        self.vol_history = {
            product_strings[Product.VOUCHER_9500]: [],
            product_strings[Product.VOUCHER_9750]: [],
            product_strings[Product.VOUCHER_10000]: [],
            product_strings[Product.VOUCHER_10250]: [],
            product_strings[Product.VOUCHER_10500]: [],
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
        
        return math.math.sqrt(variance)

    # gamma is the risk aversion coefficient; < 0.1 for less volatile, > 0.5 for more volatile
    def calculate_rpf(self, product, mid_price, position, gamma, timestamp):
        time_left = 1000000 - (timestamp % 1000000) # T-t
        variance = statistics.variance(self.mid_prices[product]) if len(self.mid_prices[product]) >= 2 else 0 # variance
        
        rpf = mid_price - position*gamma*variance*time_left
        logger.print(f"rpf: {rpf}")
        
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
        logger.print("-- Basket 1 Arbitrage --")
    
    def voucher_orders(self, z_score_threshold, product: Product, result, voucher_order_depth, position, implied_vol):
        key = product_strings[product]
        self.vol_history[key].append(implied_vol)
        
        if len(self.vol_history[key]) < self.vol_history_size:
            return
        elif len(self.vol_history[key]) > self.vol_history_size:
            self.vol_history[key].pop(0)
        
        vol_z_score = (implied_vol - self.mean_vol[product]) / np.std(self.vol_history[key])
        
        if vol_z_score >= z_score_threshold and position != self.LIMIT[product]:            
            target_position = -self.LIMIT[product]
            if len(voucher_order_depth.buy_orders) > 0:
                best_bid = max(voucher_order_depth.buy_orders.keys())
                target_quantity = abs(target_position - position)
                quantity = min(target_quantity, abs(voucher_order_depth.buy_orders[best_bid]))
                quote_quantity = target_quantity - quantity
                if quote_quantity == 0:
                    result[key].append(Order(key, best_bid, -quantity)) # take
                else:
                    result[key].append(Order(key, best_bid, -quantity)) # take
                    result[key].append(Order(key, best_bid, -quote_quantity)) # make
        elif vol_z_score <= -z_score_threshold and position != -self.LIMIT[product]:
            target_position = self.LIMIT[product]
            if len(voucher_order_depth.sell_orders) > 0:
                best_ask = min(voucher_order_depth.sell_orders.keys())
                target_quantity = abs(target_position - position)
                quantity = min(target_quantity, abs(voucher_order_depth.sell_orders[best_ask]))
                quote_quantity = target_quantity - quantity
                if quote_quantity == 0:
                    result[key].append(Order(key, best_ask, quantity)) # take
                else:
                    result[key].append(Order(key, best_ask, quantity)) # take
                    result[key].append(Order(key, best_ask, quote_quantity)) # make

    def volcano_delta_hedge(self, voucher: Product, result, volcano_order_depth, volcano_position, voucher_position, delta):
        if result[product_strings[voucher]] == None or len(result[product_strings[voucher]]) == 0:
            future_voucher_position = voucher_position
        else:
            future_voucher_position = voucher_position + sum(order.quantity for order in result[product_strings[voucher]])
        
        target_volcano_position = -delta * future_voucher_position
        
        if target_volcano_position == volcano_position:
            return
        
        target_volcano_quantity = target_volcano_position - volcano_position
        key = product_strings[Product.VOLCANIC_ROCK]
        
        if target_volcano_quantity > 0: # buy underlying asset
            best_ask = min(volcano_order_depth.sell_orders.keys())
            quantity = min(abs(target_volcano_quantity), self.LIMIT[Product.VOLCANIC_ROCK] - volcano_position)
            if quantity > 0:
                result[key].append(Order(key, best_ask, round(quantity)))
        
        elif target_volcano_quantity < 0: # sell underlying asset
            best_bid = max(volcano_order_depth.buy_orders.keys())
            quantity = min(abs(target_volcano_quantity), self.LIMIT[Product.VOLCANIC_ROCK] + volcano_position)
            if quantity > 0:
                result[key].append(Order(key, best_bid, -round(quantity)))
    
    def run(self, state: TradingState):
        conversions = None
        result = { p: [] for p in product_strings }
        
        self.buy_order_volume = 0
        self.sell_order_volume = 0
        
        # load trader data
        if state.traderData:
            trader_data = jsonpickle.decode(state.traderData)
            self.mid_prices = trader_data["mid_prices"]
            self.spread_history = trader_data["spread_history"]
            self.vol_history = trader_data["vol_history"]
        
        mid_prices = []
        best_bids = []
        best_asks = []
        positions = []
        
        # initialize mid_prices, bids, asks, and positions for convenience
        for p in product_strings:
            if p in state.order_depths:
                mid_price = self.calculate_mid_price(state.order_depths[p].buy_orders, state.order_depths[p].sell_orders)
                best_bid = max(state.order_depths[p].buy_orders.keys()) if state.order_depths[p].buy_orders else 0
                best_ask = min(state.order_depths[p].sell_orders.keys()) if state.order_depths[p].sell_orders else float("inf")
            else:
                mid_price = self.mid_prices[p][-1] if self.mid_prices[p] else self.historical_avgs[p]
                best_bid = 0
                best_ask = float("inf")
            
            mid_prices.append(mid_price)
            best_bids.append(best_bid)
            best_asks.append(best_ask)
            positions.append(state.position.get(p, 0))
        
        # update mid_prices for next iteration
        for p in products:
            self.mid_prices[p].append(mid_prices[p])
            if len(self.mid_prices[p]) > self.mp_window_size:
                self.mid_prices[p].pop(0)
        
        ### VOLCANIC ROCK VOUCHERS ###
        
        # Results:
        # Product.VOUCHER_9500: -915
        # Product.VOUCHER_9750: -24675
        # Product.VOUCHER_10000: -281
        # Product.VOUCHER_10250: 1682
        # Product.VOUCHER_10500: 178
        
        p = Product.VOUCHER_9500
        strike = 9500 + 250*(p - 9)
        
        # calculate implied volatility
        tte = (5/7) - (state.timestamp / 1000000 / 7) # (7 - 2) days until expiry because this is round 3 = 1 + 2
        implied_vol = BlackScholes.implied_volatility(mid_prices[p], mid_prices[Product.VOLCANIC_ROCK], strike, tte)
        delta = BlackScholes.delta(mid_prices[Product.VOLCANIC_ROCK], strike, tte, implied_vol)
        
        self.voucher_orders(2.5, p, result, state.order_depths.get(product_strings[Product.VOLCANIC_ROCK]), positions[Product.VOLCANIC_ROCK], implied_vol)
        self.volcano_delta_hedge(p, result, state.order_depths.get(product_strings[Product.VOLCANIC_ROCK]), positions[Product.VOLCANIC_ROCK], positions[p], delta)
        
        # update trader data
        trader_data = jsonpickle.encode({
            "mid_prices": self.mid_prices,
            "spread_history": self.spread_history,
            "vol_history": self.vol_history
        })
        
        logger.flush(state, result, conversions, trader_data)
        
        return result, conversions, trader_data