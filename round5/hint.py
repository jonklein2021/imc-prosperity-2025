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
        
        self.mp_window_size = 10 # also affects realized volatility calculation
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
        self.spread_history_size = 50
        self.spread_history = []
        
        self.mean_vol = {
            Product.VOUCHER_9500: 0.03336990208786825,
            Product.VOUCHER_9750: 0.0344296202495087,
            Product.VOUCHER_10000: 0.03129081943938605,
            Product.VOUCHER_10250: 0.029347176353511733,
            Product.VOUCHER_10500: 0.03001790981783722
        }
        
        self.parabolas = {
            Product.VOUCHER_9500: [0.00042769, -0.00258184, 0.03692129],
            Product.VOUCHER_9750: [0.00014304, -0.00036579, 0.03335499],
            Product.VOUCHER_10000: [3.97464607e-05, -2.49355363e-04, 3.14655950e-02],
            Product.VOUCHER_10250: [4.91265253e-05, -7.66743285e-05, 2.74558237e-02],
            Product.VOUCHER_10500: [2.78538060e-05, 3.97421286e-04, 2.38293126e-02]
        }
        
        self.base_iv = {
            Product.VOUCHER_9500: np.polyval(self.parabolas[Product.VOUCHER_9500], 0),
            Product.VOUCHER_9750: np.polyval(self.parabolas[Product.VOUCHER_9750], 0),
            Product.VOUCHER_10000: np.polyval(self.parabolas[Product.VOUCHER_10000], 0),
            Product.VOUCHER_10250: np.polyval(self.parabolas[Product.VOUCHER_10250], 0),
            Product.VOUCHER_10500: np.polyval(self.parabolas[Product.VOUCHER_10500], 0)
        }
        
        self.prev_spot = None
        
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
        n = len(prices)
        if n < 2:
            return 0.0  # Not enough data to calculate volatility

        squared_returns = []

        for t in range(1, n):
            # Check if P(t-1) is 0 and skip that price pair if true
            if prices[t-1] == 0 or prices[t] == 0:
                continue

            # Calculate log return for each pair of consecutive prices
            log_return = math.log(prices[t] / prices[t-1])
            squared_returns.append(log_return**2)

        # Calculate the average squared return and take the square root to get volatility
        if squared_returns:
            volatility = math.sqrt(sum(squared_returns) / (n-1))
            return volatility
        else:
            return 0.0  # If no valid prices to compute volatility

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
    
    def vol_spread_arb(self, threshold, option, result, option_order_depth, spot_order_depth, mid_prices, spot_pos, option_pos, timestamp):
        logger.print('-- Volatility spread arbitrage --')
        
        round_num = 5
        tte = ((8 - round_num) /7) - (timestamp / 1000000 / 7)
        strike = 9500 + (250 * (option - Product.VOUCHER_9500))
        
        # calculate IV
        iv = BlackScholes.implied_volatility(mid_prices[option], mid_prices[Product.VOLCANIC_ROCK], strike, tte)
        
        # calculate RV (should probably improve this calculation)
        rv = self.calculate_volatility(self.mid_prices[option])
        
        spread = iv - rv
        logger.print(f"IV: {iv}, RV: {rv}, Spread: {spread}")
        
        self.vol_spread_history[product_strings[option]].append(spread)
        if len(self.vol_spread_history[product_strings[option]]) > self.vol_spread_history_size:
            self.vol_spread_history[product_strings[option]].pop(0)
        elif len(self.vol_spread_history[product_strings[option]]) < self.vol_spread_history_size:
            return
        
        # calculate delta
        delta = BlackScholes.delta(mid_prices[Product.VOLCANIC_ROCK], strike, tte, iv)
        
        # short voucher
        if spread >= threshold:
            if len(option_order_depth.buy_orders) > 0:
                best_bid = max(option_order_depth.buy_orders)
                size = min(abs(option_order_depth.buy_orders[best_bid]), self.LIMIT[option] + option_pos)
                if size > 0:
                    result[product_strings[option]].append(Order(product_strings[option], best_bid, -size))

                # delta hedge: buy spot
                hedge_qty = round(size * delta)
                if len(spot_order_depth.sell_orders) > 0:
                    best_ask = min(spot_order_depth.sell_orders)
                    hedge_qty = min(hedge_qty, self.LIMIT[Product.VOLCANIC_ROCK] - spot_pos)
                    if hedge_qty > 0:
                        result[product_strings[Product.VOLCANIC_ROCK]].append(Order(product_strings[Product.VOLCANIC_ROCK], best_ask, hedge_qty))
            
        # long voucher
        elif spread <= -threshold:
            if len(option_order_depth.sell_orders) > 0:
                best_ask = min(option_order_depth.sell_orders)
                size = min(abs(option_order_depth.sell_orders[best_ask]), self.LIMIT[option] - option_pos)
                if size > 0:
                    result[product_strings[option]].append(Order(product_strings[option], best_ask, size))

                # delta hedge: sell spot
                hedge_qty = round(size * delta)
                if len(spot_order_depth.buy_orders) > 0:
                    best_bid = max(spot_order_depth.buy_orders)
                    hedge_qty = min(hedge_qty, self.LIMIT[Product.VOLCANIC_ROCK] + spot_pos)
                    if hedge_qty > 0:
                        result[product_strings[Product.VOLCANIC_ROCK]].append(Order(product_strings[Product.VOLCANIC_ROCK], best_bid, -hedge_qty))
        
    def delta_hedge(self, voucher: Product, result, volcano_order_depth, volcano_position, voucher_position, delta):
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
            self.prev_spot = trader_data["prev_spot"]
        
        mid_prices = []
        best_bids = []
        best_asks = []
        positions = []
        
        # initialize mid_prices, bids, asks, and positions for convenience
        for i, p in enumerate(product_strings):
            if p in state.order_depths:
                mid_price = self.calculate_mid_price(state.order_depths[p].buy_orders, state.order_depths[p].sell_orders)
                best_bid = max(state.order_depths[p].buy_orders.keys()) if state.order_depths[p].buy_orders else 0
                best_ask = min(state.order_depths[p].sell_orders.keys()) if state.order_depths[p].sell_orders else float("inf")
            else:
                mid_price = self.mid_prices[i][-1] if self.mid_prices[i] else self.historical_avgs[i]
                best_bid = 0
                best_ask = float("inf")
            
            if mid_price == 0:
                mid_price = self.historical_avgs[i]
            
            mid_prices.append(mid_price)
            best_bids.append(best_bid)
            best_asks.append(best_ask)
            positions.append(state.position.get(p, 0))
        
        # update mid_prices for next iteration
        for p_i in products:
            self.mid_prices[p_i].append(mid_prices[p_i])
            if len(self.mid_prices[p_i]) > self.mp_window_size:
                self.mid_prices[p_i].pop(0)
        
        ### VOLCANIC ROCK VOUCHERS ORDERS ###
        
        for option in [Product.VOUCHER_9500, Product.VOUCHER_9750, Product.VOUCHER_10000, Product.VOUCHER_10250, Product.VOUCHER_10500]:
            logger.print(f"--- {product_strings[option]} ---")
            round = 5
            tte = ((8 - round) / 7) - (state.timestamp / 1000000 / 7)
            spot = mid_prices[Product.VOLCANIC_ROCK]
            option_price = mid_prices[option]
            strike = 9500 + (250 * (option - Product.VOUCHER_9500))
            
            # calculate IV
            iv = BlackScholes.implied_volatility(option_price, spot, strike, tte)
            
            # calculate delta
            delta = BlackScholes.delta(spot, strike, tte, iv)
            logger.print(f"Delta: {delta}")
            
            # calculate gamma
            gamma = BlackScholes.gamma(spot, strike, tte, iv)
            logger.print(f"Gamma: {gamma}")
            
            # calculate m_t = log(strike / option_price)/ sqrt(tte)
            m_t = math.log(strike / option_price) / math.sqrt(tte)
            
            # calculate v_t
            v_t = iv
            
            # get expected v_t for this m_t
            expected_v_t = np.polyval(self.parabolas[option], m_t)
            base_v_t = self.base_iv[option]
            v_spread = base_v_t - v_t
            # v_spread = expected_v_t - v_t
            logger.print(f"Expected v_t: {expected_v_t}, Base v_t: {base_v_t}, Current v_t: {v_t}")
            logger.print(f"v_spread: {v_spread}")
            
            threshold = 0.01
            str_option = product_strings[option]
            
            # long the voucher
            if v_spread >= threshold and str_option in state.order_depths and state.order_depths[str_option].sell_orders:
                best_ask = min(state.order_depths[str_option].sell_orders.keys())
                best_ask_amount = state.order_depths[str_option].sell_orders[best_ask]
                self.safe_order(option, best_ask, -best_ask_amount, positions[option], result[str_option])
                
            # short the voucher
            elif v_spread <= -threshold and str_option in state.order_depths and state.order_depths[str_option].buy_orders:
                best_bid = max(state.order_depths[str_option].buy_orders.keys())
                best_bid_amount = state.order_depths[str_option].buy_orders[best_bid]
                self.safe_order(option, best_bid, -best_bid_amount, positions[option], result[str_option])
            
            ### DELTA HEDGE ###
            self.delta_hedge(option, result, state.order_depths[product_strings[Product.VOLCANIC_ROCK]], positions[Product.VOLCANIC_ROCK], positions[option], delta)
        
        # update trader data
        trader_data = jsonpickle.encode({
            "mid_prices": self.mid_prices,
            "spread_history": self.spread_history,
            "prev_spot": self.prev_spot
        })
        
        logger.flush(state, result, conversions, trader_data)
        
        return result, conversions, trader_data
