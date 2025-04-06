from datamodel import OrderDepth, TradingState, Order
from typing import List, Any
import json
import jsonpickle
import math
import statistics
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

class Trader:
    def __init__(self):
        self.LIMIT = {
            Product.RESIN: 50,
            Product.KELP: 50,
        }
        
    def calculate_fair_price(self, buy_orders, sell_orders):
        avg_buy = statistics.mean(buy_orders.keys()) if buy_orders else 0
        avg_sell = statistics.mean(sell_orders.keys()) if sell_orders else 0
        
        # take midpoint of buy and sell averages
        fair_price = (avg_buy + avg_sell) / 2
        logger.print(f"Avg Buy: {avg_buy}, Avg Sell: {avg_sell}, Fair Price: {fair_price}")
        
        return fair_price
    
    def calculate_mid_price(self, buy_orders, sell_orders):
        # calculate best bid and ask
        best_bid = max(buy_orders.keys())
        best_ask = min(sell_orders.keys())
        
        logger.print(f"Best Bid: {best_bid}, Best Ask: {best_ask}")
        
        # calculate mid price
        if buy_orders and sell_orders:
            fair_price = (best_bid + best_ask) / 2
        elif buy_orders:
            fair_price = best_bid  # use bid price if no asks available
        elif sell_orders:
            fair_price = best_ask  # use ask price if no bids available
        else:
            raise ValueError("No buy or sell orders available to calculate fair price.")

        # Calculate the fair price as the average of the best bid and ask prices
        fair_price = (best_bid + best_ask) / 2

        return fair_price
    
    def obi(self, product, threshold, orders, order_depth, position, buy_order_volume, sell_order_volume):
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
    
    # Essentially a risk management function; it neutralizes or reduces the current position 
    def clear_position_order(self, product, fair_value, orders, order_depth, position, buy_order_volume, sell_order_volume):
        new_position = position + buy_order_volume - sell_order_volume
        
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)

        # how much we can buy/sell without breaching position limits
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if new_position > 0:
            if fair_for_ask in order_depth.buy_orders.keys():
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], new_position)
                sent_quantity = min(sell_quantity, clear_quantity)
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if new_position < 0:
            if fair_for_bid in order_depth.sell_orders.keys():
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(new_position))
                sent_quantity = min(buy_quantity, clear_quantity)
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
    
        return buy_order_volume, sell_order_volume
    
    def market_make(self, product, state: TradingState, orders, fair_price, position, buy_order_volume, sell_order_volume):
        logger.print(f"=== Market Making {product} ===")
        
        # calculate best bid and ask
        # bid = max([p for p in state.order_depths[product].buy_orders.keys() if p < fair_price - 1]) + 1
        # ask = min([p for p in state.order_depths[product].sell_orders.keys() if p > fair_price + 1]) - 1
        bid = max(state.order_depths[product].buy_orders.keys())
        ask = min(state.order_depths[product].sell_orders.keys())

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
        traderData = jsonpickle.encode([])
        conversions = None
        result = {}
        
        buy_order_volume = 0
        sell_order_volume = 0
        
        ### RAINFOREST RESIN ###
        
        product = Product.RESIN
        position = state.position.get(product, 0)
        fair_price = self.calculate_fair_price(state.order_depths[product].buy_orders, state.order_depths[product].sell_orders)
        order_depth = state.order_depths[product]
        orders = []
        
        buy_order_volume, sell_order_volume = self.obi(product, 0.3, orders, order_depth, position, buy_order_volume, sell_order_volume)
        buy_order_volume, sell_order_volume = self.clear_position_order(product, fair_price, orders, order_depth, position, buy_order_volume, sell_order_volume)
        buy_order_volume, sell_order_volume = self.market_make(product, state, orders, fair_price, position, buy_order_volume, sell_order_volume)
        
        result[product] = orders
        
        ### KELP ###

        product = Product.KELP
        position = state.position.get(product, 0)
        fair_price = self.calculate_fair_price(state.order_depths[product].buy_orders, state.order_depths[product].sell_orders)
        order_depth = state.order_depths[product]
        orders = []
        
        buy_order_volume, sell_order_volume = self.obi(product, 0.3, orders, order_depth, position, buy_order_volume, sell_order_volume)
        buy_order_volume, sell_order_volume = self.clear_position_order(product, fair_price, orders, order_depth, position, buy_order_volume, sell_order_volume)
        _, _ = self.market_make(product, state, orders, fair_price, position, buy_order_volume, sell_order_volume)
        
        result[product] = orders
        
        logger.flush(state, result, conversions, traderData)
        
        return result, conversions, traderData
