from typing import Any
import statistics
import json
import jsonpickle
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
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"

class Trader:
    def __init__(self):
        self.window_size = 3  # number of past prices to consider for moving average
        self.alpha = 2 / (self.window_size + 1) # EMA smoothing factor
        
        self.fair_prices = {
            Product.RAINFOREST_RESIN: [],
            Product.KELP: []
        }
        
        self.prev_ema_value = {
            Product.RAINFOREST_RESIN: 0,
            Product.KELP: 0
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
        best_bid = max(buy_orders.keys()) if buy_orders else 0
        best_ask = min(sell_orders.keys()) if sell_orders else float("inf")
        
        logger.print(f"Best Bid: {best_bid}, Best Ask: {best_ask}")
        
        # calculate mid price
        if buy_orders and sell_orders:
            fair_price = (best_bid + best_ask) / 2
        elif buy_orders:
            fair_price = best_bid  # use bid price if no asks available
        elif sell_orders:
            fair_price = best_ask  # use ask price if no bids available
        else:
            logger.print("No available prices.")
            return []

        # Calculate the fair price as the average of the best bid and ask prices
        fair_price = (best_bid + best_ask) / 2

        return fair_price
        
    def ema(self, state: TradingState, product):
        orders = []
        order_depth = state.order_depths[product]

        logger.print(f"=== {product} ===")

        # calculate fair price
        fair_price = self.calculate_fair_price(order_depth.buy_orders, order_depth.sell_orders)

        # add price to list
        self.fair_prices[product].append(fair_price)

        # calculate EMA
        if len(self.fair_prices[product]) == 1:
            ema = fair_price # initialize that jawn
        else:
            if len(self.fair_prices[product]) > self.window_size:
                self.fair_prices[product].pop(0)
            
            # calculate EMA using the previous EMA value
            ema = self.alpha * fair_price + (1 - self.alpha) * self.prev_ema_value[product]

        self.prev_ema_value[product] = ema

        logger.print(f"Current EMA: {ema}, Mid-Price: {fair_price}")

        # price below EMA → Buy signal
        if fair_price < ema:
            logger.print(f"[Buy signal] Mid-Price: {fair_price}")
            # find sell orders that are profitable to buy
            for ask, ask_quantity in order_depth.sell_orders.items():
                logger.print(f"Ask: {ask}, Ask Quantity: {ask_quantity}")
                # ensure good entry price
                if ask < fair_price:
                    logger.print(f"BUY {-ask_quantity} @ {ask}")
                    orders.append(Order(product, ask, -ask_quantity))

        # price above EMA → Sell signal
        elif fair_price > ema:
            logger.print(f"[Sell signal] Mid-Price: {fair_price}")
            # find buy orders that are profitable to sell
            for bid, bid_quantity in order_depth.buy_orders.items():
                logger.print(f"Bid: {bid}, Bid Quantity: {bid_quantity}")
                # ensure profitable exit
                if bid > fair_price:
                    logger.print(f"SELL {-bid_quantity} @ {bid}")
                    orders.append(Order(product, bid, -bid_quantity))

        return orders
    
    def run(self, state: TradingState):        
        conversions = None
        result = {}
        
        # get past midprices and emas from traderData
        if len(state.traderData) != 0:
            oldData = jsonpickle.decode(state.traderData)
            self.fair_prices = oldData["fair_prices"]
            self.prev_ema_value = oldData["prev_ema_value"]
            logger.print("fair_prices: " + str(self.fair_prices))
            logger.print("prev_ema_value: " + str(self.prev_ema_value))
        
        # apply EMA to each product
        for product in state.listings.keys():
            result[product] = self.ema(state, product)
        
        # update mid prices and ema values for next iteration
        traderData = jsonpickle.encode({
            "fair_prices": self.fair_prices,
            "prev_ema_value": self.prev_ema_value,
        })
        
        logger.flush(state, result, conversions, traderData)
        
        return result, conversions, traderData
