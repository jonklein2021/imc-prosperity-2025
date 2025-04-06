def get_spread(self, state):
    spread = {}
    for product in state.order_depths:
        order_depth = state.order_depths[product]
        buy_orders = list(order_depth.buy_orders.items())
        sell_orders = list(order_depth.sell_orders.items())

        best_bid = max([price for price, qty in buy_orders])
        best_ask = min([price for price, qty in sell_orders])

        spread[product] = best_ask - best_bid
    return spread

