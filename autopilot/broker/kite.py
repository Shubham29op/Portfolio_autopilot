"""KiteBroker - NOT IMPLEMENTED in v1 (paper only).

Mapping for when you go live (kiteconnect SDK):
  submit(buy)        -> kite.place_order(variety="regular", exchange="NSE", product="CNC",
                                         order_type="LIMIT", price=limit_price, ...)
  GTT OCO creation   -> kite.place_gtt(trigger_type=kite.GTT_TYPE_OCO,
                                       trigger_values=[stop_trigger, target_trigger], ...)
  modify_stop        -> kite.modify_gtt(...)
  positions/holdings -> kite.holdings() / kite.positions()  (reconcile against ledger daily)
  fills              -> kite.orders() / kite.trades() or postback webhook

Before enabling:
  * static IP whitelisted in the Kite developer console (orders are rejected otherwise)
  * DDPI submitted, so GTT sells don't need TPIN
  * daily login: access tokens expire each day; the runner must refuse to trade without one
  * host in India (retail algo rules)
"""


class KiteBroker:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Live trading is disabled in v1. Use PaperBroker.")
