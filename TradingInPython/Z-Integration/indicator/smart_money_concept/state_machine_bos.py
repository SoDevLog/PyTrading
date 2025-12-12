"""
    Machine à états afin de détecter les BOS (Break Of Structure)
"""
from enum import Enum, auto

class MarketState(Enum):
    RANGE = auto()
    UPTREND = auto()
    DOWNTREND = auto()
    POTENTIAL_REVERSAL = auto()

class MarketStockStateMachine:
    def __init__(self):
        self.state = MarketState.RANGE

        # derniers pivots structurels
        self.last_HH = None
        self.last_HL = None
        self.last_LH = None
        self.last_LL = None

    def update(self, swing_type, price):
        """
        swing_type ∈ {"HH", "HL", "LH", "LL"}
        price = valeur du pivot (pivot High ou pivot Low)
        """

        prev_state = self.state

        # --- mémorisation des pivots ---
        if swing_type == "HH":
            self.last_HH = price
        elif swing_type == "HL":
            self.last_HL = price
        elif swing_type == "LH":
            self.last_LH = price
        elif swing_type == "LL":
            self.last_LL = price

        # ----------- UPTREND -----------
        if self.state == MarketState.UPTREND:

            # cassure du HL → CHoCH / BOS down
            if swing_type in ("LH", "LL") and self.last_HL and price < self.last_HL:
                self.state = MarketState.POTENTIAL_REVERSAL

            # confirme la tendance
            elif swing_type == "HH":
                self.state = MarketState.UPTREND

        # ----------- DOWNTREND -----------
        elif self.state == MarketState.DOWNTREND:

            # cassure du LH → CHoCH / BOS up
            if swing_type in ("HH", "HL") and self.last_LH and price > self.last_LH:
                self.state = MarketState.POTENTIAL_REVERSAL

            elif swing_type == "LL":
                self.state = MarketState.DOWNTREND

        # ----------- POTENTIAL_REVERSAL -----------
        elif self.state == MarketState.POTENTIAL_REVERSAL:

            if swing_type in ("HH", "HL"):
                self.state = MarketState.UPTREND
            elif swing_type in ("LL", "LH"):
                self.state = MarketState.DOWNTREND

        # ----------- RANGE (initial) -----------
        elif self.state == MarketState.RANGE:
            if swing_type == "HH" and self.last_HL:
                self.state = MarketState.UPTREND
            elif swing_type == "LL" and self.last_LH:
                self.state = MarketState.DOWNTREND

        return prev_state, self.state
