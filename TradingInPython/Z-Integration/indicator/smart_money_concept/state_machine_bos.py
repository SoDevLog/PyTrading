"""
    Machine à états afin de détecter les BOS (Break Of Structure)
"""
from enum import Enum, auto

class MarketState( Enum ):
    RANGE = auto()
    UPTREND = auto()
    DOWNTREND = auto()
    POTENTIAL_REVERSAL = auto()

class MarketStockStateMachine:
    def __init__(self):
        self.state = MarketState.RANGE

        # derniers pivots structurels
        self.last_HH_price = None
        self.last_HL_price = None
        self.last_LH_price = None
        self.last_LL_price = None

    def update( self, structure, price ):
        """
        swing_type ∈ {"HH", "HL", "LH", "LL"}
        price = valeur du pivot (pivot High ou pivot Low)
        """

        prev_state = self.state
        
        if structure is None:
            return prev_state, self.state

        # --- mémorisation des pivots ---
        if structure == "HH":
            self.last_HH_price = price
        elif structure == "HL":
            self.last_HL_price = price
        elif structure == "LH":
            self.last_LH_price = price
        elif structure == "LL":
            self.last_LL_price = price

        if self.state == MarketState.UPTREND:

            # cassure du HL → CHoCH / BOS down
            if structure in ("LH", "LL") and self.last_HL_price and price < self.last_HL_price:
                self.state = MarketState.POTENTIAL_REVERSAL

            # confirme la tendance
            elif structure == "HH":
                self.state = MarketState.UPTREND

        elif self.state == MarketState.DOWNTREND:

            # cassure du LH → CHoCH / BOS up
            if structure in ("HH", "HL") and self.last_LH_price and price > self.last_LH_price:
                self.state = MarketState.POTENTIAL_REVERSAL

            elif structure == "LL":
                self.state = MarketState.DOWNTREND

        elif self.state == MarketState.POTENTIAL_REVERSAL:

            if structure in ("HH", "HL"):
                self.state = MarketState.UPTREND
            elif structure in ("LL", "LH"):
                self.state = MarketState.DOWNTREND

        elif self.state == MarketState.RANGE:
            
            if structure in ("HH", "HL"):
                self.state = MarketState.UPTREND
            elif structure in ("LL", "LH"):
                self.state = MarketState.DOWNTREND

        return prev_state, self.state
