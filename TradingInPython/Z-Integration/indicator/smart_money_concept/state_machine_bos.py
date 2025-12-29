"""
    Machine à états afin de détecter les BOS (Break Of Structure)
    avec gestion du retour au RANGE
"""
from enum import Enum, auto

class MarketState(Enum):
    RANGE = auto()
    UPTREND = auto()
    DOWNTREND = auto()
    POTENTIAL_REVERSAL = auto()

class MarketStockStateMachine:
    def __init__(self, max_reversal_count=2, price_compression_threshold=0.02):
        self.state = MarketState.RANGE
        self.structure = 'LL'
        
        # Paramètres pour détecter un RANGE
        self.max_reversal_count = max_reversal_count  # Nb max de POTENTIAL_REVERSAL avant RANGE
        self.price_compression_threshold = price_compression_threshold  # 2% par défaut
        
        # Compteurs
        self.reversal_count = 0
        self.structure_count_in_state = 0
        
        # Derniers pivots structurels
        self.last_HH = None
        self.last_HL = None
        self.last_LH = None
        self.last_LL = None
        
        # Historique récent des structures (pour détecter confusion)
        self.recent_structures = []
        self.max_history = 6

    def _is_price_compressed(self):
        """Détecte si le prix est compressé dans une fourchette étroite"""
        prices = [p[1] for p in [self.last_HH, self.last_HL, 
                              self.last_LH, self.last_LL] if p is not None]
        
        if len(prices) < 3:
            return False
            
        price_range = max(prices) - min(prices)
        avg_price = sum(prices) / len(prices)
        
        return (price_range / avg_price) < self.price_compression_threshold

    def _is_structure_confused(self):
        """Détecte une confusion dans la structure (alternance incohérente)"""
        if len(self.recent_structures) < 4:
            return False
        
        # Compte les structures haussières vs baissières dans l'historique récent
        bullish = sum(1 for s in self.recent_structures if s in ("HH", "HL"))
        bearish = sum(1 for s in self.recent_structures if s in ("LL", "LH"))
        
        # Si alternance équilibrée = range
        return abs(bullish - bearish) <= 1

    def _check_range_conditions(self):
        """Vérifie si les conditions de RANGE sont remplies"""
        # Trop de reversals successifs
        if self.reversal_count >= self.max_reversal_count:
            return True
        
        # Prix compressé dans une zone étroite
        if self._is_price_compressed():
            return True
        
        # Structure confuse/contradictoire
        if self._is_structure_confused():
            return True
        
        return False

    def update( self, structure, idx, price ):
        """
        structure ∈ {"HH", "HL", "LH", "LL"}
        price = valeur du pivot (pivot High ou pivot Low)
        """
        prev_state = self.state
        
        if structure is not None:
            self.structure = structure
        
            # Mise à jour de l'historique
            self.recent_structures.append( self.structure )
            if len( self.recent_structures ) > self.max_history:
                self.recent_structures.pop(0)
            
            self.structure_count_in_state += 1
            
            # --- Mémorisation des pivots ---
            if self.structure == "HH":
                self.last_HH = (idx, price)
            elif self.structure == "HL":
                self.last_HL = (idx, price)
            elif self.structure == "LH":
                self.last_LH = (idx, price)
            elif self.structure == "LL":
                self.last_LL = (idx, price)
        
        # --- Logique de la machine à états ---
        
        if self.state == MarketState.UPTREND:
            # Cassure du HL → CHoCH / BOS down
            if self.structure in ("LH", "LL") and self.last_HL and price < self.last_HL[1]:
                self.state = MarketState.POTENTIAL_REVERSAL
                self.reversal_count += 1
            
            # Confirme la tendance
            elif self.structure == "HH":
                self.state = MarketState.UPTREND
                self.reversal_count = 0  # Reset du compteur
            
            # Structure contradictoire faible (HL après plusieurs HH)
            elif self.structure == "HL":
                self.reversal_count = 0

        elif self.state == MarketState.DOWNTREND:
            # Cassure du LH → CHoCH / BOS up
            if self.structure in ("HH", "HL") and self.last_LH and price > self.last_LH[1]:
                self.state = MarketState.POTENTIAL_REVERSAL
                self.reversal_count += 1
            
            elif self.structure == "LL":
                self.state = MarketState.DOWNTREND
                self.reversal_count = 0
            
            elif self.structure == "LH":
                self.reversal_count = 0

        elif self.state == MarketState.POTENTIAL_REVERSAL:
            # Confirmation de reversal vers UPTREND
            if self.structure in ("HH", "HL"):
                self.state = MarketState.UPTREND
                self.reversal_count = 0
                self.structure_count_in_state = 0
            
            # Confirmation de reversal vers DOWNTREND
            elif self.structure in ("LL", "LH"):
                self.state = MarketState.DOWNTREND
                self.reversal_count = 0
                self.structure_count_in_state = 0
            
            # Si on reste trop longtemps en POTENTIAL_REVERSAL sans confirmation
            elif self.structure_count_in_state > 3:
                self.state = MarketState.RANGE
                self.reversal_count = 0
                self.structure_count_in_state = 0

        elif self.state == MarketState.RANGE:
            # Sortie du RANGE vers UPTREND
            if self.structure in ("HH", "HL"):
                # Besoin de 2 structures cohérentes pour sortir du RANGE
                if len( self.recent_structures ) >= 2 and \
                   all(s in ("HH", "HL") for s in self.recent_structures[-2:]):
                    self.state = MarketState.UPTREND
                    self.reversal_count = 0
                    self.structure_count_in_state = 0
            
            # Sortie du RANGE vers DOWNTREND
            elif self.structure in ("LL", "LH"):
                if len(self.recent_structures) >= 2 and \
                   all(s in ("LL", "LH") for s in self.recent_structures[-2:]):
                    self.state = MarketState.DOWNTREND
                    self.reversal_count = 0
                    self.structure_count_in_state = 0
        
        # --- Vérification globale des conditions de RANGE ---
        if self.state != MarketState.RANGE and self._check_range_conditions():
            self.state = MarketState.RANGE
            self.reversal_count = 0
            self.structure_count_in_state = 0
        
        # Reset du compteur de structures si changement d'état
        if prev_state != self.state:
            self.structure_count_in_state = 0
        
        return prev_state, self.state
    
    # -------------------------------------------------------------------------
    
    def reset(self):
        """Réinitialise complètement la machine à états"""
        self.state = MarketState.RANGE
        self.reversal_count = 0
        self.structure_count_in_state = 0
        self.last_HH = None
        self.last_HL = None
        self.last_LH = None
        self.last_LL = None
        self.recent_structures = []