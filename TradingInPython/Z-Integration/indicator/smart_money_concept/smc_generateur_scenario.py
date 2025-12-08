""" 
    Génération de Candle sticks
    
    - generate_smc_scenario
    - generate_sample_data
"""
import pandas as pd
import numpy as np

def generate_smc_scenario( lg_data=150, start_price=100.0, seed=None ):
    """
    Scénario SMC ultra-réaliste avec :
    - Mouvement naturel avec consolidations
    - Pullbacks et mini-retracements
    - Structure fractale (petits swings dans la tendance)
    - Volatilité variable
    - Bougies authentiques (vraies mèches)
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    prices = []
    volatility = []  # Volatilité adaptative
    p = start_price
    current_vol = 0.15

    # ========================================================
    # 1) RANGE INITIAL - Accumulation (15-20 bougies)
    # ========================================================
    range_duration = np.random.randint(15, 21)
    range_center = start_price
    
    for i in range(range_duration):
        # Mouvement oscillant autour du centre
        mean_reversion = (range_center - p) * 0.15
        p += mean_reversion + np.random.uniform(-0.25, 0.25)
        prices.append(p)
        volatility.append(0.18)

    range_high = max(prices)
    range_low = min(prices)

    # ========================================================
    # 2) FORMATION SWING LOW - Descente progressive
    # ========================================================
    swing_low = range_low - np.random.uniform(1.0, 1.5)
    steps_down = 5
    
    for i in range(steps_down):
        progress = (i + 1) / steps_down
        target = range_low - (range_low - swing_low) * progress
        noise = np.random.uniform(-0.12, 0.12)
        p = target + noise
        prices.append(p)
        volatility.append(0.22)
    
    # Le vrai swing low
    prices.append(swing_low - np.random.uniform(0, 0.1))
    volatility.append(0.25)
    
    # Mini rebond technique
    for i in range(2):
        p = swing_low + np.random.uniform(0.15, 0.35)
        prices.append(p)
        volatility.append(0.20)

    # ========================================================
    # 3) SWEEP - Liquidity Grab (fausse cassure)
    # ========================================================
    sweep_low = swing_low - np.random.uniform(0.5, 0.8)
    prices.append(sweep_low)
    volatility.append(0.30)
    
    # Rejet violent (bougie à longue mèche)
    prices.append(swing_low + np.random.uniform(0.2, 0.5))
    volatility.append(0.28)

    # ========================================================
    # 4) CHoCH - Cassure du range
    # ========================================================
    # Montée progressive vers CHoCH
    choch_break = range_high + np.random.uniform(0.7, 1.0)
    steps_to_choch = 4
    
    for i in range(steps_to_choch):
        progress = (i + 1) / steps_to_choch
        target = prices[-1] + (choch_break - prices[-1]) * progress
        p = target + np.random.uniform(-0.15, 0.15)
        prices.append(p)
        volatility.append(0.24)

    # ========================================================
    # 5) IMPULSION + FVG
    # ========================================================
    # Forte impulsion sur 3 bougies
    imp1 = choch_break + np.random.uniform(1.2, 1.6)
    imp2 = imp1 + np.random.uniform(1.0, 1.4)
    imp3 = imp2 + np.random.uniform(0.6, 1.0)
    
    prices.extend([imp1, imp2, imp3])
    volatility.extend([0.35, 0.40, 0.35])

    # ========================================================
    # 6) ORDER BLOCK - Pause/correction
    # ========================================================
    ob_level = imp2 - np.random.uniform(0.3, 0.6)
    prices.append(ob_level)
    volatility.append(0.20)

    # ========================================================
    # 7) RETRACEMENT RÉALISTE - Descente en escalier
    # ========================================================
    impulse_top = imp3
    impulse_start = choch_break
    retrace_pct = np.random.uniform(0.55, 0.70)
    target_retrace = impulse_top - (impulse_top - impulse_start) * retrace_pct
    
    # Descente avec mini-rebonds (structure fractale)
    current_p = prices[-1]
    retrace_steps = np.random.randint(6, 10)
    
    for i in range(retrace_steps):
        # Descente générale
        step_size = (current_p - target_retrace) / (retrace_steps - i)
        p = prices[-1] - step_size
        
        # Mini-rebond tous les 2-3 bars
        if i % 3 == 1:
            p += np.random.uniform(0.15, 0.35)
        
        p += np.random.uniform(-0.12, 0.12)
        prices.append(p)
        volatility.append(0.18)

    # ========================================================
    # 8) BOS - Break of Structure
    # ========================================================
    bos_level = impulse_top + np.random.uniform(0.5, 1.0)
    
    # Montée vers BOS
    for i in range(3):
        p = prices[-1] + np.random.uniform(0.4, 0.7)
        prices.append(p)
        volatility.append(0.28)
    
    prices.append(bos_level)
    volatility.append(0.32)

    # ========================================================
    # 9) CONTINUATION RÉALISTE - Tendance avec pullbacks
    # ========================================================
    p = prices[-1]
    remaining = lg_data - len(prices)
    
    trend_mode = 'up'
    bars_in_mode = 0
    mode_duration = np.random.randint(8, 15)
    
    for i in range(remaining):
        bars_in_mode += 1
        
        # Alternance tendance / consolidation / pullback
        if bars_in_mode >= mode_duration:
            modes = ['up', 'consolidation', 'pullback']
            weights = [0.5, 0.3, 0.2]
            trend_mode = np.random.choice(modes, p=weights)
            bars_in_mode = 0
            mode_duration = np.random.randint(6, 12)
        
        if trend_mode == 'up':
            p += np.random.uniform(0.20, 0.40)
            vol = 0.20
        elif trend_mode == 'consolidation':
            p += np.random.uniform(-0.10, 0.15)
            vol = 0.15
        else:  # pullback
            p += np.random.uniform(-0.25, 0.05)
            vol = 0.18
        
        prices.append(p)
        volatility.append(vol)

    # ========================================================
    # Construction OHLC ULTRA-RÉALISTE
    # ========================================================
    ohlc = {"Open": [], "High": [], "Low": [], "Close": []}

    for idx, close_price in enumerate(prices):
        vol = volatility[idx]
        
        # Détermination du type de bougie basée sur le momentum
        if idx > 0:
            momentum = close_price - prices[idx - 1]
            is_bullish = momentum > 0
        else:
            is_bullish = np.random.rand() < 0.5
        
        # Taille du corps BEAUCOUP plus variable et aléatoire
        body_ratio = np.random.choice([
            np.random.uniform(0.3, 0.6),   # Petits corps (30%)
            np.random.uniform(0.6, 1.2),   # Corps moyens (40%)
            np.random.uniform(1.2, 2.5)    # Gros corps (30%)
        ], p=[0.3, 0.4, 0.3])
        
        body_size = body_ratio * vol * np.random.uniform(1.5, 3.0)
        
        # Mèches très variables
        wick_ratio_up = np.random.choice([
            np.random.uniform(0.1, 0.3),   # Mèche courte
            np.random.uniform(0.3, 0.6),   # Mèche moyenne
            np.random.uniform(0.6, 1.2)    # Longue mèche
        ], p=[0.5, 0.3, 0.2])
        
        wick_ratio_down = np.random.choice([
            np.random.uniform(0.1, 0.3),
            np.random.uniform(0.3, 0.6),
            np.random.uniform(0.6, 1.2)
        ], p=[0.5, 0.3, 0.2])
        
        if is_bullish:
            o = close_price - body_size
            c = close_price
            wick_up = wick_ratio_up * vol
            wick_down = wick_ratio_down * vol
        else:
            o = close_price + body_size
            c = close_price
            wick_up = wick_ratio_up * vol
            wick_down = wick_ratio_down * vol
        
        h = max(o, c) + wick_up
        l = min(o, c) - wick_down
        
        # Sécurité
        h = max(h, o, c)
        l = min(l, o, c)
        
        ohlc["Open"].append(o)
        ohlc["High"].append(h)
        ohlc["Low"].append(l)
        ohlc["Close"].append(c)

    # ========================================================
    # DataFrame final
    # ========================================================
    df = pd.DataFrame(ohlc)
    df["Date"] = pd.date_range(start="2025-01-01", periods=len(df), freq="h")
    df = df.set_index("Date")
    
    # Métadonnées simplifiées
    metadata = {
        'range_high': range_high,
        'range_low': range_low,
        'swing_low': min(prices[:len(prices)//3]),
        'choch_level': choch_break,
        'impulse_high': imp3,
        'bos_level': bos_level
    }
    
    return df, metadata

def generate_smc_scenario_4(lg_data=150, start_price=100.0, seed=None):
    """
    Scénario SMC ultra-réaliste avec :
    - Mouvement naturel avec consolidations
    - Pullbacks et mini-retracements
    - Structure fractale (petits swings dans la tendance)
    - Volatilité variable
    - Bougies authentiques (vraies mèches)
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    prices = []
    volatility = []  # Volatilité adaptative
    p = start_price
    current_vol = 0.15

    # ---------------------------------------------
    # 1) RANGE INITIAL - Accumulation (15-20 bougies)
    # ------------------------------------------------------
    range_duration = np.random.randint(15, 21)
    range_center = start_price
    
    for i in range(range_duration):
        # Mouvement oscillant autour du centre
        mean_reversion = (range_center - p) * 0.15
        p += mean_reversion + np.random.uniform(-0.25, 0.25)
        prices.append(p)
        volatility.append(0.18)

    range_high = max(prices)
    range_low = min(prices)

    # ------------------------------------------------------
    # 2) FORMATION SWING LOW - Descente progressive
    # ------------------------------------------------------
    swing_low = range_low - np.random.uniform(1.0, 1.5)
    steps_down = 5
    
    for i in range(steps_down):
        progress = (i + 1) / steps_down
        target = range_low - (range_low - swing_low) * progress
        noise = np.random.uniform(-0.12, 0.12)
        p = target + noise
        prices.append(p)
        volatility.append(0.22)
    
    # Le vrai swing low
    prices.append(swing_low - np.random.uniform(0, 0.1))
    volatility.append(0.25)
    
    # Mini rebond technique
    for i in range(2):
        p = swing_low + np.random.uniform(0.15, 0.35)
        prices.append(p)
        volatility.append(0.20)

    # ------------------------------------------------------
    # 3) SWEEP - Liquidity Grab (fausse cassure)
    # ------------------------------------------------------
    sweep_low = swing_low - np.random.uniform(0.5, 0.8)
    prices.append(sweep_low)
    volatility.append(0.30)
    
    # Rejet violent (bougie à longue mèche)
    prices.append(swing_low + np.random.uniform(0.2, 0.5))
    volatility.append(0.28)

    # ------------------------------------------------------
    # 4) CHoCH - Cassure du range
    # ------------------------------------------------------
    # Montée progressive vers CHoCH
    choch_break = range_high + np.random.uniform(0.7, 1.0)
    steps_to_choch = 4
    
    for i in range(steps_to_choch):
        progress = (i + 1) / steps_to_choch
        target = prices[-1] + (choch_break - prices[-1]) * progress
        p = target + np.random.uniform(-0.15, 0.15)
        prices.append(p)
        volatility.append(0.24)

    # ------------------------------------------------------
    # 5) IMPULSION + FVG
    # ------------------------------------------------------
    # Forte impulsion sur 3 bougies
    imp1 = choch_break + np.random.uniform(1.2, 1.6)
    imp2 = imp1 + np.random.uniform(1.0, 1.4)
    imp3 = imp2 + np.random.uniform(0.6, 1.0)
    
    prices.extend([imp1, imp2, imp3])
    volatility.extend([0.35, 0.40, 0.35])

    # ------------------------------------------------------
    # 6) ORDER BLOCK - Pause/correction
    # ------------------------------------------------------
    ob_level = imp2 - np.random.uniform(0.3, 0.6)
    prices.append(ob_level)
    volatility.append(0.20)

    # ------------------------------------------------------
    # 7) RETRACEMENT RÉALISTE - Descente en escalier
    # ------------------------------------------------------
    impulse_top = imp3
    impulse_start = choch_break
    retrace_pct = np.random.uniform(0.55, 0.70)
    target_retrace = impulse_top - (impulse_top - impulse_start) * retrace_pct
    
    # Descente avec mini-rebonds (structure fractale)
    current_p = prices[-1]
    retrace_steps = np.random.randint(6, 10)
    
    for i in range(retrace_steps):
        # Descente générale
        step_size = (current_p - target_retrace) / (retrace_steps - i)
        p = prices[-1] - step_size
        
        # Mini-rebond tous les 2-3 bars
        if i % 3 == 1:
            p += np.random.uniform(0.15, 0.35)
        
        p += np.random.uniform(-0.12, 0.12)
        prices.append(p)
        volatility.append(0.18)

    # ------------------------------------------------------
    # 8) BOS - Break of Structure
    # ------------------------------------------------------
    bos_level = impulse_top + np.random.uniform(0.5, 1.0)
    
    # Montée vers BOS
    for i in range(3):
        p = prices[-1] + np.random.uniform(0.4, 0.7)
        prices.append(p)
        volatility.append(0.28)
    
    prices.append(bos_level)
    volatility.append(0.32)

    # ------------------------------------------------------
    # 9) CONTINUATION RÉALISTE - Tendance avec pullbacks
    # ------------------------------------------------------
    p = prices[-1]
    remaining = lg_data - len(prices)
    
    trend_mode = 'up'
    bars_in_mode = 0
    mode_duration = np.random.randint(8, 15)
    
    for i in range(remaining):
        bars_in_mode += 1
        
        # Alternance tendance / consolidation / pullback
        if bars_in_mode >= mode_duration:
            modes = ['up', 'consolidation', 'pullback']
            weights = [0.5, 0.3, 0.2]
            trend_mode = np.random.choice(modes, p=weights)
            bars_in_mode = 0
            mode_duration = np.random.randint(6, 12)
        
        if trend_mode == 'up':
            p += np.random.uniform(0.20, 0.40)
            vol = 0.20
        elif trend_mode == 'consolidation':
            p += np.random.uniform(-0.10, 0.15)
            vol = 0.15
        else:  # pullback
            p += np.random.uniform(-0.25, 0.05)
            vol = 0.18
        
        prices.append(p)
        volatility.append(vol)

    # ------------------------------------------------------
    # Construction OHLC ULTRA-RÉALISTE
    # ------------------------------------------------------
    ohlc = {"Open": [], "High": [], "Low": [], "Close": []}

    for idx, close_price in enumerate(prices):
        vol = volatility[idx]
        
        # Détermination du type de bougie basée sur le momentum
        if idx > 0:
            momentum = close_price - prices[idx - 1]
            is_bullish = momentum > 0
        else:
            is_bullish = np.random.rand() < 0.5
        
        # Taille du corps variable
        body_size = np.random.uniform(0.10, 0.25) * vol * 2
        
        # Mèches asymétriques et réalistes
        if is_bullish:
            o = close_price - body_size
            c = close_price
            # Mèche haute plus petite, mèche basse plus grande (rejet vendeurs)
            wick_up = np.random.uniform(0.05, 0.15) * vol
            wick_down = np.random.uniform(0.15, 0.35) * vol
        else:
            o = close_price + body_size
            c = close_price
            # Mèche haute plus grande (rejet acheteurs)
            wick_up = np.random.uniform(0.15, 0.35) * vol
            wick_down = np.random.uniform(0.05, 0.15) * vol
        
        h = max(o, c) + wick_up
        l = min(o, c) - wick_down
        
        # Sécurité
        h = max(h, o, c)
        l = min(l, o, c)
        
        ohlc["Open"].append(o)
        ohlc["High"].append(h)
        ohlc["Low"].append(l)
        ohlc["Close"].append(c)

    # ------------------------------------------------------
    # DataFrame final
    # ------------------------------------------------------
    df = pd.DataFrame(ohlc)
    df["Date"] = pd.date_range(start="2025-01-01", periods=len(df), freq="h")
    df = df.set_index("Date")
    
    # Métadonnées simplifiées
    metadata = {
        'range_high': range_high,
        'range_low': range_low,
        'swing_low': min(prices[:len(prices)//3]),
        'choch_level': choch_break,
        'impulse_high': imp3,
        'bos_level': bos_level
    }
    
    return df, metadata
   
def generate_smc_scenario_3( lg_data=150, start_price=100.0 ):
    """
    Scénario SMC propre :
    - Range initial
    - Swing Low + Sweep
    - CHoCH haussier
    - Impulsion haussière + FVG
    - Order Block (bougie rouge)
    - Retracement dans OB/FVG
    - BOS haussier
    - Reprise de tendance
    """

    prices = []
    p = start_price

    # ----------------------------------------------------
    # 1) RANGE INITIAL – 12 bougies
    # ----------------------------------------------------
    for i in range(12):
        p += np.random.uniform(-0.20, 0.20)
        prices.append(p)

    range_high = max(prices)
    range_low  = min(prices)

    # ----------------------------------------------------
    # 2) SWING LOW propre (HL)
    # ----------------------------------------------------
    swing_low = range_low - 0.9
    prices.extend([
        swing_low + np.random.uniform(-0.05, 0.10),
        swing_low + np.random.uniform(0.05, 0.15)
    ])

    # ----------------------------------------------------
    # 3) SWEEP du Swing Low
    # ----------------------------------------------------
    sweep = swing_low - 0.6
    prices.extend([
        sweep + np.random.uniform(0.05, 0.10),
        sweep + np.random.uniform(0.10, 0.20)
    ])

    # ----------------------------------------------------
    # 4) CHoCH – cassure du range high
    # ----------------------------------------------------
    choch_break = range_high + 0.7
    prices.append(choch_break + np.random.uniform(0.10, 0.20))

    # ----------------------------------------------------
    # 5) IMPULSION + FVG
    # ----------------------------------------------------
    imp1 = choch_break + 1.2
    imp2 = imp1 + 1.1  # gap = FVG
    prices.extend([imp1, imp2])

    # ----------------------------------------------------
    # 6) ORDER BLOCK (dernière bougie rouge)
    # ----------------------------------------------------
    ob = imp1 - 0.35
    prices.append(ob)

    # ----------------------------------------------------
    # 7) RETRACEMENT dans OB + FVG
    # ----------------------------------------------------
    retrace1 = ob - 0.2
    retrace2 = retrace1 - 0.2
    prices.extend([retrace1, retrace2])

    # ----------------------------------------------------
    # 8) BOS haussier
    # ----------------------------------------------------
    bos_break = imp2 + 1.0
    prices.append(bos_break)

    # ----------------------------------------------------
    # 9) REPRISE DE TENDANCE
    # ----------------------------------------------------
    p = retrace2
    for _ in range(lg_data - len(prices)):
        p += np.random.uniform(0.15, 0.28)
        prices.append(p)

    # ----------------------------------------------------
    # Construction OHLC réaliste
    # ----------------------------------------------------
    ohlc = {"Open":[], "High":[], "Low":[], "Close":[]}

    for base in prices:
        o = base + np.random.uniform(-0.12, 0.12)
        c = base + np.random.uniform(-0.12, 0.12)
        h = max(o, c) + np.random.uniform(0.10, 0.25)
        l = min(o, c) - np.random.uniform(0.10, 0.25)
        ohlc["Open"].append(o)
        ohlc["High"].append(h)
        ohlc["Low"].append(l)
        ohlc["Close"].append(c)

    df = pd.DataFrame(ohlc)
    df["Date"] = pd.date_range(start="2025-01-01", periods=len(df), freq="h")
    df = df.set_index("Date")

    return df


def generate_smc_scenario_2(lg_data=150, start_price=100.0):
    """
    Scénario SMC réaliste :
    - Range initial
    - Swing Low + Sweep (Liquidity Grab)
    - CHoCH haussier
    - Impulsion + vrai FVG
    - Order Block
    - Retracement profond dans OB/FVG
    - BOS haussier
    - Reprise de tendance propre
    """

    prices = []
    p = start_price

    # ----------------------------------------------------
    # 1) Range initial : création de liquidité
    # ----------------------------------------------------
    for i in range(15):
        p += np.random.uniform(-0.25, 0.25)
        prices.append(p)

    range_high = max(prices)
    range_low  = min(prices)

    # ----------------------------------------------------
    # 2) Swing Low propre (HL)
    # ----------------------------------------------------
    swing_low = range_low - 0.9
    for i in range(3):
        prices.append(swing_low + np.random.uniform(-0.10, 0.15))

    # ----------------------------------------------------
    # 3) Sweep / Liquidity Grab : cassure du swing_low
    # ----------------------------------------------------
    sweep_low = swing_low - 0.6
    for i in range(2):
        prices.append(sweep_low + np.random.uniform(-0.05, 0.10))

    # ----------------------------------------------------
    # 4) CHoCH haussier (cassure du range high)
    # ----------------------------------------------------
    choch_break = range_high + 0.7
    prices.append(choch_break + np.random.uniform(0.05, 0.20))

    # ----------------------------------------------------
    # 5) Impulsion + vrai FVG de deux bougies ICT
    # ----------------------------------------------------
    imp1 = choch_break + 1.2
    imp2 = imp1 + 1.1   # FVG car aucune bougie ne revient sur cette zone
    prices.extend([imp1, imp2])

    # ----------------------------------------------------
    # 6) Order Block : petite bougie rouge après impulsion
    # ----------------------------------------------------
    ob = imp1 - 0.4
    prices.append(ob)

    # ----------------------------------------------------
    # 7) Retracement profond dans OB + FVG
    # ----------------------------------------------------
    retrace1 = ob - 0.2
    retrace2 = retrace1 - 0.2
    prices.extend([retrace1, retrace2])

    # ----------------------------------------------------
    # 8) BOS haussier
    # ----------------------------------------------------
    bos = imp2 + 1.0
    prices.append(bos)

    # ----------------------------------------------------
    # 9) Reprise de tendance légère
    # ----------------------------------------------------
    p = retrace2
    for i in range(lg_data - len(prices)):
        p += np.random.uniform(0.15, 0.30)
        prices.append(p)

    # ----------------------------------------------------
    # Construction OHLC (bougies réalistes)
    # ----------------------------------------------------
    ohlc = {"Open":[],"High":[],"Low":[],"Close":[]}

    for base in prices:
        o = base + np.random.uniform(-0.12, 0.12)
        c = base + np.random.uniform(-0.12, 0.12)
        h = max(o, c) + np.random.uniform(0.10, 0.30)
        l = min(o, c) - np.random.uniform(0.10, 0.30)
        ohlc["Open"].append(o)
        ohlc["High"].append(h)
        ohlc["Low"].append(l)
        ohlc["Close"].append(c)

    df = pd.DataFrame(ohlc)
    df["Date"] = pd.date_range(start="2025-01-01", periods=len(df), freq="h")
    df = df.set_index("Date")

    return df


def generate_smc_scenario_1( lg_data=120, start_price=100.0 ):
    """
    Génère un petit marché artificiel contenant TOUS les éléments SMC :
    - swing_low
    - liquidity grab (sweep)
    - CHoCH ICT : choch_break
    - Impulsion
    - FVG
    - OB
    - BOS
    - retracement
    - continuation

    Parfait pour tester BOS/CHoCH/OB/FVG.
    """

    prices = []

    # ----------------------------------------------------
    # 1) Phase 1 : Range propre (construction de liquidité)
    # ----------------------------------------------------
    p = start_price
    for i in range( 20 ):
        p += np.random.uniform( -0.2, 0.2 )
        prices.append(p)

    # ----------------------------------------------------
    # 2) Phase 2 : Swing Low propre
    # ----------------------------------------------------
    swing_low = p - 1.0
    prices.append( swing_low )

    # petite remontée
    for i in range(4):
        p = swing_low + (i+1) * 0.2
        prices.append( p )

    # ----------------------------------------------------
    # 3) Phase 3 : Sweep / Liquidity Grab (ICT)
    # ----------------------------------------------------
    sweep_low = swing_low - 0.6
    prices.append(sweep_low)

    # ----------------------------------------------------
    # 4) Phase 4 : CHoCH UP (ICT)
    #     cassure du swing high précédent (range)
    # ----------------------------------------------------
    choch_break = start_price + 0.7      # break du haut du range
    prices.append(choch_break)

    # ----------------------------------------------------
    # 5) Phase 5 : Impulsion (création du FVG)
    # ----------------------------------------------------
    impulse1 = choch_break + 1.4
    impulse2 = impulse1 + 1.0            # 2 bougies impulsives = FVG
    prices.extend([impulse1, impulse2])

    # ----------------------------------------------------
    # 6) Phase 6 : petite continuation puis BOS
    # ----------------------------------------------------
    bos_break = impulse2 + 1.2
    prices.append( bos_break )

    # ----------------------------------------------------
    # 7) Phase 7 : Retracement profond (vers OB/FVG)
    # ----------------------------------------------------
    retrace1 = impulse1 - 0.6
    retrace2 = retrace1 - 0.4
    prices.extend([retrace1, retrace2])

    # ----------------------------------------------------
    # 8) Phase 8 : Retour tendance haussière
    # ----------------------------------------------------
    p = retrace2
    for i in range( lg_data - len( prices ) ):
        p += np.random.uniform( 0.2, 0.25 )
        prices.append(p)

    # ----------------------------------------------------
    # Construction OHLC (bougies réalistes)
    # ----------------------------------------------------
    ohlc = {
        "Open": [],
        "High": [],
        "Low": [],
        "Close": []
    }

    print( f"Longeur du scénario: {len(prices)}")
    
    for i in range(len(prices)):
        base = prices[i]
        o = base + np.random.uniform(-0.15, 0.15)
        c = base + np.random.uniform(-0.15, 0.15)
        h = max(o, c) + np.random.uniform(0, 0.25)
        l = min(o, c) - np.random.uniform(0, 0.25)
        ohlc["Open"].append(o)
        ohlc["High"].append(h)
        ohlc["Low"].append(l)
        ohlc["Close"].append(c)

    df = pd.DataFrame(ohlc)
    df["Date"] = pd.date_range(start="2025-01-01", periods=len(df), freq="h")
    df = df.set_index("Date")

    return df

# -------------------------
# Sample data generator
# -------------------------
def generate_sample_data( nb=240, start="2024-01-01", seed=42 ):
    np.random.seed(seed)
    dates = pd.date_range(start=start, periods=nb, freq='D')
    price = 100.0
    prices = []
    for i in range(nb):
        # accumulation
        if i < nb * 0.2:
            drift = -0.012
            vol = 0.08
        # descente progressive     
        elif i < nb * 0.4:
            drift = -0.06
            vol = 0.12
        # CHoCH cassure
        elif i < nb * 0.5:
            drift = 0.32
            vol = 0.4
        # range
        elif i < nb * 0.6:
            drift = 0.05
            vol = 0.25
        elif i < nb * 0.8:
            drift = -0.4
            vol = 0.8
        else:
            drift = 0.2
            vol = 0.4
        price += drift + np.random.normal( 0, vol )
        prices.append( price )
        
    opens = np.array(prices) + np.random.normal( 0, 0.3, nb )
    highs = opens + np.abs( np.random.normal( 0.5, 0.3, nb ) )
    lows = opens - np.abs( np.random.normal( 0.5, 0.3, nb ) )
    closes = lows + (highs - lows) * np.random.rand( nb )

    df = pd.DataFrame({"Open":opens, "High":highs, "Low":lows, "Close":closes}, index=dates)
    return df

# Exemple d'utilisation
if __name__ == "__main__":
    df, meta = generate_smc_scenario( lg_data=150, start_price=100.0, seed=42 )
    
    print("=== Scénario SMC généré ===")
    print(f"Nombre de bougies: {len(df)}")
    print(f"\nNiveaux clés:")
    for key, value in meta.items():
        print(f"  {key}: {value:.2f}")
    
    print(f"\nPremières bougies:\n{df.head()}")
    print(f"\nDernières bougies:\n{df.tail()}")