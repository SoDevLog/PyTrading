""" strategy_tool - SMA12E - Simple Mobile Average 12 and Exponential

    - update_graph
    - create_config_window
    - get_stock
    - get_configuration
    - command_save_configuration
    - save_configuration
    - command_get_default_configuration
     - command_get_item_configration
      - update_interface
       - complete_graph_window
    - toggle_visibility
    - draw_main_graph
    - add_title
    - set_data
    - set_fig
    
"""
import json
import os
import numpy
import tkinter as tk
import config.func as conf
import digitsignalprocessing.func as dsp
import helper as h
import figure.helper as fighelper
import styles.palette_colors as pc

from dataclasses import dataclass
from tkinter import ttk
from matplotlib.colors import to_rgba
from matplotlib.ticker import FuncFormatter, StrMethodFormatter
from figure.candlestick_ohlc import candlestick_ohlc, set_visibility_ohlc, remove_ohlc
from figure.linedeltaselector import LineDeltaSelector
from tkinterh.helper import Tooltip

# -----------------------------------------------------------------------------

# Exécution des ordres (ne change pas les signaux, seulement le prix et le coût du spread)
#   "close"     : clôture de la barre du signal (comportement d'origine)
#   "next_open" : ouverture de la barre suivante (on ne peut pas trader la clôture qui a servi au calcul)
EXECUTION = "close"
COST_PCT = 0.0  # frais par ordre, en % du prix (0.1 = 0,1 %)

PERIOD_MIN = 3   # bornes des périodes de moyennes mobiles (curseurs)
PERIOD_MAX = 400

# -----------------------------------------------------------------------------

@dataclass
class SignalResult:
    buy_bars:      numpy.ndarray # barres des signaux d'achat
    sell_bars:     numpy.ndarray # barres des signaux de vente
    buy_executed:  list          # un booléen par signal d'achat (False = ignoré)
    sell_executed: list          # un booléen par signal de vente
    spread:        float         # somme des (vente - achat), position ouverte soldée au dernier cours
    trades:        int           # nombre d'allers-retours, position ouverte comprise
    is_open:       bool          # True si la position est encore ouverte à la fin
    events:        list          # (libellé, prix, spread cumulé ou None) pour l'affichage console

def compute_signals( ma1, ma2, mae, close, open_, strategy_1_2=False, execution=EXECUTION, cost_pct=COST_PCT ):
    """ Signaux d'achat / vente (croisement MA1/MA2 confirmé par MAE) et spread.

        Achat : MA1 croise MA2 à la hausse et MAE est au-dessus des deux.
        Vente : MA1 croise MA2 à la baisse et MAE est sous les deux.
        Long seul, achats et ventes alternés : un achat alors qu'on est déjà en
        position, ou une vente alors qu'on n'y est pas, est un signal ignoré.

        strategy_1_2 : le signal est confirmé par la barre suivante (i+1). Il n'est donc
                       connu, et exécutable, qu'à la barre i+1 : c'est là qu'il est
                       placé et exécuté (et non à la barre i, ce qui serait de l'anticipation).
    """
    ma1 = numpy.asarray( ma1, dtype=float )
    ma2 = numpy.asarray( ma2, dtype=float )
    mae = numpy.asarray( mae, dtype=float )
    close = numpy.asarray( close, dtype=float )
    open_ = numpy.asarray( open_, dtype=float )

    n = len( close )
    cost = cost_pct / 100.0
    last = n - 1 if strategy_1_2 else n # strategy 1/2 lit la barre i+1

    def exec_price( k, side ): # side : +1 achat, -1 vente
        base = open_[k + 1] if ( execution == "next_open" and k + 1 < n ) else close[k]
        return base * ( 1.0 + side * cost )

    buy_bars, sell_bars, buy_executed, sell_executed, events = [], [], [], [], []
    in_position = False
    spread = 0.0
    trades = 0

    for i in range( 1, last ):
        up = ma1[i-1] <= ma2[i-1] and ma1[i] > ma2[i]
        down = ma1[i-1] >= ma2[i-1] and ma1[i] < ma2[i]
        buy = up and mae[i] > ma1[i] and mae[i] > ma2[i]
        sell = down and mae[i] < ma1[i] and mae[i] < ma2[i]
        k = i # barre où le signal est connu

        if strategy_1_2:
            j = i + 1
            buy = buy and ma1[j] > ma2[j] and mae[j] > ma1[j] and mae[j] > ma2[j]
            sell = sell and ma1[j] < ma2[j] and mae[j] < ma1[j] and mae[j] < ma2[j]
            k = j

        if buy:
            executed = not in_position
            price = exec_price( k, +1 ) if executed else close[k]
            if executed:
                in_position = True
                spread -= price
            buy_bars.append( k )
            buy_executed.append( executed )
            events.append( ( "Buy", price, spread if executed else None ) )

        if sell:
            executed = in_position
            price = exec_price( k, -1 ) if executed else close[k]
            if executed:
                in_position = False
                spread += price
                trades += 1
            sell_bars.append( k )
            sell_executed.append( executed )
            events.append( ( "Sell", price, spread if executed else None ) )

    # The spread is not closed still we are in "buy mode"
    # let sell at the last price for demonstration
    if in_position:
        price = close[-1] * ( 1.0 - cost )
        spread += price
        trades += 1
        events.append( ( "Sell", price, spread ) )

    return SignalResult(
        numpy.array( buy_bars, dtype=int ), numpy.array( sell_bars, dtype=int ),
        buy_executed, sell_executed, float( spread ), trades, in_position, events
    )

# -----------------------------------------------------------------------------

class strategy_sma12e:
    def __init__( self, window, message_entry, intraday, display ):
        self.window = window
        self.message_entry = message_entry
        self.intraday = intraday	
        self.display = display # tkinter checkbox
  
        self.name = None
        self.mobile_average_1 = None
        self.mobile_average_2 = None
        self.mobile_average_3 = None
        self.mobile_average_4 = None
        self.mobile_average_5 = None
        self.mobile_average_6 = None
        self.mobile_average_exp = None
        self.cum_price = None
        self.slope = None
        self.data = None
        self.change_chk_sates = True
        self.trades = None
        self.hold = None
        self.lines = []          # artistes retirés/recréés à chaque redessin
        self.candles = None
        self.line_trend = None
        self.selector = None
  
        self.get_configuration()

    # -------------------------------------------------------------------------

    def set_data( self, data, name ):
        self.data = data
        self.name = name

    def set_fig( self, fig ):
        self.fig = fig

    def set_canvas( self, canvas ):
        self.canvas = canvas

    def set_title( self, template ):
        self.template = template	
        return template.format( self.add_title() )

    # -------------------------------------------------------------------------

    def _on_period_change( self, var, value ):
        """ Commun aux trois réglages de période.
            value : str : valeur du curseur (nombre en texte)
                    -1/+1 : boutons - et +
                    autre : valeur saisie dans le champ (touche Entrée)
            La période est bornée à [PERIOD_MIN, PERIOD_MAX].
        """
        try:
            if isinstance( value, str ):
                value = int( float( value ) )
            elif value == -1:
                value = var.get() - 1
            elif value == +1:
                value = var.get() + 1
            else:
                value = var.get()
        except ( ValueError, tk.TclError ): # saisie invalide dans le champ
            return

        var.set( max( PERIOD_MIN, min( PERIOD_MAX, int( value ) ) ) )
        self.update_graph()

    def _snap_period( self, var, value ):
        """ Arrondit la valeur du curseur à l'entier sans redessiner :
            le graphique est mis à jour quand on relâche le curseur.
        """
        var.set( max( PERIOD_MIN, min( PERIOD_MAX, int( float( value ) ) ) ) )

    # ----------------------------------------------------------------------------

    def _add_period_row( self, frame, tk_row, text, var ):
        """ Une ligne de réglage de période : libellé, boutons - et +, curseur, champ.
            Retourne la prochaine ligne libre de la grille.
        """
        ttk.Label( frame, text=text ).grid( row=tk_row, column=0, columnspan=2, sticky="e" )
        tk_row += 1

        # Minus button
        ttk.Button( frame, text="-", width=2, command=lambda: self._on_period_change( var, -1 ) ).grid( row=tk_row, column=0, sticky="w" )

        # Slider pour la période de la moyenne mobile (de PERIOD_MIN à PERIOD_MAX jours)
        slider = ttk.Scale(
            frame,
            from_=PERIOD_MIN,
            to=PERIOD_MAX,
            orient=tk.HORIZONTAL,
            variable=var,
            length=130,
            command=lambda value: self._snap_period( var, value )
        )
        slider.grid( row=tk_row, column=1, sticky="w" )
        slider.bind( "<ButtonRelease-1>", lambda e: self.update_graph() )
        slider.bind( "<KeyRelease>", lambda e: self.update_graph() )

        # Plus button
        ttk.Button( frame, text="+", width=2, command=lambda: self._on_period_change( var, +1 ) ).grid( row=tk_row, column=2, sticky="w" )

        # Entry pour afficher la valeur actuelle du slider
        entry = ttk.Entry( frame, width=5, textvariable=var )
        entry.grid( row=tk_row, column=3, sticky="e", padx=5, pady=5 )
        entry.bind( "<Return>", lambda e: self._on_period_change( var, 0 ) )

        return tk_row + 1
  
    # ----------------------------------------------------------------------------  
    # Mettre à jour le graphique seulement quand on relâche le slider
    # ou lorsque la valeur change significativement
    #	
    def update_graph( self ):

        if not self.lines: # rien n'a encore été dessiné
            return

        # self.ax_main.clear() DO NOT clear() otherwise annotations won't work
        # remove alls lines because strategy by/sell has to be recalculate
        # Les bougies et la droite de tendance sont retirées aussi : sinon elles
        # s'empilent à chaque redessin (et les anciennes bougies restent visibles
        # quand on décoche "Candle")
        #
        for l in self.lines:
            l.remove()
        remove_ohlc( self.candles )
        self.line_trend.remove()
   
        self.draw_main_graph( self.ax_main, self.intraday, self.width )
        self.ax_main.set_title( self.template.format( self.add_title() ) )
        self.toggle_visibility()
        #self.fig.autofmt_xdate()
        self.canvas.draw_idle() # rafraichit lorsque le processeur est dispo
    
    # ----------------------------------------------------------------------------
    #
    def create_config_window( self ):

        self.window.title('MA1/2 MAE')
        content = ttk.Frame( self.window, padding=( 15, 10, 15, 10 ) ) # left top right bottom   
        content.grid( row=0, column=0, sticky="nsew" )
        
        tk_row = 0 
        padding_options = {'padx': 5, 'pady': 5}

        # ------------------------------------------------------------------------

        frame_ma12e = ttk.Frame( content )
        frame_ma12e.grid( row=0, column=0, columnspan=3, sticky="nsew" )
        
        # ------------------------------------------------------------------------

        self.mobile_average_1 = tk.IntVar( value=self.get_stock('MA1') )
        tk_row = self._add_period_row( frame_ma12e, tk_row, "Moyenne Mobile 1", self.mobile_average_1 )

        # ------------------------------------------------------------------------

        self.mobile_average_2 = tk.IntVar( value=self.get_stock('MA2') )
        tk_row = self._add_period_row( frame_ma12e, tk_row, "Moyenne Mobile 2", self.mobile_average_2 )

        # ------------------------------------------------------------------------

        self.mobile_average_exp = tk.IntVar( value=self.get_stock('MAE') )
        tk_row = self._add_period_row( frame_ma12e, tk_row, "Moyenne Mobile Exp", self.mobile_average_exp )
  
        # ------------------------------------------------------------------------

        separator = ttk.Separator( content, orient='horizontal' )
        separator.grid( row=tk_row, column=0, columnspan=2, sticky="ew", **padding_options)

        tk_row += 1
        
        # ------------------------------------------------------------------------
        
        ttk.Label( content, text="Mobile Average 3   :").grid( row=tk_row, column=0, sticky="e" )
        self.mobile_average_3 = tk.IntVar( value=self.get_stock('MA3') )
        _tki = ttk.Entry( content, width=10, textvariable=self.mobile_average_3 )
        _tki.grid( row=tk_row, column=1, sticky="w", **padding_options )
        _tki.bind( "<Return>", lambda e: self._on_period_change( self.mobile_average_3, 0 ) )
        
        tk_row += 1

        # ------------------------------------------------------------------------
        
        ttk.Label( content, text="Mobile Average 4   :").grid( row=tk_row, column=0, sticky="e" )
        self.mobile_average_4 = tk.IntVar( value=self.get_stock('MA4') )
        _tki = ttk.Entry( content, width=10, textvariable=self.mobile_average_4 )
        _tki.grid( row=tk_row, column=1, sticky="w", **padding_options )
        _tki.bind( "<Return>", lambda e: self._on_period_change( self.mobile_average_4, 0 ) )
        
        tk_row += 1

        # ------------------------------------------------------------------------
        
        ttk.Label( content, text="Mobile Average 5   :").grid( row=tk_row, column=0, sticky="e" )
        self.mobile_average_5 = tk.IntVar( value=self.get_stock('MA5') )
        _tki = ttk.Entry( content, width=10, textvariable=self.mobile_average_5 )
        _tki.grid( row=tk_row, column=1, sticky="w", **padding_options )
        _tki.bind( "<Return>", lambda e: self._on_period_change( self.mobile_average_5, 0 ) )
        
        tk_row += 1

        # ------------------------------------------------------------------------
        
        ttk.Label( content, text="Mobile Average 6   :").grid( row=tk_row, column=0, sticky="e" )
        self.mobile_average_6 = tk.IntVar( value=self.get_stock('MA6') )
        _tki = ttk.Entry( content, width=10, textvariable=self.mobile_average_6 )
        _tki.grid( row=tk_row, column=1, sticky="w", **padding_options )
        _tki.bind( "<Return>", lambda e: self._on_period_change( self.mobile_average_6, 0 ) )
        
        tk_row += 1
      
        # ------------------------------------------------------------------------

        separator = ttk.Separator( content, orient='horizontal' )
        separator.grid( row=tk_row, column=0, columnspan=2, sticky="ew", **padding_options)

        tk_row += 1
  
        # ------------------------------------------------------------------------

        _tki = ttk.Button(content, text="Default", command=self.command_get_default_configuration)
        _tki.grid( row=tk_row, column=0,  columnspan=2, **padding_options  )
        
        tk_row += 1
        
        _tki = ttk.Button(content, text="Read", command=self.command_get_item_configration)
        _tki.grid( row=tk_row, column=0,  columnspan=2, **padding_options  )

        tk_row += 1

        _tki = ttk.Button(content, text="Save", style="Custom.TButton", command=self.command_save_configuration)
        _tki.grid( row=tk_row, column=0,  columnspan=2, **padding_options  )

    # ----------------------------------------------------------------------------
    # Retreive an item from stock in configration file
    #
    def get_stock( self, item ):
        # Look in configuration if there is value for this compagny
        for stock in self.configuration.get( 'stocks' ):
            if stock['name'] == self.name:
                return stock.get(item, 0)

        # Take DEFAULT configuration
        for stock in self.configuration.get( 'stocks' ):
            if stock['name'] == 'DEFAULT':
                return stock.get(item, 0)
        
        return 0 # Erreur  

    # ----------------------------------------------------------------------------
 
    def get_configuration( self ):
        self.configuration, self.path_for_configuration_file = conf.read_configuration( 'strategy_sma12e.json' )
  
    # ----------------------------------------------------------------------------

    def command_save_configuration( self ):
        self.save_configuration()
        self.get_configuration()
        self.update_interface()
        self.message_entry.config( text=f"Configuration saved", foreground="orange" )

    # ----------------------------------------------------------------------------
  
    def save_configuration( self ):

        # Take DEFAULT values
        _default_ma1 = _default_ma2 = _default_ma3 = _default_ma4 = _default_ma5 = _default_ma6 = _default_mae = None # pas d'entrée DEFAULT dans le fichier
        for stock in self.configuration.get( 'stocks' ):
            if stock['name'] == 'DEFAULT':
                _default_ma1 = stock.get('MA1')
                _default_ma2 = stock.get('MA2')
                _default_ma3 = stock.get('MA3')
                _default_ma4 = stock.get('MA4')
                _default_ma5 = stock.get('MA5')
                _default_ma6 = stock.get('MA6')
                _default_mae = stock.get('MAE')
                break

        # Values are the same has DEFAULT
        # user retake DEFAULT values
        # we can suppress in configuration file
        #
        if _default_ma1 == self.mobile_average_1.get() \
            and _default_ma2 == self.mobile_average_2.get() \
            and _default_ma3 == self.mobile_average_3.get() \
            and _default_ma4 == self.mobile_average_4.get() \
            and _default_ma5 == self.mobile_average_5.get() \
            and _default_ma6 == self.mobile_average_6.get() \
            and _default_mae == self.mobile_average_exp.get():
            # Suppress element that it's like DEFAULT
            self.configuration['stocks'] = [stock for stock in self.configuration['stocks'] if stock['name'] != self.name]
            self._write_configuration()
            return
                    
        # Update otherwise Create
        for stock in self.configuration['stocks']:
            if stock['name'] == self.name:
                # Update
                stock['MA1'] = self.mobile_average_1.get()
                stock['MA2'] = self.mobile_average_2.get()
                stock['MA3'] = self.mobile_average_3.get()
                stock['MA4'] = self.mobile_average_4.get()
                stock['MA5'] = self.mobile_average_5.get()
                stock['MA6'] = self.mobile_average_6.get()
                stock['MAE'] = self.mobile_average_exp.get()
                self._write_configuration()
                return    
        
        # Create new item from Tkinter interface
        _config = {
            "name": self.name,
            "MA1": self.mobile_average_1.get(),
            "MA2": self.mobile_average_2.get(),
            "MA3": self.mobile_average_3.get(),
            "MA4": self.mobile_average_4.get(),
            "MA5": self.mobile_average_5.get(),
            "MA6": self.mobile_average_6.get(),
            "MAE": self.mobile_average_exp.get()
        }
        
        # Finaly Append (Created item)
        self.configuration.get('stocks').append( _config )
        self._write_configuration()

    # ----------------------------------------------------------------------------

    def _write_configuration( self ):
        # Ecrit dans un fichier temporaire puis remplace : un plantage en cours
        # ne pas laisser le fichier de configuration tronqué
        _tmp = self.path_for_configuration_file + ".tmp"
        with open( _tmp, "w", encoding="utf-8" ) as file:
            json.dump( self.configuration, file, indent=4 )
        os.replace( _tmp, self.path_for_configuration_file )

    # ----------------------------------------------------------------------------

    def command_get_default_configuration( self ):
        
        for stock in self.configuration.get( 'stocks' ):
            if stock['name'] == 'DEFAULT':
                self.mobile_average_1.set( stock.get('MA1') )
                self.mobile_average_2.set( stock.get('MA2') )
                self.mobile_average_3.set( stock.get('MA3') )
                self.mobile_average_4.set( stock.get('MA4') )
                self.mobile_average_5.set( stock.get('MA5') )
                self.mobile_average_6.set( stock.get('MA6') )
                self.mobile_average_exp.set( stock.get('MAE') )
                break

        self.message_entry.config( text=f"Default config", foreground="green" )
        self.update_graph()
            
    # ----------------------------------------------------------------------------

    def command_get_item_configration( self ):
        
        for stock in self.configuration.get( 'stocks' ):
            if stock['name'] == self.name:
                self.mobile_average_1.set( stock.get('MA1') )
                self.mobile_average_2.set( stock.get('MA2') )
                self.mobile_average_3.set( stock.get('MA3') )
                self.mobile_average_4.set( stock.get('MA4') )
                self.mobile_average_5.set( stock.get('MA5') )
                self.mobile_average_6.set( stock.get('MA6') )
                self.mobile_average_exp.set( stock.get('MAE') )
                break
        else: # aucune configuration pour cette valeur (ou liste vide)
            self.command_get_default_configuration()

    # ----------------------------------------------------------------------------

    def update_interface( self ):
        self.mobile_average_1.set( self.get_stock( 'MA1' ) )
        self.mobile_average_2.set( self.get_stock( 'MA2' ) )
        self.mobile_average_3.set( self.get_stock( 'MA3' ) )
        self.mobile_average_4.set( self.get_stock( 'MA4' ) )
        self.mobile_average_5.set( self.get_stock( 'MA5' ) )
        self.mobile_average_6.set( self.get_stock( 'MA6' ) )
        self.mobile_average_exp.set( self.get_stock( 'MAE' ) )   
        self.update_interface_chk()
    
    def update_interface_chk( self ):
        self.chk31.config( text=f"ma{self.mobile_average_1.get()}" )
        self.chk32.config( text=f"ma{self.mobile_average_2.get()}" )
        self.chk33.config( text=f"mae{self.mobile_average_exp.get()}" )
        self.chk34.config( text=f"ma{self.mobile_average_3.get()}" )  
        self.chk35.config( text=f"ma{self.mobile_average_4.get()}" )
        self.chk36.config( text=f"ma{self.mobile_average_5.get()}" )
        self.chk37.config( text=f"ma{self.mobile_average_6.get()}" )
  
    # ----------------------------------------------------------------------------
 
    def complete_graph_window( self, check_frame, command_update_graphs, command_update_lines ):
        
        # Create checkbox fro drawing lines
        self.var20 = tk.IntVar()
        self.var21 = tk.IntVar()
        self.var22 = tk.IntVar()
        self.var31 = tk.IntVar()
        self.var32 = tk.IntVar()
        self.var33 = tk.IntVar()
        self.var34 = tk.IntVar()
        self.var35 = tk.IntVar()
        self.var36 = tk.IntVar()
        self.var37 = tk.IntVar()
        self.var40 = tk.IntVar()
        self.var41 = tk.IntVar()
        self.var_all = tk.IntVar()

        # Strategy
        self.var70 = tk.IntVar()

        # lines = [line_price, line_price2, line_ma1, line_ma2, line_mae, line_ma4, line_ma4, fill1, fill2, line_stem1, line_stem2]
        
        # Init
        self.var20.set(0)
        self.var21.set(0)
        self.var22.set(1) # candles
        self.var31.set(1)
        self.var32.set(1)
        self.var33.set(1)
        self.var34.set(0)
        self.var35.set(0)
        self.var36.set(0)
        self.var37.set(1)
        self.var40.set(1)
        self.var41.set(1)
        self.var70.set(0)
        self.var_all.set(0)
        
        # Strategy
        chk70 = ttk.Checkbutton( check_frame, text="Strat1/2", variable=self.var70, command=command_update_graphs )
        Tooltip( chk70, "Stratégie de filtrage des signaux avec condition sur le cours + 1" )
    
        chk20 = ttk.Checkbutton( check_frame, text="Price", variable=self.var20, command=command_update_lines)
        chk21 = ttk.Checkbutton( check_frame, text="AdjClose", variable=self.var21, command=command_update_lines)
        chk22 = ttk.Checkbutton( check_frame, text="Candle", variable=self.var22, command=command_update_lines)

        self.chk31 = ttk.Checkbutton( check_frame, text=f"ma{self.mobile_average_1.get()}", variable=self.var31, command=command_update_lines)
        self.chk32 = ttk.Checkbutton( check_frame, text=f"ma{self.mobile_average_2.get()}", variable=self.var32, command=command_update_lines)
        self.chk33 = ttk.Checkbutton( check_frame, text=f"mae{self.mobile_average_exp.get()}", variable=self.var33, command=command_update_lines)
        self.chk34 = ttk.Checkbutton( check_frame, text=f"ma{self.mobile_average_3.get()}", variable=self.var34, command=command_update_lines)
        self.chk35 = ttk.Checkbutton( check_frame, text=f"ma{self.mobile_average_4.get()}", variable=self.var35, command=command_update_lines)
        self.chk36 = ttk.Checkbutton( check_frame, text=f"ma{self.mobile_average_5.get()}", variable=self.var36, command=command_update_lines)
        self.chk37 = ttk.Checkbutton( check_frame, text=f"ma{self.mobile_average_6.get()}", variable=self.var37, command=command_update_lines)
        chk40 = ttk.Checkbutton( check_frame, text="Fill", variable=self.var40, command=command_update_lines)
        chk41 = ttk.Checkbutton( check_frame, text="Marks", variable=self.var41, command=command_update_lines)
        chk_all = ttk.Checkbutton( check_frame, text="All", variable=self.var_all, command=command_update_lines)

        chk70.pack( side=tk.LEFT, padx=5, pady=5 )

        chk20.pack( side=tk.LEFT, padx=5, pady=5 )
        chk21.pack( side=tk.LEFT, padx=5, pady=5 )
        chk22.pack( side=tk.LEFT, padx=5, pady=5 )
        
        self.chk31.pack( side=tk.LEFT, padx=5, pady=5 )
        self.chk32.pack( side=tk.LEFT, padx=5, pady=5 )
        self.chk33.pack( side=tk.LEFT, padx=5, pady=5 )

        vertical_separator_frame = tk.Frame( check_frame, width=2, bg="gray" )  # Frame servant de séparateur
        vertical_separator_frame.pack( side=tk.LEFT, fill='y', padx=5 ) 

        self.chk34.pack( side=tk.LEFT, padx=5, pady=5 )
        self.chk35.pack( side=tk.LEFT, padx=5, pady=5 )
        self.chk36.pack( side=tk.LEFT, padx=5, pady=5 )
        self.chk37.pack( side=tk.LEFT, padx=5, pady=5 )
        chk40.pack( side=tk.LEFT, padx=5, pady=5 )
        chk41.pack( side=tk.LEFT, padx=5, pady=5 )
        chk_all.pack( side=tk.LEFT, padx=5, pady=5 )
    
    # ----------------------------------------------------------------------------

    def toggle_visibility( self ):
        lines = self.lines
        candles = self.candles		

        # lines = [line_price, line_price2, line_ma1, line_ma2, line_mae, line_ma3, line_ma4, line_ma5, line_ma6, fill1, fill2, line_stem1, line_stem2]
        
        idx = 0 # indice dans le tableau lines
  
        if self.var20.get():
            lines[idx].set_visible(True)
        else:
            lines[idx].set_visible(False)
        idx += 1

        if self.var21.get():
            lines[idx].set_visible(True)
        else:
            lines[idx].set_visible(False)
        idx += 1

        if self.var22.get():
            set_visibility_ohlc( candles, True )
        else:
            set_visibility_ohlc( candles, False )

        if self.var31.get():
            lines[idx].set_visible(True)
        else:
            lines[idx].set_visible(False)
        idx += 1

        if self.var32.get():
            lines[idx].set_visible(True)
        else:
            lines[idx].set_visible(False)
        idx += 1

        if self.var33.get():
            lines[idx].set_visible(True)
        else:
            lines[idx].set_visible(False)         
        idx += 1

        if self.var34.get():
            lines[idx].set_visible(True)
        else:
            lines[idx].set_visible(False)  
        idx += 1
   
        if self.var35.get():
            lines[idx].set_visible(True)
        else:
            lines[idx].set_visible(False)  
        idx += 1

        if self.var36.get():
            lines[idx].set_visible(True)
        else:
            lines[idx].set_visible(False)  
        idx += 1

        if self.var37.get():
            lines[idx].set_visible(True)
        else:
            lines[idx].set_visible(False)  
        idx += 1
                   
        if self.var40.get():
            lines[idx].set_visible(True)
            idx += 1
            lines[idx].set_visible(True)
        else:
            lines[idx].set_visible(False)
            idx += 1                      
            lines[idx].set_visible(False)
        idx += 1

        if self.var41.get():
            lines[idx].set_visible(True)
            idx += 1
            lines[idx].set_visible(True)
        else:
            lines[idx].set_visible(False)                          
            idx += 1
            lines[idx].set_visible(False)   
        idx += 1

        if self.var_all.get() and self.change_chk_sates:
            for l in lines:
                l.set_visible(True)
            self.var20.set(1)
            self.var21.set(1)
            #self.var22.set(1) candles
            self.var31.set(1)
            self.var32.set(1)
            self.var33.set(1)
            self.var34.set(1)
            self.var35.set(1)
            self.var36.set(1)
            self.var37.set(1)
            self.var40.set(1)
            self.var41.set(1)
            self.change_chk_sates = False
   
        if self.var_all.get() == False and self.change_chk_sates == False:
            for l in lines:
                l.set_visible(False)
            self.var20.set(0)
            self.var21.set(0)
            #self.var22.set(0) candles
            self.var31.set(0)
            self.var32.set(0)
            self.var33.set(0)
            self.var34.set(0)
            self.var35.set(0)
            self.var36.set(0)
            self.var37.set(0)
            self.var40.set(0)
            self.var41.set(0)
            self.change_chk_sates = True	

    # ----------------------------------------------------------------------------
    # Because I found it impossible to just redraw an ax or another
    # I must redraw all things
    #
    def draw_main_graph( self, ax_main, intraday, width ):
        global selector # for event to be called by main program  

        data = self.data.copy()
        self.ax_main = ax_main
        self.intraday = intraday
        self.width = width
        
        WIDTH_MA1 = self.mobile_average_1.get()
        WIDTH_MA2 = self.mobile_average_2.get()
        WIDTH_MA3 = self.mobile_average_3.get()
        WIDTH_MA4 = self.mobile_average_4.get()
        WIDTH_MA5 = self.mobile_average_5.get()
        WIDTH_MA6 = self.mobile_average_6.get()
        WIDTH_MAE = self.mobile_average_exp.get()
  
        self.update_interface_chk()
        
        # Abscisse axe	
        axe_x = data['Date2num']
        
        y_data = data['Close']

        length_data = len( data )
        self.message_entry.config( text=f"Data lenght: {length_data}", foreground="green" )
            
        # Permettre au Graph de dessiner
        # ------------------------------
        fighelper.set_axe( ax_main, data )

        # Set the delta selector : un seul, réutilisé (sinon un gestionnaire de clic
        # de plus est connecté à chaque redessin)
        if self.selector is None or self.selector.ax is not ax_main or self.selector.fig is not self.fig:
            self.selector = LineDeltaSelector( ax_main, self.fig, self.display )
        selector = self.selector
        selector.set_curve_data( axe_x, data['Close'] )

        # Candle Sticks
        #
        ohlc = numpy.column_stack([
            axe_x,
            data['Open'].values,
            data['High'].values,
            data['Low'].values,
            data['Close'].values
        ])

        self.candles = candlestick_ohlc(
            ax_main, 
            ohlc, 
            width=width, 
            colorup=pc.CANDLE_BULL,
            colordown=pc.CANDLE_BEAR
        )
        
        # Price
        #        
        price2 = data['Adj Close'].values
        price = data['Close'].values
        line_price, = ax_main.plot( axe_x, price, linewidth=0.9, label='Prix' )
        line_price2, = ax_main.plot( axe_x, price2, linewidth=0.9, label='Adj Close' )

  
        # Tendency Line Calculation
        # -------------------------
        # slope, intercept, rvalue, pvalue, stderr, intercept_stderr
        # 
        result = dsp.linregress( axe_x, data['Adj Close'].values )
        color=pc.CANDLE_BEAR
        if result.slope >= 0:
            color=pc.CANDLE_BULL
        line_tendency = result.slope * axe_x + result.intercept # y = a * x + b
        self.line_trend, = ax_main.plot( axe_x, line_tendency, color=color, linewidth=1, label=f"Slope", linestyle='--' )

        # Moyenne mobile calculation
        # --------------------------
        ma1 = dsp.moving_average_extended( y_data, WIDTH_MA1 )
        line_ma1, = ax_main.plot( axe_x, ma1, color="green", linewidth=0.9, label=f"ma1", linestyle='-' )

        ma2 = dsp.moving_average_extended( y_data, WIDTH_MA2 )
        line_ma2, = ax_main.plot( axe_x, ma2, color="red", linewidth=0.9, label=f"ma2", linestyle='-' )
        
        mae = dsp.moving_average_exp_extended( y_data, WIDTH_MAE )
        line_mae, = ax_main.plot( axe_x, mae, color="darkorange", linewidth=0.9, label=f"mae", linestyle='-' )

        # Colorise between movingAverage 1 & movingAverage 2
        fill1 = ax_main.fill_between( axe_x, ma1, ma2, where=(ma1 > ma2), interpolate=True, label='fill_1', color='g', alpha=0.3 )
        fill2 = ax_main.fill_between( axe_x, ma1, ma2, where=(ma1 < ma2), interpolate=True, label='fill_2', color='r', alpha=0.3 )
  
        ma3 = dsp.moving_average_rolling( y_data, WIDTH_MA3 )
        line_ma3, = ax_main.plot( axe_x, ma3, color="blue", linewidth=0.9, label=f"ma3", linestyle='-' )
  
        ma4 = dsp.moving_average_rolling( y_data, WIDTH_MA4 )
        line_ma4, = ax_main.plot( axe_x, ma4, color="deepskyblue", linewidth=0.9, label=f"ma4", linestyle='-' )
  
        ma5 = dsp.moving_average_rolling( y_data, WIDTH_MA5 )
        line_ma5, = ax_main.plot( axe_x, ma5, color="orange", linewidth=0.9, label=f"ma5", linestyle='-' )

        ma6 = dsp.moving_average_rolling( y_data, WIDTH_MA6 )
        line_ma6, = ax_main.plot( axe_x, ma6, color="purple", linewidth=0.9, label=f"ma6", linestyle='-' )
          
        # Signaux d'achat / vente et spread (voir compute_signals)
        # --------------------------------------------------------
        signals = compute_signals(
            ma1, ma2, mae,
            data['Close'].values, data['Open'].values,
            strategy_1_2=bool( self.var70.get() )
        )

        for _label, _price, _cum in signals.events:
            _str = f"{_label} -> {h.format_number3f(_price)}"
            if _cum is not None:
                _str += f" Cum_price: {h.format_number3f(_cum)}"
            print( _str )

        print( "---------------------------------" )

        # Marqueurs : signaux exécutés en plein, signaux ignorés (déjà en position
        # pour un achat, pas en position pour une vente) en pâle
        low = data['Low'].values
        high = data['High'].values
        buy_colors = [ to_rgba( 'green', 1.0 if executed else 0.25 ) for executed in signals.buy_executed ]
        sell_colors = [ to_rgba( 'red', 1.0 if executed else 0.25 ) for executed in signals.sell_executed ]

        line_stem1 = ax_main.scatter( signals.buy_bars, low[ signals.buy_bars ] * 0.99, c=buy_colors, marker='^', s=100, label='Buy Signal' )
        line_stem2 = ax_main.scatter( signals.sell_bars, high[ signals.sell_bars ] * 1.01, c=sell_colors, marker='v', s=100, label='Sell Signal' )

        self.cum_price = signals.spread
        self.trades = signals.trades
        self.hold = data['Close'].iloc[-1] - data['Close'].iloc[0] # acheter-et-garder, en points
        self.slope = result.slope

        self.lines = [line_price, line_price2, line_ma1, line_ma2, line_mae, line_ma3, line_ma4, line_ma5, line_ma6, fill1, fill2, line_stem1, line_stem2]

        return axe_x

    # ----------------------------------------------------------------------------
    # Add in title specific data for strategy
    #
    def add_title( self ):
        WIDTH_MA1 = self.mobile_average_1.get()
        WIDTH_MA2 = self.mobile_average_2.get()
        WIDTH_MAE = self.mobile_average_exp.get()

        _s = "{:.6f}".format( self.slope )
        title =  f" - Slope: {_s}"       
        title += f" - MAx/MAE: {WIDTH_MA1} {WIDTH_MA2} {WIDTH_MAE}"
        title += f" - Spread: {h.format_number3f(self.cum_price)}"
        title += f" - trades: {self.trades}"
        title += f" - B&H: {h.format_number3f(self.hold)}"
        if EXECUTION != "close" or COST_PCT > 0:
            title += f" - exec: {EXECUTION}, frais {COST_PCT}%"
        return title