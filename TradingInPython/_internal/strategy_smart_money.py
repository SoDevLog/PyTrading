""" strategy_tool - Smart Money Detection

    - Init_Check_Box : intialiser les checkbox du graphe
    
    - create_config_window
    - complete_graph_window
    - draw_main_graph
    - draw_update_graph
    - create_overlays

"""
import os
import json
import threading
import numpy
import tkinter as tk
import matplotlib.pyplot as plt
import config.func as conf
import figure.helper as fighelper
import styles.palette_colors as pc

from tkinter import ttk
from tkinterh.helper import Tooltip
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter, StrMethodFormatter
from figure.candlestick_ohlc import candlestick_ohlc

from figure.linedeltaselector import LineDeltaSelector
from figure.order_blocks_drawer import OrderBlockDrawer
from strategies.smc_engine import SMC_Engine, SMC_Params

# -----------------------------------------------------------------------------

class strategy_smart_money:
    def __init__( self, window, message_entry, display ):
        self.window = window
        self.message_entry = message_entry
        self.display = display # tkinter checkbox
        self.config = 'INTRADAY'
        self.intraday = False
        self.slope = None
        self.data = None
        self.data_smc = None
        self.params = SMC_Params()
        self.engine = SMC_Engine( self.params )
        self.init = False

        self.get_configuration()

        # Stocker les artistes graphiques
        self.artists = {
            'candles': [],
            'structure': [],
            'swings': [],
            'segments': [],
            'displacement': [],
            'market_state': [],
            'liquidity': [],
            'bos': [],
            'choch': [],
            'order_blocks': [],
            'fvg': [],
            'ote': []
        }

        # Figure et axes matplotlib
        self.fig = None
        self.ax = None
        self.canvas = None
        self.toolbar = None

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

    # ----------------------------------------------------------------------------

    def do_print( self, print_doc ):
        if print_doc:
            print( "- Maintenir la touche 'r' et cliquez pour supprimer un élément graphique." )
            print( "- Maintenir la touche 'e' et cliquez pour afficher un élément supprimé." )
            
    # ----------------------------------------------------------------------------

    def get_configuration( self ):
        self.configuration, self.path_for_configuration_file = conf.read_configuration( 'strategy_kalman_filter.json' )
        self.message_entry.config( text=f"Configuration ok", foreground="green" )

    # ----------------------------------------------------------------------------

    def command_get_configuration( self, config ):

        self.config = config

        _founded = False
        for stock in self.configuration.get( 'stocks' ):
            if stock['name'] == self.config:
                self.n_forecast_entry.set( stock.get('N_FORECAST') )
                self.process_noise_factor_entry.set( stock.get('PROCESS_NOISE_FACTOR') )
                self.measurement_noise_factor_entry.set( stock.get('MEASUREMENT_NOISE_FACTOR') )
                self.look_back_entry.set( stock.get('LOOK_BACK') )
                self.volatility_window_entry.set( stock.get('VOLATILITY_WINDOW') )
                _founded = True
                self.message_config_entry.config( text=f"{self.config}", foreground="green" )
                break

        if _founded == False:
            self.message_entry.config( text=f"Config not founded", foreground="red" )

    # ----------------------------------------------------------------------------
    # Retreive an item from stock in configration file using self.config
    #
    def get_stock( self, item ):

        for stock in self.configuration.get( 'stocks' ):
            if stock['name'] == self.config:
                return stock.get( item, 0 )

        return 0 # Erreur

	# ----------------------------------------------------------------------------

    def update_interface( self ):
        # self.process_noise_factor_entry.set( self.get_stock( 'PROCESS_NOISE_FACTOR' ) )
        # self.measurement_noise_factor_entry.set( self.get_stock( 'MEASUREMENT_NOISE_FACTOR' ) )
        # self.look_back_entry.set( self.get_stock( 'LOOK_BACK' ) )
        # self.n_forecast_entry.set( self.get_stock( 'N_FORECAST' ) )
        # self.volatility_window_entry.set( self.get_stock( 'VOLATILITY_WINDOW' ) )
        pass

    # ----------------------------------------------------------------------------

    def create_config_window( self ):

        self.window.title('Smart Money')
        content = ttk.Frame( self.window, padding=( 15, 10, 15, 10 ) ) # left top right bottom
        content.grid( column=3, row=0 )
        padding_options = {'padx': 5, 'pady': 5}

        #self.root = tk.Tk()
        self.window.protocol( "WM_DELETE_WINDOW", self.on_close )
        #self.window.title("TradingInPython - SMC Engine")
        self.stop_event = threading.Event()

        # Swing width
        _row = 0
        ttk.Label( content, text="Swing width :" ).grid( row=_row, column=0, sticky="w", **padding_options )
        self.var_sw = tk.IntVar( value=self.params.swing_width )
        ttk.Spinbox( content, from_=2, to=10, textvariable=self.var_sw, width=8, command=self.on_spinbox_click).grid( row=_row, column=1, **padding_options )

        # Liquidity threshold
        _row += 1
        ttk.Label( content, text="Liquidity threshold :" ).grid( row=_row, column=0, sticky="w", **padding_options )
        self.var_liquidity = tk.DoubleVar( value=self.params.liquidity_threshold )
        ttk.Spinbox( content, from_=0.001, to=0.05, increment=0.001,
            textvariable=self.var_liquidity, width=8).grid( row=_row, column=1, **padding_options )

        # ATR period
        _row += 1
        ttk.Label( content, text="ATR period :").grid( row=_row, column=0, sticky="w", **padding_options )
        self.var_atr = tk.IntVar( value=self.params.atr_period )
        ttk.Spinbox( content, from_=5, to=50, textvariable=self.var_atr, width=8).grid( row=_row, column=1, **padding_options )

        # Displacement threshold
        _row += 1
        ttk.Label( content, text="Displacement ratio :" ).grid( row=_row, column=0, sticky="w", **padding_options )
        self.var_disp = tk.DoubleVar( value=self.params.displacement_body_ratio )
        ttk.Spinbox( content, from_=0.1, to=0.95, increment=0.05,
            textvariable=self.var_disp, width=8).grid( row=_row, column=1, **padding_options )

        # Minimum body vs ATR to mark FVG
        _row += 1
        ttk.Label( content, text="FVG displacement ATR :" ).grid( row=_row, column=0, sticky="w", **padding_options )
        self.var_fvg = tk.DoubleVar( value=self.params.fvg_displacement_atr_ratio )
        ttk.Spinbox( content, from_=0.1, to=0.7, increment=0.05,
            textvariable=self.var_fvg, width=8).grid( row=_row, column=1, **padding_options )
        
        # ---------------------------------------------------------------------
        
        _row += 1
        _tki = ttk.Button( content, text="Reset", command=self.reset_smc_params )
        Tooltip( _tki, "Réinitialiser les paramètres au moteur SMC" )
        _tki.grid( row=_row, column=0, columnspan=2, padx=5, pady=(15,5) )
        
        _row += 1
        _tki = ttk.Button( content, text="Apply", command=self.run_engine_smc_and_redraw )
        Tooltip( _tki, "Appliquer les nouveaux paramètres au moteur SMC et redessiner le graph" )
        _tki.grid( row=_row, column=0, columnspan=2, padx=5, pady=(5,15) )
        
        # _row += 1
        # _tti = ttk.Button( content, text="Redraw", command=self.draw_update_graph )
        # Tooltip( _tti, "Redessiner le graphe avec les nouveaux paramètres" )
        # _tti.grid( row=_row, column=0, columnspan=2, **padding_options )
        
    # ----------------------------------------------------------------------------

    def complete_graph_window( self, check_frame, command_update_graphs, command_update_lines ):

        self.command_update_graphs = command_update_graphs

        # Init_Check_Box
        self.show_all = tk.BooleanVar( value=True )
        self.show_candles = tk.BooleanVar( value=True )
        self.show_swings = tk.BooleanVar( value=False )
        self.show_structure = tk.BooleanVar( value=True )
        self.show_segments = tk.BooleanVar( value=True )
        self.show_market_state = tk.BooleanVar( value=True )
        self.show_displacement = tk.BooleanVar( value=False )
        self.show_liquidity = tk.BooleanVar( value=True )
        self.show_bos = tk.BooleanVar( value=True )
        self.show_choch = tk.BooleanVar( value=True )
        self.show_order_blocks = tk.BooleanVar( value=False )
        self.show_fvg = tk.BooleanVar( value=False )
        self.show_ote = tk.BooleanVar( value=False )

        self.check_boxes = [
            ("All", self.show_all, 'all'),
            ("Candles", self.show_candles, 'candles'),
            ("Swings", self.show_swings, 'swings'),
            ("Structure", self.show_structure, 'structure'),
            ("Segments", self.show_segments, 'segments'),
            ("Market state", self.show_market_state, 'market_state'),
            ("Displacement", self.show_displacement, 'displacement'),
            ("Liquidity", self.show_liquidity, 'liquidity'),
            ("BOS", self.show_bos, 'bos'),
            ("CHoCH", self.show_choch, 'choch'),
            ("Order Blocks", self.show_order_blocks, 'order_blocks'),
            ("FVG", self.show_fvg, 'fvg'),
            ("OTE", self.show_ote, 'ote'),
        ]

        for i, (txt, var, key) in enumerate( self.check_boxes ):
            cb = ttk.Checkbutton( check_frame, text=txt, variable=var,
                command=lambda k=key: self.toggle_overlay(k) )
            cb.grid( row=0, column=i, sticky="w", padx=5, pady=2 )

    # -------------------------------------------------------------------------

    def on_close( self ):
        self.stop_event.set()

        try:
            plt.close( 'all' )
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass

        os._exit( 0 )

    # -------------------------------------------------------------------------

    def update_smc_params( self ):
        self.params.swing_width = self.var_sw.get()
        self.params.liquidity_threshold = self.var_liquidity.get()
        self.params.atr_period = self.var_atr.get()
        self.params.displacement_body_ratio = self.var_disp.get()
        self.params.fvg_displacement_atr_ratio = self.var_fvg.get()
        print( f"swing_width: {self.params.swing_width}" )
        print( f"liquidity_threshold: {self.params.liquidity_threshold}" )
        print( f"atr_period: {self.params.atr_period}" )
        print( f"displacement_body_ratio: {self.params.displacement_body_ratio}" )
        print( f"fvg_displacement_atr_ratio: {self.params.fvg_displacement_atr_ratio}" )

    # -------------------------------------------------------------------------

    def reset_smc_params( self ):
        smc = SMC_Params()
        self.var_sw.set( smc.swing_width )
        self.var_liquidity.set( smc.liquidity_threshold )
        self.var_atr.set( smc.atr_period )
        self.var_disp.set( smc.displacement_body_ratio )
        self.var_fvg.set( smc.fvg_displacement_atr_ratio )

    # -------------------------------------------------------------------------

    def run_engine_smc_and_redraw( self ):
        self.update_smc_params()
        self.engine = SMC_Engine( self.params )
        self.data_smc = self.engine.apply( self.data )
        print( "----------------- SMC paramètres appliqués." )
        self.draw_update_graph()
        print( "----------------- GRAPH redessiné." )
        
    # ----------------------------------------------------------------------------
    # API
    #
    def toggle_visibility( self ):
        pass

    # -------------------------------------------------------------------------

    def toggle_overlay( self, overlay_key ):
        """Active ou désactive la visibilité d'un overlay"""
        if self.ax is None:
            return

        visible = getattr( self, f'show_{overlay_key}' ).get()

        if overlay_key != 'all':
            # Modifier la visibilité de tous les artistes de cet overlay
            for artist in self.artists[ overlay_key ]:
                artist.set_visible( visible )
        else:
            for i, (txt, var, key) in enumerate( self.check_boxes ):
                if key != 'all':
                    for artist in self.artists[ key ]:
                        artist.set_visible( visible )
                        if self.show_all.get():
                            var.set( True )
                        else:
                            var.set( False )

        # Redessiner le canvas
        if self.canvas:
            self.canvas.draw_idle()

    # -------------------------------------------------------------------------

    def on_spinbox_click( self ):
        self.run_engine_smc_and_redraw()

    # -------------------------------------------------------------------------
    # 'r' to remove element
    # 'e' to set visible element
    # -------------------------------------------------------------------------
    #
    def on_click( self, event ) :
        if event.inaxes != self.ax:
            return

        if event.button == 1 and event.key == 'r' or event.key == 'e':

            ALLOWED = ( Line2D, Patch, Text )

            for artist in list( self.ax.get_children() ):
                if not isinstance( artist, ALLOWED ):
                    continue

                # exclure éléments structurels
                if artist.axes is None:
                    continue

                try:
                    contains, _ = artist.contains(event)
                except Exception:
                    continue

                if not contains:
                    continue

                try:
                    #artist.remove()
                    if event.key == 'r': # not visible
                        artist.set_visible( False )
                    if event.key == 'e': # set visible
                        artist.set_visible( True )
                    event.canvas.draw_idle()
                except NotImplementedError:
                    # artist non supprimable (ticks, spines, etc.)
                    continue

    # -------------------------------------------------------------------------

    def plot_price( self, ax, df ):

        df = df.copy()
        #df.index = df.index.tz_convert("Europe/Paris")

        # Indices séquentiels
        indices = numpy.arange( len( df ) )

        # Préparer OHLC
        ohlc = numpy.column_stack([
            indices,
            df['Open'].values,
            df['High'].values,
            df['Low'].values,
            df['Close'].values
        ])

        width = 0.4
        candlestick_ohlc(
            ax,
            ohlc,
            width=width,
            colorup= pc.CANDLE_BULL,
            colordown= pc.CANDLE_BEAR
        )

        self.artists[ 'candles' ] = ax.collections + ax.patches + ax.lines
        self.toggle_overlay( 'candles' ) # must be done because candlestick_ohlc has no visible param

    # ----------------------------------------------------------------------------
    # Because I found it impossible to just redraw an ax or another
    # I must redraw all things
    #
    def draw_main_graph( self, ax_main, intraday, width ):
        global selector # for event to be called by main program
        global drawer # for event to be called by main program     
        self.ax = ax_main
        self.intraday = intraday

        length_data = len( self.data )
        self.message_entry.config( text=f"Data lenght: {length_data}", foreground="green" )

        # Abscisse axe
        self.axe_x = self.data['Date2num']

        # Permettre au Graph de dessiner
        #
        fighelper.set_axe( self.ax, self.data )

        # Set the delta selector
        selector = LineDeltaSelector( ax_main, self.fig, self.display )
        selector.set_curve_data( self.data['Date2num'], self.data['Close'] )

        # Dessiner les Order Blocks sur le graphique
        drawer = OrderBlockDrawer( ax_main, self.data )
        drawer.do_print( self.init == False)
        self.do_print( self.init==False )
        self.init = True
        
        # Appliquer l'SMC Engine
        #
        self.data_smc = self.engine.apply( self.data )
        if self.data_smc is None:
            print( "Veuillez d'abord appliquer SMC." )
            return

        # Réinitialiser les artistes
        for key in self.artists:
            self.artists[ key ] = []

        # Connecter le clic souris
        self.fig.canvas.mpl_connect( "button_press_event", self.on_click )

        # Tout afficher
        self.plot_price( self.ax, self.data_smc )

        # Créer les overlays et stocker les artistes
        self.create_overlays( self.ax, self.data_smc )

        if self.intraday:
            _date_mask = '%d-%m %H:%M'
        else:
            _date_mask = '%d-%m-%y'

        # Fonction pour convertir index -> date
        def _format_date( x, pos=None ): # pos resuired by FuncFormatter
            idx = int( x )
            if 0 <= idx < len(self.data_smc):
                return self.data_smc.index[idx].strftime( _date_mask )
            return ''

        self.ax.yaxis.set_major_formatter( StrMethodFormatter('{x:.2f}') )
        self.ax.xaxis.set_major_formatter( FuncFormatter( _format_date ) )
        self.ax.set_xlim( -0.5, len( self.data ) - 0.5 )
        self.ax.grid( True, alpha=0.8 )
        self.ax.set_autoscale_on( False )

        return self.axe_x, self.ax

    # -------------------------------------------------------------------------

    def create_overlays( self, ax, df ):
        """Créer les overlays et stocker les artistes"""

        # Structure
        self.engine.overlays_swings(
            df=df,
            ax=ax,
            artists=self.artists,
            key='swings',
            visible_start=self.show_swings.get()
        )

        # Structure
        self.engine.overlays_structure(
            df=df,
            ax=ax,
            artists=self.artists,
            key='structure',
            visible_start=self.show_structure.get()
        )

        # Segments
        self.engine.overlays_segments(
            df=df,
            ax=ax,
            artists=self.artists,
            key='segments',
            visible_start=self.show_segments.get()
        )

        # Market state
        self.engine.overlays_market_state(
            df=df,
            ax=ax,
            artists=self.artists,
            key='market_state',
            visible_start=self.show_market_state.get()
        )

        # Displacement
        self.engine.overlays_displacement(
            df=df,
            ax=ax,
            artists=self.artists,
            key='displacement',
            visible_start=self.show_displacement.get()
        )

        # Liquidity
        self.engine.overlays_liquidity(
            df=df,
            ax=ax,
            artists=self.artists,
            key='liquidity',
            visible_start=self.show_liquidity.get()
        )
        
        # BOS
        self.engine.overlays_bos(
            df=df,
            ax=ax,
            artists=self.artists,
            key='bos',
            visible_start=self.show_bos.get()
        )

        # CHoCH
        self.engine.overlays_choch(
            df=df,
            ax=ax,
            artists=self.artists,
            key='choch',
            visible_start=self.show_choch.get()
        )

        # FVG
        self.engine.overlays_fvg(
            df=df,
            ax=ax,
            artists=self.artists,
            key='fvg',
            visible_start=self.show_fvg.get()
        )

        # Order Blocks ICT BoS
        self.engine.overlays_order_blocks(
            df=df,
            ax=ax,
            artists=self.artists,
            key='order_blocks',
            visible_start=self.show_order_blocks.get()
        )
        
        # OTE
        self.engine.overlays_ote(
            df=df,
            ax=ax,
            artists=self.artists,
            key='ote',
            visible_start=self.show_ote.get()
        )

    # -------------------------------------------------------------------------

    # Keep the update meaning
    #
    def draw_update_graph( self ):

        # Nettoyer l'axe existant
        self.ax.clear()

        self.draw_main_graph( self.ax, self.intraday, width=0.8 )

        # Recreate all annotations for ax_main
        self.command_update_graphs()

        self.ax.set_title( self.template.format( self.add_title() ) )

    # ----------------------------------------------------------------------------
    # Add in title specific data for strategy
    #
    def add_title( self ):

        return f" Smart Money"

