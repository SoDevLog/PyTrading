""" strategy_tool- Ichimoku - Ichimoku Kinko Hyo Modernised

    - get_configuration
    - create_config_window
    - config_window
    - draw_main_graph
    - add_title

"""
import json
import pandas
import numpy
import tkinter as tk
import config.func as conf
import digitsignalprocessing.func as dsp
import figure.helper as fighelper
import styles.palette_colors as pc

from tkinter import ttk
from tkinterh.helper import Tooltip
from matplotlib.ticker import FuncFormatter, StrMethodFormatter
from figure.candlestick_ohlc import candlestick_ohlc, set_visibility_ohlc
from digitsignalprocessing import ichimoku_kinko_hyo
from figure.linedeltaselector import LineDeltaSelector

# Sting in config file 
# TODO: put it in config.func
class CONFIG:
    default = 'DEFAULT'
    short = 'DEFAULT_SHORT'
    long = 'DEFAULT_LONG'

# -----------------------------------------------------------------------------

class strategy_ichimoku:
    def __init__( self, window, message_entry, intraday, display ):
        self.window = window
        self.message_entry = message_entry
        self.intraday = intraday
        self.display = display # tkinter checkbox
        self.name = None
        self.mobile_average_1 = None
        self.mobile_average_2 = None
        self.mobile_average_3 = None
        self.nb_days_forcasted = 3 # by default
        self.nb_days_past = 9 # by default
        self.slope = None
        self.data = None
        self.MAKE_FORCASTING = False
        self.artists_ichimoku = {}

        # Read Configuration to configure Tkinter UI
        self.get_configuration()

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

    def create_config_window( self ):

        self.window.title('Ichimoku')
        content = ttk.Frame( self.window, padding=( 15, 10, 15, 10 ) ) # left top right bottom
        content.grid( column=3, row=0 )
        tk_row = 0
        padding_options = {'padx': 5, 'pady': 5}

        ttk.Label( content, text="Tenkan sen (MA1) :").grid( row=tk_row, column=0, sticky="e" )
        self.mobile_average_1 = tk.IntVar( value=self.get_stock('MA1') )
        _tki = ttk.Spinbox(
            content,
            from_=1,
            to=100,
            increment=1,
            width=8,
            textvariable=self.mobile_average_1,
            command=self.on_mobile_average_params_change
        )
        _tki.grid( row=tk_row, column=1, sticky="w", **padding_options )
        _tki.bind( "<Return>", lambda e: self.on_mobile_average_params_change() )
        _tki.bind( "<FocusOut>", lambda e: self.on_mobile_average_params_change() )
        
        tk_row += 1

        # ------------------------------------------------------------------------

        ttk.Label( content, text="Kijun sen (MA2) :").grid( row=tk_row, column=0, sticky="e" )
        self.mobile_average_2 = tk.IntVar( value=self.get_stock('MA2') )
        _tki = ttk.Spinbox(
            content,
            from_=1,
            to=100,
            increment=1,
            width=8,
            textvariable=self.mobile_average_2,
            command=self.on_mobile_average_params_change
        )
        _tki.grid( row=tk_row, column=1, sticky="w", **padding_options )
        _tki.bind( "<Return>", lambda e: self.on_mobile_average_params_change() )
        _tki.bind( "<FocusOut>", lambda e: self.on_mobile_average_params_change() )
        
        tk_row += 1

        # ------------------------------------------------------------------------

        ttk.Label( content, text="Senkou span (MA3) :").grid( row=tk_row, column=0, sticky="e" )
        self.mobile_average_3 = tk.IntVar( value=self.get_stock('MA3') )
        _tki = ttk.Spinbox(
            content,
            from_=1,
            to=100,
            increment=1,
            width=8,
            textvariable=self.mobile_average_3,
            command=self.on_mobile_average_params_change
        )
        _tki.grid( row=tk_row, column=1, sticky="w", **padding_options )
        _tki.bind( "<Return>", lambda e: self.on_mobile_average_params_change() )
        _tki.bind( "<FocusOut>", lambda e: self.on_mobile_average_params_change() )
        
        tk_row += 1

        # ------------------------------------------------------------------------

        separator = ttk.Separator( content, orient='horizontal' )
        separator.grid( row=tk_row, column=0, columnspan=2, sticky="ew", **padding_options)

        tk_row += 1

        # ------------------------------------------------------------------------

        label = ttk.Label( content, text="Nb days forcasted :")
        label.grid( row=tk_row, column=0, sticky="e" )
        Tooltip( label, "Nombre de jours prédits dans le futur")
        self.nb_days_forcasted_entry = tk.IntVar( value=self.nb_days_forcasted )
        _tki = ttk.Entry( content, width=10, textvariable=self.nb_days_forcasted_entry )
        _tki.grid( row=tk_row, column=1, sticky="w", **padding_options )

        tk_row += 1

          # ------------------------------------------------------------------------

        label = ttk.Label( content, text="Nb days in past :")
        label.grid( row=tk_row, column=0, sticky="e" )
        Tooltip( label, "Fenêtre de prédiction")
        self.nb_days_past_entry = tk.IntVar( value=self.nb_days_past )
        _tki = ttk.Entry( content, width=10, textvariable=self.nb_days_past_entry )
        _tki.grid( row=tk_row, column=1, sticky="w", **padding_options )

        tk_row += 1

          # ------------------------------------------------------------------------

        separator = ttk.Separator( content, orient='horizontal' )
        separator.grid( row=tk_row, column=0, columnspan=2, sticky="ew", **padding_options)

        tk_row += 1

        # ------------------------------------------------------------------------

        _tki = ttk.Button(content, text="Default Short", command=lambda: self.command_get_default_configurations( CONFIG.short ) )
        _tki.grid( row=tk_row, column=0,  columnspan=2, **padding_options  )

        tk_row += 1

        _tki = ttk.Button( content, text="Default", command=lambda: self.command_get_default_configurations( CONFIG.default ) )
        _tki.grid( row=tk_row, column=0,  columnspan=2, **padding_options  )

        tk_row += 1

        _tki = ttk.Button( content, text="Default Long", command=lambda: self.command_get_default_configurations( CONFIG.long ) )
        _tki.grid( row=tk_row, column=0,  columnspan=2, **padding_options  )

        tk_row += 1

        _tki = ttk.Button( content, text="Read", command=self.command_get_item_configration )
        Tooltip( _tki, "Lire les valeurs pour cette action" )
        _tki.grid( row=tk_row, column=0,  columnspan=2, **padding_options  )

        tk_row += 1

        _tki = ttk.Button( content, text="Save", style="Custom.TButton", command=self.command_save_configuration )
        Tooltip( _tki, "Sauver les valeurs pour cette action" )
        _tki.grid( row=tk_row, column=0,  columnspan=2, **padding_options  )

        tk_row += 1

    # ----------------------------------------------------------------------------

    def command_get_default_configurations( self,  config ):

        for stock in self.configuration.get( 'stocks' ):
            if stock['name'] == config:
                self.mobile_average_1.set( stock.get('MA1') )
                self.mobile_average_2.set( stock.get('MA2') )
                self.mobile_average_3.set( stock.get('MA3') )
                break

        self.message_entry.config( text=f"Config {config}", foreground="green" )
        
        self.on_mobile_average_params_change()

    # ----------------------------------------------------------------------------

    def command_get_item_configration( self ):

        for stock in self.configuration.get( 'stocks' ):
            founded = False
            if stock['name'] == self.name:
                self.mobile_average_1.set( stock.get('MA1') )
                self.mobile_average_2.set( stock.get('MA2') )
                self.mobile_average_3.set( stock.get('MA3') )
                founded = True
                break

        if founded == False:
            self.command_get_default_configurations( CONFIG.default )

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
        self.configuration, self.path_for_configuration_file = conf.read_configuration( 'strategy_ichimoku.json' )

        forcast = self.configuration.get( 'forcast' )
        self.nb_days_forcasted = forcast[0].get('NB_DAYS_FORCASTED')
        self.nb_days_past = forcast[0].get('NB_DAYS_IN_PAST')

    # ----------------------------------------------------------------------------

    def command_save_configuration( self ):
        self.save_configuration()
        self.get_configuration()
        self.update_interface()
        self.message_entry.config( text=f"Configuration saved", foreground="orange" )

    # ----------------------------------------------------------------------------

    def save_configuration( self ):

        self.configuration['forcast'][0]['NB_DAYS_FORCASTED'] = self.nb_days_forcasted_entry.get()
        self.configuration['forcast'][0]['NB_DAYS_IN_PAST'] = self.nb_days_past_entry.get()

        # Take DEFAULT values
        for stock in self.configuration.get( 'stocks' ):
            if stock['name'] == 'DEFAULT':
                _default_ma1 = stock.get('MA1')
                _default_ma2 = stock.get('MA2')
                _default_ma3 = stock.get('MA3')
                break

        # Values are the same has DEFAULT
        # user retake DEFAULT values
        # we can suppress in configuration file
        #
        if _default_ma1 == self.mobile_average_1.get() \
            and _default_ma2 == self.mobile_average_2.get() \
            and _default_ma3 == self.mobile_average_3.get():
            # Suppress element that it's like DEFAULT
            self.configuration['stocks'] = [stock for stock in self.configuration['stocks'] if stock['name'] != self.name]
            with open( self.path_for_configuration_file, "w" ) as file:
                json.dump( self.configuration, file, indent=4 )
            return

        # Update otherwise Create
        for stock in self.configuration['stocks']:
            if stock['name'] == self.name:
                # Update
                stock['MA1'] = self.mobile_average_1.get()
                stock['MA2'] = self.mobile_average_2.get()
                stock['MA3'] = self.mobile_average_3.get()
                with open( self.path_for_configuration_file, "w" ) as file:
                    json.dump( self.configuration, file, indent=4 )
                return

        # Create new item from Tkinter interface
        _config = {
            "name": self.name,
            "MA1": self.mobile_average_1.get(),
            "MA2": self.mobile_average_2.get(),
            "MA3": self.mobile_average_3.get()
        }

        # Finaly Append (Created item)
        self.configuration.get('stocks').append( _config )
        with open( self.path_for_configuration_file, "w" ) as file:
            json.dump( self.configuration, file, indent=4 )

    # ----------------------------------------------------------------------------

    def update_interface( self ):
        self.mobile_average_1.set( self.get_stock( 'MA1' ) )
        self.mobile_average_2.set( self.get_stock( 'MA2' ) )
        self.mobile_average_3.set( self.get_stock( 'MA3' ) )
        self.nb_days_forcasted_entry.set( self.nb_days_forcasted )
        self.nb_days_past_entry.set( self.nb_days_past )

    # ----------------------------------------------------------------------------

    def complete_graph_window( self, check_frame, command_update_graphs, command_update_lines ):

        # Create checkbox for drawing lines
        self.var_price = tk.IntVar( value=0 )
        self.var_candles = tk.IntVar( value=1 )
        self.var_tenkan_sen = tk.IntVar( value=1 )
        self.var_kijun_sen = tk.IntVar( value=1 )
        self.var_kijun_upper = tk.IntVar( value=0 )
        self.var_kijun_lower = tk.IntVar( value=0 )
        self.var_kijun_bands = tk.IntVar( value=1 )
        self.var_chikou = tk.IntVar( value=1 )
        self.var_kumo = tk.IntVar( value=1 )

        self.var_make_forcasting = tk.BooleanVar( value=False )
        self.ichimoku_futuriste = tk.BooleanVar( value=False )

        chk20 = ttk.Checkbutton( check_frame, text="Price", variable=self.var_price, command=command_update_lines)
        chk_candles = ttk.Checkbutton( check_frame, text="Candle", variable=self.var_candles, command=command_update_lines)

        chk31 = ttk.Checkbutton( check_frame, text="Tenkan", variable=self.var_tenkan_sen, command=command_update_lines)
        Tooltip( chk31, "Momentum court terme")
        
        chk32 = ttk.Checkbutton( check_frame, text="Kijun", variable=self.var_kijun_sen, command=command_update_lines)
        Tooltip( chk32, "Equilibre long terme")
        
        chk33 = ttk.Checkbutton( check_frame, text="Kj Up", variable=self.var_kijun_upper, command=command_update_lines)
        chk34 = ttk.Checkbutton( check_frame, text="Kj Down", variable=self.var_kijun_lower, command=command_update_lines)
        chk35 = ttk.Checkbutton( check_frame, text="Kj Bands", variable=self.var_kijun_bands, command=command_update_lines)
        Tooltip( chk35, "Canal de volatilité Kijun ATR")
        
        chk36 = ttk.Checkbutton( check_frame, text="Chikou", variable=self.var_chikou, command=command_update_lines)
        Tooltip( chk36, "Signal retardé, confirmation de la tendance")
        
        # Double visibly lines
        chk40 = ttk.Checkbutton( check_frame, text="Kumo", variable=self.var_kumo, command=command_update_lines)
        Tooltip( chk40, "Nuage des équilibres du marché.")

        # CONTINUS, MAKE_FORCASTING
        #chk60 = ttk.Checkbutton( check_frame, text="CONTINUS", variable=self.var60, command=command_update_graphs)
        chk70 = ttk.Checkbutton( check_frame, text="Forcasting", variable=self.var_make_forcasting, command=command_update_graphs)
        Tooltip( chk70, "Prévisions modèle de machine learning entraîné sur les données historiques.")
        
        chk80 = ttk.Checkbutton( check_frame, text="Futuriste", variable=self.ichimoku_futuriste, command=command_update_graphs)
        Tooltip( chk80, "Projecter le nuage Kumo pour visualiser les zones de support et résistance futures.")
        
        # PACK ALL THINGS
        #chk60.pack( side=tk.LEFT, padx=5, pady=5 )
        chk70.pack( side=tk.LEFT, padx=5, pady=5 )
        chk80.pack( side=tk.LEFT, padx=5, pady=5 )
        chk20.pack( side=tk.LEFT, padx=5, pady=5 )
        chk_candles.pack( side=tk.LEFT, padx=5, pady=5 )

        chk31.pack( side=tk.LEFT, padx=5, pady=5 )
        chk32.pack( side=tk.LEFT, padx=5, pady=5 )
        chk33.pack( side=tk.LEFT, padx=5, pady=5 )
        chk34.pack( side=tk.LEFT, padx=5, pady=5 )
        chk35.pack( side=tk.LEFT, padx=5, pady=5 )
        chk36.pack( side=tk.LEFT, padx=5, pady=5 )

        chk40.pack( side=tk.LEFT, padx=5, pady=5 )

    # ----------------------------------------------------------------------------

    def toggle_visibility( self ):
        candles = self.candles

        # Price
        if self.var_price.get():
            self.line_price.set_visible( True )
        else:
            self.line_price.set_visible( False )

        # Tenkan sen
        if self.var_tenkan_sen.get():
            self.artists_ichimoku['line_tenkan_sen'].set_visible( True )
        else:
            self.artists_ichimoku['line_tenkan_sen'].set_visible( False )
        
        # Kijun sen
        if self.var_kijun_sen.get():
            self.artists_ichimoku['line_kijun_sen'].set_visible( True )
        else:
            self.artists_ichimoku['line_kijun_sen'].set_visible( False )
        
        # Kijun sen upper
        if self.var_kijun_upper.get():
            self.artists_ichimoku['line_kijun_sen_upper'].set_visible( True )
        else:
            self.artists_ichimoku['line_kijun_sen_upper'].set_visible( False )

        # Kijun sen lower
        if self.var_kijun_lower.get():
            self.artists_ichimoku['line_kijun_sen_lower'].set_visible( True )
        else:
            self.artists_ichimoku['line_kijun_sen_lower'].set_visible( False )
        
        # Kijun bands
        if self.var_kijun_bands.get():
            self.artists_ichimoku['fill_Kijun_bands'].set_visible( True )
        else:
            self.artists_ichimoku['fill_Kijun_bands'].set_visible( False )
            
        # Chikou span
        if self.var_chikou.get():
            self.artists_ichimoku['line_kijun_chikou_span'].set_visible( True )
        else:
            self.artists_ichimoku['line_kijun_chikou_span'].set_visible( False )
                        
        # Kumo cloud
        if self.var_kumo.get():
            self.artists_ichimoku['fill_senkou_bands_A'].set_visible( True )
            self.artists_ichimoku['fill_senkou_bands_B'].set_visible( True )
        else:
            self.artists_ichimoku['fill_senkou_bands_A'].set_visible( False )
            self.artists_ichimoku['fill_senkou_bands_B'].set_visible( False )

        # Candles
        if self.var_candles.get():
            set_visibility_ohlc( candles, True )
        else:
            set_visibility_ohlc( candles, False )
            
        if self.MAKE_FORCASTING:
            if self.var_make_forcasting.get():
                self.artists_ichimoku['line_signal'].set_visible( True )
                self.artists_ichimoku['line_forcasted_signal'].set_visible( True )
                self.artists_ichimoku['line_predicted_signal'].set_visible( True )
            else:
                self.artists_ichimoku['line_signal'].set_visible( False )
                self.artists_ichimoku['line_forcasted_signal'].set_visible( False )
                self.artists_ichimoku['line_predicted_signal'].set_visible( False )

    # ----------------------------------------------------------------------------
    
    def on_mobile_average_params_change( self ):
        try:
            ma1 = self.mobile_average_1.get()
            ma2 = self.mobile_average_2.get()
            ma3 = self.mobile_average_3.get()

            self.message_entry.config( text=f"MA changed: {ma1} {ma2} {ma3}", foreground="green" )
            
        except (tk.TclError, ValueError):
            # Saisie en cours / valeur temporairement invalide
            return
        
        # Bloquer l'autoscale pour que le graphe reste en place
        self.ax_main.set_autoscale_on( False )

        # Supprimer les anciens éléments graphiques
        for e in self.artists_ichimoku:
            try:
                self.artists_ichimoku[ e ].remove()
            except ( NotImplementedError, ValueError ):
                pass

        WIDTH_MA1 = self.mobile_average_1.get()
        WIDTH_MA2 = self.mobile_average_2.get()
        WIDTH_MA3 = self.mobile_average_3.get()

        # Recalculer uniquement les objets impacté par la modif
        data = self.data.copy()
        data.set_index( 'Date2num', inplace=True )
        axe_x = numpy.arange( len( data ) )

        data = ichimoku_kinko_hyo.ichimoku_modernise( data, period1=WIDTH_MA1, period2=WIDTH_MA2, period3=WIDTH_MA3  )

        self.artists_ichimoku['line_tenkan_sen'], = self.ax_main.plot( axe_x, data['Tenkan_sen'], label='Tenkan-sen', color=pc.TENKAN, linewidth=1.5 )
        self.artists_ichimoku['line_kijun_sen'], = self.ax_main.plot( axe_x, data['Kijun_sen'], label='Kijun-sen', color=pc.KIJUN, linewidth=1.5 )
        self.artists_ichimoku['line_kijun_sen_upper'], = self.ax_main.plot( axe_x, data['Kijun_sen_upper'], label='Kijun-sen Upper', linestyle='-', linewidth=0.5, color='blue' )
        self.artists_ichimoku['line_kijun_sen_lower'], = self.ax_main.plot( axe_x, data['Kijun_sen_lower'], label='Kijun-sen Lower', linestyle='-', linewidth=0.5, color='blue' )
        self.artists_ichimoku['line_kijun_chikou_span'], = self.ax_main.plot( axe_x, data['Chikou_span'], label='Chikou-span', linestyle='--', linewidth=1, color=pc.royalblue )
        self.artists_ichimoku['fill_Kijun_bands'] = self.ax_main.fill_between( axe_x, data['Kijun_sen_upper'], data['Kijun_sen_lower'], where=data['Kijun_sen_upper'] >= data['Kijun_sen_lower'], facecolor='lightskyblue', edgecolor='none', alpha=0.3, interpolate=True )

        self.artists_ichimoku['fill_senkou_bands_A'] = self.ax_main.fill_between( axe_x, data['Senkou_span_A'], data['Senkou_span_B'], where=data['Senkou_span_A'] >= data['Senkou_span_B'], facecolor='lightgreen', edgecolor='none', alpha=0.7, interpolate=True )
        self.artists_ichimoku['fill_senkou_bands_B'] = self.ax_main.fill_between( axe_x, data['Senkou_span_A'], data['Senkou_span_B'], where=data['Senkou_span_A'] < data['Senkou_span_B'], facecolor='lightcoral', edgecolor='none', alpha=0.7, interpolate=True )
        
        self.toggle_visibility()
        
        self.canvas.draw_idle()
        
    # ----------------------------------------------------------------------------
    # Because I found it impossible to just redraw an ax or another
    # I must redraw all things
    #
    def draw_main_graph( self, ax_main, intraday, width ):
        global selector # for event to be called by main program

        PROJECTION = self.ichimoku_futuriste.get()

        data = self.data #.copy()

        WIDTH_MA1 = self.mobile_average_1.get()
        WIDTH_MA2 = self.mobile_average_2.get()
        WIDTH_MA3 = self.mobile_average_3.get()

        # Abscisse axe
        axe_x = data['Date2num']
        self.ax_main = ax_main

        if PROJECTION:
            last_date = data.index[-1]

            future_dates = pandas.date_range(
                start=last_date + pandas.Timedelta(days=1),
                periods=WIDTH_MA2,
                freq='B'
            )

            future_df = pandas.DataFrame( index=future_dates )

            # Colonnes OHLC vides
            for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
                future_df[col] = numpy.nan

            # Concat historique + futur
            data = pandas.concat( [data, future_df] )
            future_mask = data['Close'].isna()

            data['Date2num'] = numpy.arange(len(data))
            axe_x = data['Date2num']

        length_data = len( data )
        if length_data < WIDTH_MA3:
            self.message_entry.config( text=f"Pas assez de connées: {length_data}", foreground="red" )
            return axe_x, False

        self.message_entry.config( text=f"Data length: {length_data}", foreground="green" )

        # Check for forcasting enabled
        _forcasting = self.var_make_forcasting.get()
        if intraday and _forcasting:
            _forcasting = False # disable for intraday
            self.message_entry.config( text="Prévisions désactivées en intraday.", foreground="red" )
            self.var_make_forcasting.set( False )

        # Set new state
        self.MAKE_FORCASTING = _forcasting

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
            colorup= pc.CANDLE_BULL,
            colordown= pc.CANDLE_BEAR
        )

        # Permettre au Graph de dessiner
        # ------------------------------
        fighelper.set_axe( ax_main, data )

        # Set the delta selector
        selector = LineDeltaSelector( ax_main, self.fig, self.display )
        selector.set_curve_data( axe_x, data['Close'] )

        # Tendency Line Calculation
        # -------------------------
        # slope, intercept, rvalue, pvalue, stderr, intercept_stderr
        #
        result = dsp.linregress( axe_x, data['Adj Close'].values )
        self.slope = result.slope
        color='darkorange'
        if result.slope >= 0:
            color='darkgreen'
        line_tendency = result.slope * axe_x + result.intercept # y = a * x + b
        ax_main.plot( axe_x, line_tendency, color=color, linewidth=1, label=f"Tendance", linestyle='--', alpha=0.5 )

        # -------------------
        # ichimoku_kinko_hyo
        # -------------------
        data = ichimoku_kinko_hyo.ichimoku_modernise( data, period1=WIDTH_MA1, period2=WIDTH_MA2, period3=WIDTH_MA3  )

        if PROJECTION:
            # Effacer les projection futures des signaux Tenkan, Kijun et Chikou span
            data.loc[future_mask, 'Tenkan_sen'] = numpy.nan
            data.loc[future_mask, 'Kijun_sen'] = numpy.nan

        if self.MAKE_FORCASTING:
            # Generer le signal d'achat/vente
            data = ichimoku_kinko_hyo.generate_signals( data )

            # Entraîner le modèle
            model = ichimoku_kinko_hyo.train_predictive_model( data )

            # Appliquer le modèle pour prédire les signaux futurs
            data = ichimoku_kinko_hyo.apply_model( model, data )

            days_in_futur = self.nb_days_forcasted_entry.get()
            days_in_past = self.nb_days_past_entry.get() # 9 # 26 # 52

            # Créer des périodes futures pour les prévisions
            future_dates = pandas.date_range( start=data['DateSaved'].max() + pandas.Timedelta(days=1), periods=days_in_futur, freq='B')
            future_df = pandas.DataFrame( index=future_dates )
            future_df['Signal_Predicted'] = numpy.nan  # Initialiser les signaux prédits futurs à NaN

            # Joindre les données historiques avec les dates futures
            data = pandas.concat( [data, future_df], axis=0 )

            y_prediction = ichimoku_kinko_hyo.predict_future_signals( data , model, days_in_futur, days_in_past )
            data.loc[ future_dates, 'Signal_Predicted' ] = y_prediction

            axe_x_copy = axe_x # make a copy for secondary indicators
            axe_x = numpy.arange( len( data ) ) # pandas.concat( [axe_x, pandas.Series(future_dates)] )

        # Visualiser les résultats
        #
        self.line_price, = ax_main.plot( axe_x, data['Close'], label='Prix de clôture', linewidth=0.8 )
        self.artists_ichimoku['line_tenkan_sen'], = ax_main.plot( axe_x, data['Tenkan_sen'], label='Tenkan-sen', color=pc.TENKAN, linewidth=1.5 )
        self.artists_ichimoku['line_kijun_sen'], = ax_main.plot( axe_x, data['Kijun_sen'], label='Kijun-sen', color=pc.KIJUN, linewidth=1.5 )
        self.artists_ichimoku['line_kijun_sen_upper'], = ax_main.plot( axe_x, data['Kijun_sen_upper'], label='Kijun-sen Upper', linestyle='-', linewidth=0.5, color='blue' )
        self.artists_ichimoku['line_kijun_sen_lower'], = ax_main.plot( axe_x, data['Kijun_sen_lower'], label='Kijun-sen Lower', linestyle='-', linewidth=0.5, color='blue' )
        self.artists_ichimoku['line_kijun_chikou_span'], = ax_main.plot( axe_x, data['Chikou_span'], label='Chikou-span', linestyle='--', linewidth=1, color=pc.royalblue )
        self.artists_ichimoku['fill_Kijun_bands'] = ax_main.fill_between( axe_x, data['Kijun_sen_upper'], data['Kijun_sen_lower'], where=data['Kijun_sen_upper'] >= data['Kijun_sen_lower'], facecolor='lightskyblue', edgecolor='none', alpha=0.3, interpolate=True )

        if self.MAKE_FORCASTING:
            self.artists_ichimoku['line_signal'], = ax_main.plot( axe_x, data['Signal_display'], label='Signal', linewidth=1.5, color='plum' )
            self.artists_ichimoku['line_forcasted_signal'], = ax_main.plot( axe_x, data['Signal_Forcasted'], label='Forcasted Signal', linewidth=1.5, color='violet' )
            self.artists_ichimoku['line_predicted_signal'], = ax_main.plot( axe_x, data['Signal_Predicted'], label='Predicted Signal', linewidth=2, color='mediumorchid' )

        # Kumo cloud
        self.artists_ichimoku['fill_senkou_bands_A'] = ax_main.fill_between( axe_x, data['Senkou_span_A'], data['Senkou_span_B'], where=data['Senkou_span_A'] >= data['Senkou_span_B'], facecolor='lightgreen', edgecolor='none', alpha=0.7, interpolate=True )
        self.artists_ichimoku['fill_senkou_bands_B'] = ax_main.fill_between( axe_x, data['Senkou_span_A'], data['Senkou_span_B'], where=data['Senkou_span_A'] < data['Senkou_span_B'], facecolor='lightcoral', edgecolor='none', alpha=0.7, interpolate=True )

        if self.MAKE_FORCASTING:
            return axe_x_copy

        ax_main.set_xlim( -0.5, len( data ) - 0.5 )

        return axe_x

    # ----------------------------------------------------------------------------
    # Add in title specific data for strategy
    #
    def add_title( self ):
        WIDTH_MA1 = self.mobile_average_1.get()
        WIDTH_MA2 = self.mobile_average_2.get()
        WIDTH_MA3 = self.mobile_average_3.get()

        _s = "{:.6f}".format( self.slope )
        title =  f" - slope: {_s}"
        title += f" - MAx: {WIDTH_MA1} {WIDTH_MA2} {WIDTH_MA3}"
        if self.MAKE_FORCASTING:
            nb_futur = self.nb_days_forcasted_entry.get()
            nb_past = self.nb_days_past_entry.get()
            title += f" - Forcast: {nb_futur} {nb_past}"

        return title