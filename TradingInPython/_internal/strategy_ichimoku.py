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
from figure.candlestick_ohlc import candlestick_ohlc
from digitsignalprocessing import ichimoku_kinko_hyo
from figure.linedeltaselector import LineDeltaSelector

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

        # Read Configuration to configure Tkinter UI
        self.get_configuration()

    def set_data( self, data, name ):
        self.data = data
        self.name = name

    def set_fig( self, fig ):
        self.fig = fig

    def set_canvas( self, canvas ):
        pass

    def set_title( self, template ):
        self.template = template
        return template.format( self.add_title() )

    def create_config_window( self ):

        self.window.title('Ichimoku')
        content = ttk.Frame( self.window, padding=( 15, 10, 15, 10 ) ) # left top right bottom
        content.grid( column=3, row=0 )
        tk_row = 0
        padding_options = {'padx': 5, 'pady': 5}

        ttk.Label( content, text="Mobile Average 1 (Tenkan) :").grid( row=tk_row, column=0, sticky="e" )
        self.mobile_average_1 = tk.IntVar( value=self.get_stock('MA1') )
        _tki = ttk.Entry( content, width=10, textvariable=self.mobile_average_1 )
        _tki.grid( row=tk_row, column=1, sticky="w", **padding_options )

        tk_row += 1

        # ------------------------------------------------------------------------

        ttk.Label( content, text="Mobile Average 2 (Kijun) :").grid( row=tk_row, column=0, sticky="e" )
        self.mobile_average_2 = tk.IntVar( value=self.get_stock('MA2') )
        _tki = ttk.Entry( content, width=10, textvariable=self.mobile_average_2 )
        _tki.grid( row=tk_row, column=1, sticky="w", **padding_options )

        tk_row += 1

        # ------------------------------------------------------------------------

        ttk.Label( content, text="Mobile Average 3 (Senkou) :").grid( row=tk_row, column=0, sticky="e" )
        self.mobile_average_3 = tk.IntVar( value=self.get_stock('MA3') )
        _tki = ttk.Entry( content, width=10, textvariable=self.mobile_average_3 )
        _tki.grid( row=tk_row, column=1, sticky="w", **padding_options )

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

        _tki = ttk.Button(content, text="Default Short", command=lambda: self.command_get_default_configurations(1))
        _tki.grid( row=tk_row, column=0,  columnspan=2, **padding_options  )

        tk_row += 1

        _tki = ttk.Button(content, text="Default", command=self.command_get_default_configurations)
        _tki.grid( row=tk_row, column=0,  columnspan=2, **padding_options  )

        tk_row += 1

        _tki = ttk.Button(content, text="Default Long", command=lambda: self.command_get_default_configurations(2))
        _tki.grid( row=tk_row, column=0,  columnspan=2, **padding_options  )

        tk_row += 1

        _tki = ttk.Button(content, text="Read", command=self.command_get_item_configration)
        _tki.grid( row=tk_row, column=0,  columnspan=2, **padding_options  )

        tk_row += 1

        _tki = ttk.Button(content, text="Save", style="Custom.TButton", command=self.command_save_configuration)
        _tki.grid( row=tk_row, column=0,  columnspan=2, **padding_options  )

        tk_row += 1

    # ----------------------------------------------------------------------------

    def command_get_default_configurations( self,  n=0 ):

        _name = 'DEFAULT' # n==0
        if n == 1:
            _name = 'DEFAULT_SHORT'
        if n == 2:
            _name = 'DEFAULT_LONG'

        for stock in self.configuration.get( 'stocks' ):
            if stock['name'] == _name:
                self.mobile_average_1.set( stock.get('MA1') )
                self.mobile_average_2.set( stock.get('MA2') )
                self.mobile_average_3.set( stock.get('MA3') )
                break

        self.message_entry.config( text=f"Config {_name}", foreground="green" )

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
            self.command_get_default_configurations()

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

    def command_get_default_configuration( self ):

        for stock in self.configuration.get( 'stocks' ):
            if stock['name'] == 'DEFAULT':
                self.mobile_average_1.set( stock.get('MA1') )
                self.mobile_average_2.set( stock.get('MA2') )
                self.mobile_average_3.set( stock.get('MA3') )
                break

        self.message_entry.config( text=f"Default config", foreground="green" )

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
            self.command_get_default_configuration()

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
        self.var20 = tk.IntVar()
        self.var_candles = tk.IntVar()
        self.var31 = tk.IntVar()
        self.var32 = tk.IntVar()
        self.var33 = tk.IntVar()
        self.var34 = tk.IntVar()
        self.var35 = tk.IntVar()
        self.var36 = tk.IntVar()
        self.var40 = tk.IntVar()
        self.var50 = tk.IntVar()

        self.var70 = tk.BooleanVar( value=False ) # MAKE_FORCASTING
        self.ichimoku_futuriste = tk.BooleanVar( value=False )

        # lines = [line_price, line_tenkan_sen, line_kijun_sen, line_kijun_sen_upper, line_kijun_sen_lower, fill_Kijun_bands, line_kijun_chikou_span, fill1, fill2, line_s1, line_s2, line_s3 ]

        # Init
        self.var20.set(0)
        self.var_candles.set(1)
        self.var31.set(1)
        self.var32.set(1)
        self.var33.set(0)
        self.var34.set(0)
        self.var35.set(1)
        self.var36.set(1)
        self.var40.set(1)

        # lines = [line_price, line_tenkan_sen, line_kijun_sen, line_kijun_sen_upper, line_kijun_sen_lower, fill_Kijun_bands, line_kijun_chikou_span, fill1, fill2, line_s1, line_s2, line_s3 ]

        chk20 = ttk.Checkbutton( check_frame, text="Price", variable=self.var20, command=command_update_lines)
        chk_candles = ttk.Checkbutton( check_frame, text="Candle", variable=self.var_candles, command=command_update_lines)

        chk31 = ttk.Checkbutton( check_frame, text="Tenkan", variable=self.var31, command=command_update_lines)
        Tooltip( chk31, "Momentum court terme")
        
        chk32 = ttk.Checkbutton( check_frame, text="Kijun", variable=self.var32, command=command_update_lines)
        Tooltip( chk32, "Equilibre long terme")
        
        chk33 = ttk.Checkbutton( check_frame, text="Kj Up", variable=self.var33, command=command_update_lines)
        chk34 = ttk.Checkbutton( check_frame, text="Kj Down", variable=self.var34, command=command_update_lines)
        chk35 = ttk.Checkbutton( check_frame, text="Kj Bands", variable=self.var35, command=command_update_lines)
        Tooltip( chk35, "Canal de volatilité Kijun ATR")
        
        chk36 = ttk.Checkbutton( check_frame, text="Chikou", variable=self.var36, command=command_update_lines)
        Tooltip( chk36, "Signal retardé, confirmation de la tendance")
        
        # Double visibly lines
        chk40 = ttk.Checkbutton( check_frame, text="Kumo", variable=self.var40, command=command_update_lines)
        Tooltip( chk40, "Nuage des équilibres du marché.")

        # CONTINUS, MAKE_FORCASTING
        #chk60 = ttk.Checkbutton( check_frame, text="CONTINUS", variable=self.var60, command=command_update_graphs)
        chk70 = ttk.Checkbutton( check_frame, text="Forcasting", variable=self.var70, command=command_update_graphs)
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
        lines = self.lines
        candles = self.candles

        def set_candles_visibility( v, candles ):
            for candlestick in candles:
                for c in candlestick:
                    c.set_visible( v )

        # lines = [line_price, line_tenkan_sen, line_kijun_sen, line_kijun_sen_upper, line_kijun_sen_lower, fill_Kijun_bands, line_kijun_chikou_span, fill1, fill2, line_s1, line_s2, line_s3 ]

        # Simple line's checkbox
        # to find index autommaticaly
        #
        _line_to_display = [
            self.var20, # line_price
            self.var31, # line_tenkan_sen
            self.var32, # line_kijun_sen
            self.var33, # line_kijun_sen_upper
            self.var34, # line_kijun_sen_lower
            self.var35,	# fill_Kijun_bands
            self.var36  # line_kijun_chikou_span
        ]

        for var in _line_to_display:
            _idx = _line_to_display.index( var ) # don't have to count index
            if var.get():
                lines[_idx].set_visible(True)
            else:
                lines[_idx].set_visible(False)

        # Special way to set visible for candles
        if self.var_candles.get():
            set_candles_visibility(True, candles)
        else:
            set_candles_visibility(False, candles)

        # Kumo cloud
        _idx = 7 # fill1, fill2 must count index in table 'lines'
        if self.var40.get():
            lines[_idx].set_visible(True)
            lines[_idx+1].set_visible(True)
        else:
            lines[_idx].set_visible(False)
            lines[_idx+1].set_visible(False)

        if self.MAKE_FORCASTING:
            _idx = 9 # line_s1, line_s2, line_s3 must count index in table 'lines'
            if self.var70.get():
                lines[_idx].set_visible(True)
                lines[_idx+1].set_visible(True)
                lines[_idx+2].set_visible(True)
            else:
                lines[_idx].set_visible(False)
                lines[_idx+1].set_visible(False)
                lines[_idx+2].set_visible(False)

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
        _forcasting = self.var70.get()
        if intraday and _forcasting:
            _forcasting = False # disable for intraday
            self.message_entry.config( text="Prévisions désactivées en intraday.", foreground="red" )
            self.var70.set( False )

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
        line_price, = ax_main.plot( axe_x, data['Close'], label='Prix de clôture', linewidth=0.8 )
        line_tenkan_sen, = ax_main.plot( axe_x, data['Tenkan_sen'], label='Tenkan-sen', color=pc.TENKAN, linewidth=1.5 )
        line_kijun_sen, = ax_main.plot( axe_x, data['Kijun_sen'], label='Kijun-sen', color=pc.KIJUN, linewidth=1.5 )
        line_kijun_sen_upper, = ax_main.plot( axe_x, data['Kijun_sen_upper'], label='Kijun-sen Upper', linestyle='-', linewidth=0.5, color='blue' )
        line_kijun_sen_lower, = ax_main.plot( axe_x, data['Kijun_sen_lower'], label='Kijun-sen Lower', linestyle='-', linewidth=0.5, color='blue' )
        fill_Kijun_bands = ax_main.fill_between( axe_x, data['Kijun_sen_upper'], data['Kijun_sen_lower'], where=data['Kijun_sen_upper'] >= data['Kijun_sen_lower'], facecolor='lightskyblue', edgecolor='none', alpha=0.3, interpolate=True )
        line_kijun_chikou_span, = ax_main.plot( axe_x, data['Chikou_span'], label='Chikou-span', linestyle='--', linewidth=1 )

        if self.MAKE_FORCASTING:
            line_s1, = ax_main.plot( axe_x, data['Signal_display'], label='Signal', linewidth=1.5, color='plum' )
            line_s2, = ax_main.plot( axe_x, data['Signal_Forcasted'], label='Forcasted Signal', linewidth=1.5, color='violet' )
            line_s3, = ax_main.plot( axe_x, data['Signal_Predicted'], label='Predicted Signal', linewidth=2, color='mediumorchid' )

        # Kumo cloud
        fill1 = ax_main.fill_between( axe_x, data['Senkou_span_A'], data['Senkou_span_B'], where=data['Senkou_span_A'] >= data['Senkou_span_B'], facecolor='lightgreen', edgecolor='none', alpha=0.7, interpolate=True )
        fill2 = ax_main.fill_between( axe_x, data['Senkou_span_A'], data['Senkou_span_B'], where=data['Senkou_span_A'] < data['Senkou_span_B'], facecolor='lightcoral', edgecolor='none', alpha=0.7, interpolate=True )

        # Table of line's objects to set_visible or not
        if self.MAKE_FORCASTING:
            self.lines = [line_price, line_tenkan_sen, line_kijun_sen, line_kijun_sen_upper, line_kijun_sen_lower, fill_Kijun_bands, line_kijun_chikou_span, fill1, fill2, line_s1, line_s2, line_s3 ]
        else:
            self.lines = [line_price, line_tenkan_sen, line_kijun_sen, line_kijun_sen_upper, line_kijun_sen_lower, fill_Kijun_bands, line_kijun_chikou_span, fill1, fill2 ]

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