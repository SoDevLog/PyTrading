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
import numpy
import tkinter as tk
import config.func as conf
import matplotlib.dates as mdates
import digitsignalprocessing.func as dsp
import helper as h
import figure.helper as fighelper

from tkinter import ttk
from matplotlib.ticker import ScalarFormatter, StrMethodFormatter
from mplfinance.original_flavor import candlestick_ohlc
from figure.linedeltaselector import LineDeltaSelector

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

	def on_slider_mobile_average_1_change( self, value ):

		# Filtrer les valeurs de value
		if isinstance( value, str ):
			value = int( float(value) )
			self.mobile_average_1.set( value )
        
		if value == -1:
			v = self.mobile_average_1.get()
			value = v - 1
			self.mobile_average_1.set( value )

		if value == +1:
			v = self.mobile_average_1.get()
			value = v + 1
			self.mobile_average_1.set( value )
   
		self.update_graph()

	# ----------------------------------------------------------------------------  

	def on_slider_mobile_average_2_change( self, value ):

		# Filtrer les valeurs de value
		if isinstance( value, str ):
			value = int( float(value) )
			self.mobile_average_2.set( value )
   		
		if value == -1:
			v = self.mobile_average_2.get()
			value = v - 1
			self.mobile_average_2.set( value )

		if value == +1:
			v = self.mobile_average_2.get()
			value = v + 1
			self.mobile_average_2.set( value )
   
		self.update_graph()
  
  	# ----------------------------------------------------------------------------  

	def on_slider_mobile_average_exp_change( self, value ):

		# Filtrer les valeurs de value
		if isinstance( value, str ):
			value = int( float(value) )
			self.mobile_average_exp.set( value )
   		
		if value == -1:
			v = self.mobile_average_exp.get()
			value = v - 1
			self.mobile_average_exp.set( value )

		if value == +1:
			v = self.mobile_average_exp.get()
			value = v + 1
			self.mobile_average_exp.set( value )
   
		self.update_graph()
  
	# ----------------------------------------------------------------------------  
	# Mettre à jour le graphique seulement quand on relâche le slider
	# ou lorsque la valeur change significativement
	#	
	def update_graph( self ):

		# Bloquer l'autoscale pour que les graphes reste en place
		self.ax_main.set_autoscale_on( False )

		# self.ax_main.clear() DO NOT clear() otherwise annotations won't work
		# remove alls lines because strategy by/sell has to be recalculate
		#
		for l in self.lines:
			l.remove()
   
		self.draw_main_graph( self.ax_main, self.intraday, self.width )
		self.ax_main.set_title( self.template.format( self.add_title() ) )
		self.toggle_visibility()
		#self.fig.autofmt_xdate()
		self.canvas.draw_idle() # rafraichit lorsque le proce est dispo
    
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

		ttk.Label( frame_ma12e, text="Moyenne Mobile 1").grid( row=tk_row, column=0, columnspan=2, sticky="e" )
		self.mobile_average_1 = tk.IntVar( value=self.get_stock('MA1'))
		
		tk_row += 1
  
		# Minus button
		_ttk = ttk.Button( frame_ma12e, text="-", width=2, command=lambda: self.on_slider_mobile_average_1_change(-1) )
		_ttk.grid( row=tk_row, column=0, sticky="w" )
        
		# Slider pour la période de la moyenne mobile (de 5 à 200 jours)
		self.slider_ma1 = ttk.Scale(
			frame_ma12e,
			from_=5,
			to=200,
			orient=tk.HORIZONTAL,
			variable=self.mobile_average_1,
			length=130,
			command=self.on_slider_mobile_average_1_change
		).grid( row=tk_row, column=1, sticky="w" )
    
		# Plus button
		_ttk = ttk.Button( frame_ma12e, text="+", width=2, command=lambda: self.on_slider_mobile_average_1_change(+1) )
		_ttk.grid( row=tk_row, column=2, sticky="w" )
  
        # Entry pour afficher la valeur actuelle du slider
		_ttk = ttk.Entry( frame_ma12e, width=5, textvariable=self.mobile_average_1 )
		_ttk.grid( row=tk_row, column=3, sticky="e", **padding_options )
		_ttk.bind( "<Return>", lambda e: self.on_slider_mobile_average_1_change( self.mobile_average_1.get() ) )
		
		tk_row += 1 

		# ------------------------------------------------------------------------

		ttk.Label( frame_ma12e, text="Moyenne Mobile 2").grid( row=tk_row, column=0, columnspan=2, sticky="e" )
		self.mobile_average_2 = tk.IntVar( value=self.get_stock('MA1'))
		
		tk_row += 1
  
		# Minus button
		_ttk = ttk.Button( frame_ma12e, text="-", width=2, command=lambda: self.on_slider_mobile_average_2_change(-1) )
		_ttk.grid( row=tk_row, column=0, sticky="w" )
        
		# Slider pour la période de la moyenne mobile (de 5 à 200 jours)
		self.slider_ma2 = ttk.Scale(
			frame_ma12e,
			from_=5,
			to=200,
			orient=tk.HORIZONTAL,
			variable=self.mobile_average_2,
			length=130,
			command=self.on_slider_mobile_average_2_change
		).grid( row=tk_row, column=1, sticky="w" )

		# Plus button
		_ttk = ttk.Button( frame_ma12e, text="+", width=2, command=lambda: self.on_slider_mobile_average_2_change(+1) )
		_ttk.grid( row=tk_row, column=2, sticky="w" )
  
        # Entry pour afficher la valeur actuelle du slider
		_ttk = ttk.Entry( frame_ma12e, width=5, textvariable=self.mobile_average_2 )
		_ttk.grid( row=tk_row, column=3, sticky="e", **padding_options )
		_ttk.bind( "<Return>", lambda e: self.on_slider_mobile_average_2_change( self.mobile_average_2.get() ) )
		
		tk_row += 1 
  		
		# ------------------------------------------------------------------------

		ttk.Label( frame_ma12e, text="Moyenne Mobile Exp").grid( row=tk_row, column=0, columnspan=2, sticky="e" )
		self.mobile_average_exp = tk.IntVar( value=self.get_stock('MA1'))
		
		tk_row += 1
  
		# Minus button
		_ttk = ttk.Button( frame_ma12e, text="-", width=2, command=lambda: self.on_slider_mobile_average_exp_change(-1) )
		_ttk.grid( row=tk_row, column=0, sticky="w" )
        
		# Slider pour la période de la moyenne mobile (de 5 à 200 jours)
		self.slider_ma2 = ttk.Scale(
			frame_ma12e,
			from_=5,
			to=200,
			orient=tk.HORIZONTAL,
			variable=self.mobile_average_exp,
			length=130,
			command=self.on_slider_mobile_average_exp_change
		).grid( row=tk_row, column=1, sticky="w" )
    
		# Plus button
		_ttk = ttk.Button( frame_ma12e, text="+", width=2, command=lambda: self.on_slider_mobile_average_exp_change(+1) )
		_ttk.grid( row=tk_row, column=2, sticky="w" )
  
        # Entry pour afficher la valeur actuelle du slider
		_ttk = ttk.Entry( frame_ma12e, width=5, textvariable=self.mobile_average_exp )
		_ttk.grid( row=tk_row, column=3, sticky="e", **padding_options )
		_ttk.bind( "<Return>", lambda e: self.on_slider_mobile_average_exp_change( self.mobile_average_exp.get() ) )
		
		tk_row += 1
  
		# ------------------------------------------------------------------------

		separator = ttk.Separator( content, orient='horizontal' )
		separator.grid( row=tk_row, column=0, columnspan=2, sticky="ew", **padding_options)

		tk_row += 1
        
		# ------------------------------------------------------------------------
		
		ttk.Label( content, text="Mobile Average 3   :").grid( row=tk_row, column=0, sticky="e" )
		self.mobile_average_3 = tk.IntVar( value=self.get_stock('MA3') )
		_tki = ttk.Entry( content, width=10, textvariable=self.mobile_average_3 )
		_tki.grid( row=tk_row, column=1, sticky="w", **padding_options )
		
		tk_row += 1

		# ------------------------------------------------------------------------
		
		ttk.Label( content, text="Mobile Average 4   :").grid( row=tk_row, column=0, sticky="e" )
		self.mobile_average_4 = tk.IntVar( value=self.get_stock('MA4') )
		_tki = ttk.Entry( content, width=10, textvariable=self.mobile_average_4 )
		_tki.grid( row=tk_row, column=1, sticky="w", **padding_options )
		
		tk_row += 1

		# ------------------------------------------------------------------------
		
		ttk.Label( content, text="Mobile Average 5   :").grid( row=tk_row, column=0, sticky="e" )
		self.mobile_average_5 = tk.IntVar( value=self.get_stock('MA5') )
		_tki = ttk.Entry( content, width=10, textvariable=self.mobile_average_5 )
		_tki.grid( row=tk_row, column=1, sticky="w", **padding_options )
		
		tk_row += 1

		# ------------------------------------------------------------------------
		
		ttk.Label( content, text="Mobile Average 6   :").grid( row=tk_row, column=0, sticky="e" )
		self.mobile_average_6 = tk.IntVar( value=self.get_stock('MA6') )
		_tki = ttk.Entry( content, width=10, textvariable=self.mobile_average_6 )
		_tki.grid( row=tk_row, column=1, sticky="w", **padding_options )
		
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
				stock['MA4'] = self.mobile_average_4.get()
				stock['MA5'] = self.mobile_average_5.get()
				stock['MA6'] = self.mobile_average_6.get()
				stock['MAE'] = self.mobile_average_exp.get()
				with open( self.path_for_configuration_file, "w" ) as file:
					json.dump( self.configuration, file, indent=4 )
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
		with open( self.path_for_configuration_file, "w" ) as file:
			json.dump( self.configuration, file, indent=4 )

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
		self.on_slider_mobile_average_1_change( 0 )
			
	# ----------------------------------------------------------------------------

	def command_get_item_configration( self ):
		
		for stock in self.configuration.get( 'stocks' ):
			founded = False
			if stock['name'] == self.name:
				self.mobile_average_1.set( stock.get('MA1') )
				self.mobile_average_2.set( stock.get('MA2') )
				self.mobile_average_3.set( stock.get('MA3') )
				self.mobile_average_4.set( stock.get('MA4') )
				self.mobile_average_5.set( stock.get('MA5') )
				self.mobile_average_6.set( stock.get('MA6') )
				self.mobile_average_exp.set( stock.get('MAE') )
				founded = True
				break
		
		if founded == False:
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

		# Graph CONTINUS
		self.var60 = tk.IntVar()
		
		# Strategy
		self.var70 = tk.IntVar()

		# lines = [line_price, line_price2, line_ma1, line_ma2, line_mae, line_ma4, line_ma4, fill1, fill2, line_stem1, line_stem2]
		
		# Init
		self.var20.set(1)
		self.var21.set(0)
		self.var22.set(0) # candles
		self.var31.set(1)
		self.var32.set(1)
		self.var33.set(1)
		self.var34.set(0)
		self.var35.set(0)
		self.var36.set(0)
		self.var37.set(0)
		self.var40.set(1)
		self.var41.set(1)
		self.var60.set( self.intraday )
		self.var70.set(0)
		self.var_all.set(0)
		
		# CONTINUS
		chk60 = ttk.Checkbutton( check_frame, text="CONTINUS", variable=self.var60, command=command_update_graphs)

		# Strategy
		chk70 = ttk.Checkbutton( check_frame, text="Strat1/2", variable=self.var70, command=command_update_graphs)
	
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

		chk60.pack( side=tk.LEFT, padx=5, pady=5 )
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
  
		def set_candles_visibility( v, candles ):
			for candlestick in candles:
				for c in candlestick:
					c.set_visible( v )

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
			set_candles_visibility(True, candles)
		else:
			set_candles_visibility(False, candles)

		if self.var31.get():
			lines[idx].set_visible(True)
		else:
			lines[2].set_visible(False)
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

		data = self.data
		self.ax_main = ax_main
		self.intraday = intraday
		self.width = width
		continus_graph = self.var60.get()
		
		WIDTH_MA1 = self.mobile_average_1.get()
		WIDTH_MA2 = self.mobile_average_2.get()
		WIDTH_MA3 = self.mobile_average_3.get()
		WIDTH_MA4 = self.mobile_average_4.get()
		WIDTH_MA5 = self.mobile_average_5.get()
		WIDTH_MA6 = self.mobile_average_6.get()
		WIDTH_MAE = self.mobile_average_exp.get()
  
		self.update_interface_chk()
		
		y_data = self.data['Close']

		length_data = len( self.data )
		self.message_entry.config( text=f"Data lenght: {length_data}", foreground="green" )
			
		# In continus graph axe_x is a range of date	
		axe_x = data['DateSaved']

		# Convert data['DateSaved'] into new column as a TimeStamp for ohlc graph
		data.loc[:,'Date2num'] = mdates.date2num( data['DateSaved'] )

		# Make data['Date2num'] a continus range
		if continus_graph == True:
			data.loc[:,'Date2num'] = numpy.arange( data['Date2num'].min(), data['Date2num'].min() + length_data )
			axe_x = data['Date2num'] # axe_x is now an index
			
		ohlc = data[['Date2num', 'Open', 'High', 'Low', 'Close']].values.tolist()
		self.candles = candlestick_ohlc( ax_main, ohlc, width=width, colorup='#77d879', colordown='#db3f3f' )

		if length_data < 150:
			line_style = '--'
		else:
			line_style = '-'
		price2 = data['Adj Close'].values
		price = data['Close'].values
		line_price, = ax_main.plot( axe_x, price, line_style, color='black', linewidth=1, label='original data')
		line_price2, = ax_main.plot( axe_x, price2, line_style, color='red', linewidth=1, label='original data')
					
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
		result = dsp.linregress( axe_x.index, data['Adj Close'].values )
		color='r'
		if result.slope >= 0:
			color='g'
		line_tendency = result.slope * axe_x.index + result.intercept # y = a * x + b
		ax_main.plot( axe_x, line_tendency, color=color, linewidth=1, label=f"Slope", linestyle='--' )

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
  		
		# Variables pour l'algo calcul du spread
		buy_sell = False # Spread state alternate buy and sell
		cum_price = 0.0 # Cumulate spreads
		
		#
		# Create buy/sell signals
		#
		_strategy_1_2 = self.var70.get()
		lenght = len( y_data )    
		if _strategy_1_2:
			lenght = lenght - 1  

		for i in range(1, lenght ):
			# Buy
			#
			_c1 = ( ma1[i-1] <= ma2[i-1] and ma1[i] > ma2[i] ) # crossing ma1 upon ma2
			_c2 = ( mae[i] > ma1[i] and mae[i] > ma2[i] ) # mae upper ma1 and ma2
			if _strategy_1_2:
				_c2_2 = ( mae[i+1] > ma1[i+1] and mae[i+1] > ma2[i+1] ) # mae upper ma1 and ma2 continue
				_c1_2 = ( ma1[i+1] > ma2[i+1] ) # crossing continue
				_buy_condition = _c1 and _c1_2 and _c2 and  _c2_2
			else:
				_buy_condition = _c1 and _c2

			if _buy_condition:
				data.at[i, 'Buy_Signal'] = data.at[i, 'Low']
				_str = f"Buy -> {h.format_number3f(y_data[i])}"
				if buy_sell == False:
					buy_sell = True
					cum_price -= y_data[i]
					_str += f" Cum_price: {h.format_number3f(cum_price)}"
				print( _str )
			else:
				data.at[i, 'Buy_Signal'] = -1

			# Sell
			#
			_c1 = (  ma1[i-1] >= ma2[i-1] and ma1[i] < ma2[i] ) # crossing ma1 down ma2
			_c2 = ( (mae[i] < ma1[i] and mae[i] < ma2[i]) ) # mae lower ma1 and ma2
			if _strategy_1_2:
				_c1_2 = ( ma1[i+1] < ma2[i+1] ) # crossing continue
				_c2_2 = ( mae[i+1] < ma1[i+1] and mae[i+1] < ma2[i+1] ) # mae lower ma1 and ma2 continue 
				_sell_condition = _c1 and _c1_2 and _c2 and  _c2_2
			else:
				_sell_condition = _c1 and _c2
				
			if _sell_condition:
				data.at[i, 'Sell_Signal'] = data.at[i, 'High']
				_str = f"Sell -> {h.format_number3f(y_data[i])}"
				if buy_sell == True:
					buy_sell = False
					cum_price += y_data[i]
					_str += f" Cum_price: {h.format_number3f(cum_price)}"
				print( _str )
			else:
				data.at[i, 'Sell_Signal'] = -1
		
		# The spread is not closed still we are in "buy mode"
		# let sell at the last price for demonstration
		#
		if buy_sell == True:
			cum_price += y_data[ len(y_data)-1 ]
			_str = f"Sell -> {h.format_number3f(y_data[i])}"
			_str += f" Cum_price: {h.format_number3f(cum_price)}"
			print( _str )
			
		if 'Buy_Signal' in data.columns:

			# Affiche uniquement les signaux valides
			buy_signals = data['Buy_Signal'][data['Buy_Signal'] > 0]
			sell_signals = data['Sell_Signal'][data['Sell_Signal'] > 0]

			# Extraire les coordonnées X à partir de axe_x
			buy_x = axe_x[ buy_signals.index ]
			sell_x = axe_x[ sell_signals.index ]

			line_stem1 = ax_main.scatter( buy_x, buy_signals.values * 0.99, color='green', marker='^', s=100, label='Buy Signal' )
			line_stem2 = ax_main.scatter( sell_x, sell_signals.values * 1.01, color='red', marker='v', s=100, label='Buy Signal' )

		else:
			self.message_entry.config( text=f"NO Buy/Sell DATA", foreground="red" )
		
		self.cum_price = cum_price
		self.slope = result.slope

		self.lines = [line_price, line_price2, line_ma1, line_ma2, line_mae, line_ma3, line_ma4, line_ma5, line_ma6, fill1, fill2, line_stem1, line_stem2]

  		# Vertical colored strips
		#
		if intraday:
			for i, (day, _data) in enumerate( data['DateSaved'].groupby( data['DateSaved'].dt.date ) ):
				if i % 2 == 1:  # Alternate colors every other day
					a = _data.index[0]
					b = _data.index[-1]
					ax_main.axvspan( axe_x[a], axe_x[b], color='gainsboro', alpha=0.3 )

		#
		# Labels - Distinguish using Period then using Start End
		#
		if intraday:
			_date_mask = '%d-%m %H:%M'
		else:
			_date_mask = '%d-%m-%y'
		ax_main.yaxis.set_major_formatter( StrMethodFormatter('{x:.2f}') )
		ax_main.xaxis.set_major_formatter( mdates.DateFormatter( _date_mask ) ) # %Y on for digits %y year on two digits

		if continus_graph:
			step = max( 1, int( len(data) / 25 ) )
			labels_formatted = data['DateSaved'].dt.strftime( _date_mask )
			ax_main.set_xticks( data['Date2num'][::step] )
			ax_main.set_xticklabels( labels_formatted[::step]) 
			ax_main.set_xlim( data['Date2num'].min(), data['Date2num'].max() ) # after manipulations reset xlim
				
		ax_main.grid(True)

		return axe_x, continus_graph

	# ----------------------------------------------------------------------------
	# Add in title specific data for strategy
	#
	def add_title( self ):
		WIDTH_MA1 = self.mobile_average_1.get()
		WIDTH_MA2 = self.mobile_average_2.get()
		WIDTH_MAE = self.mobile_average_exp.get()

		_s = "{:.6f}".format( self.slope )
		title =  f" - slope: {_s}"       
		title += f" - MAx/MAE: {WIDTH_MA1} {WIDTH_MA2} {WIDTH_MAE}"
		title += f" - spread: {h.format_number3f(self.cum_price)}"
		return title