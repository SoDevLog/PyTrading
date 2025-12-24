"""
    SMC_Tkinter_UI for SMC/ICT strategy methode
    - Init_Check_Box
"""
from matplotlib.ticker import StrMethodFormatter
import numpy
import pandas
import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
import mplfinance as mpf

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import ttk

from smc_engine import SMC_Engine, SMC_Params

# -----------------------------------------------------------------------------

class SMC_Tkinter_UI:
    def __init__( self, df, generate_data=None ):
        self.df_raw = df
        self.generate_data = generate_data
        self.random_seed = 43
        self.df_smc = None

        self.params = SMC_Params()
        self.engine = SMC_Engine( self.params )

        # Stocker les artistes graphiques
        self.artists = {
            'structure': [],
            'swings': [],
            'segments': [],
            'displacement': [],
            'bos': [],
            'choch': [],
            'liquidity': [],
            'order_blocks': [],
            'fvg': [],
            'ote': []
        }
        
        # Figure et axes matplotlib
        self.fig = None
        self.ax = None
        self.canvas = None
        self.toolbar = None

        self.root = tk.Tk()
        self.root.title("TradingInPython - SMC Engine")

        # Frame principal avec deux colonnes
        main_frame = ttk.Frame( self.root )
        main_frame.grid( row=0, column=0, sticky="nsew" )
        
        # Configuration du redimensionnement
        self.root.columnconfigure( 0, weight=1 )
        self.root.rowconfigure( 0, weight=1 )
        main_frame.columnconfigure( 1, weight=1 )
        main_frame.rowconfigure( 0, weight=1 )

        # Frame gauche : contrôles
        left_frame = ttk.Frame( main_frame )
        left_frame.grid( row=0, column=0, padx=10, pady=10, sticky="ns" )

        # Frame : paramètres
        frm_params = ttk.LabelFrame( left_frame, text="Paramètres SMC" )
        frm_params.grid( row=0, column=0, padx=5, pady=5, sticky="ew" )

        # Swing width
        _row = 0
        ttk.Label( frm_params, text="Swing width :" ).grid( row=_row, column=0, sticky="w", padx=5, pady=2 )
        self.var_sw = tk.IntVar( value=self.params.swing_width )
        ttk.Spinbox( frm_params, from_=2, to=10, textvariable=self.var_sw, width=8).grid( row=_row, column=1, padx=5, pady=2 )

        # Liquidity threshold
        _row += 1
        ttk.Label( frm_params, text="Liquidity threshold :" ).grid( row=_row, column=0, sticky="w", padx=5, pady=2 )
        self.var_liquidity = tk.DoubleVar( value=self.params.liquidity_threshold )
        ttk.Spinbox( frm_params, from_=0.001, to=0.05, increment=0.001,
            textvariable=self.var_liquidity, width=8).grid( row=_row, column=1, padx=5, pady=2 )
        
        # ATR period
        _row += 1
        ttk.Label( frm_params, text="ATR period :").grid( row=_row, column=0, sticky="w", padx=5, pady=2 )
        self.var_atr = tk.IntVar( value=self.params.atr_period )
        ttk.Spinbox( frm_params, from_=5, to=50, textvariable=self.var_atr, width=8).grid( row=_row, column=1, padx=5, pady=2 )

        # Displacement threshold
        _row += 1
        ttk.Label( frm_params, text="Displacement ratio :" ).grid( row=_row, column=0, sticky="w", padx=5, pady=2 )
        self.var_disp = tk.DoubleVar( value=self.params.displacement_body_ratio )
        ttk.Spinbox( frm_params, from_=0.1, to=0.95, increment=0.05,
            textvariable=self.var_disp, width=8).grid( row=_row, column=1, padx=5, pady=2 )

        # Frame : overlays
        frm_overlays = ttk.LabelFrame( left_frame, text="Overlays" )
        frm_overlays.grid( row=1, column=0, padx=5, pady=5, sticky="ew" )

        # Init_Check_Box
        self.show_swings = tk.BooleanVar( value=False )
        self.show_structure = tk.BooleanVar( value=True )
        self.show_segments = tk.BooleanVar( value=True )
        self.show_displacement = tk.BooleanVar( value=False )
        self.show_bos = tk.BooleanVar( value=True )
        self.show_choch = tk.BooleanVar( value=True )
        self.show_liquidity = tk.BooleanVar( value=False )
        self.show_order_blocks = tk.BooleanVar( value=False )
        self.show_fvg = tk.BooleanVar( value=False )
        self.show_ote = tk.BooleanVar( value=False )

        chk = [
            ("Swings", self.show_swings, 'swings'),
            ("Structure", self.show_structure, 'structure'),
            ("Segments", self.show_segments, 'segments'),
            ("Displacement", self.show_displacement, 'displacement'),
            ("BOS", self.show_bos, 'bos'),
            ("CHoCH", self.show_choch, 'choch'),
            ("Liquidity", self.show_liquidity, 'liquidity'),
            ("Order Blocks", self.show_order_blocks, 'order_blocks'),
            ("FVG", self.show_fvg, 'fvg'),
            ("OTE", self.show_ote, 'ote'),
        ]

        for i, (txt, var, key) in enumerate(chk):
            cb = ttk.Checkbutton( frm_overlays, text=txt, variable=var,
                command=lambda k=key: self.toggle_overlay(k) )
            cb.grid( row=i, column=0, sticky="w", padx=5, pady=2 )

        # Frame: actions
        frm_actions = ttk.Frame( left_frame )
        frm_actions.grid( row=2, column=0, pady=10 )

        ttk.Button( frm_actions, text="Apply", command=self.run_smc ).grid(row=0, column=0, padx=5, pady=2)
        ttk.Button( frm_actions, text="New", command=self.generate_new_data ).grid(row=1, column=0, padx=5, pady=2)
        ttk.Button( frm_actions, text="Refresh Plot", command=self.refresh_plot ).grid(row=2, column=0, padx=5, pady=2)

        # Frame droit : graphique
        right_frame = ttk.Frame( main_frame )
        right_frame.grid( row=0, column=1, sticky="nsew" )

    def update_params( self ):
        self.params.swing_width = self.var_sw.get()
        self.params.liquidity_threshold = self.var_liquidity.get()
        self.params.atr_period = self.var_atr.get()
        self.params.displacement_body_ratio = self.var_disp.get()
        print( f"swing_width: {self.params.swing_width}" )
        print( f"liquidity_threshold: {self.params.liquidity_threshold}" )
        print( f"atr_period: {self.params.atr_period}" )
        print( f"displacement_body_ratio: {self.params.displacement_body_ratio}" )
        print( f"random_seed: {self.random_seed}" )
        
    def generate_new_data( self ):
        if self.generate_data is not None:
            self.random_seed += 1
            self.df_raw = self.generate_data( seed=self.random_seed )
            self.engine = SMC_Engine( self.params )
            self.df_smc = self.engine.apply( self.df_raw )
            print( f"Nouvelles données générées seed: {self.random_seed}." )
            self.plot()
        
    def run_smc( self ):
        self.update_params()
        self.engine = SMC_Engine( self.params )
        self.df_smc = self.engine.apply( self.df_raw )
        print( "----------------- SMC appliqué." )

    def toggle_overlay( self, overlay_key ):
        """Active ou désactive la visibilité d'un overlay"""
        if self.ax is None:
            return
            
        visible = getattr( self, f'show_{overlay_key}' ).get()
        
        # Modifier la visibilité de tous les artistes de cet overlay
        for artist in self.artists[ overlay_key ]:
            artist.set_visible( visible )
        
        # Redessiner le canvas
        if self.canvas:
            self.canvas.draw_idle()

    def refresh_plot( self ):
        """Recrée complètement le graphique (après Apply ou New)"""
        self.run_smc()
        self.plot()

    def plot( self, name=None ):
        self.name = name
        
        if self.df_smc is None:
            print( "Veuillez d'abord appliquer SMC." )
            return

        df = self.df_smc

        # Nettoyer l'ancien canvas et toolbar s'ils existent
        if self.toolbar:
            self.toolbar.destroy()
            self.toolbar = None
            
        # Nettoyer l'ancien canvas s'il existe
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        
        # Réinitialiser les artistes
        for key in self.artists:
            self.artists[key] = []

        # Créer nouvelle figure
        self.fig, self.ax = plt.subplots( figsize=(12, 7) )
        mpf.plot( df, type='candle', ax=self.ax, style='charles', show_nontrading=False )

        # Créer les overlays et stocker les artistes
        self.create_overlays( self.ax, df )
        
        if self.name is not None:
            self.ax.set_title( f"Smart Money Concept ICT - {name}" )

        self.ax.set_ylabel("")
        # self.ax.yaxis.set_major_formatter( StrMethodFormatter('{x:.2f}') )
        # self.ax.xaxis.set_major_formatter( mdates.DateFormatter('%d %b %Y') )
        self.fig.tight_layout()

        # Intégrer dans Tkinter
        right_frame = self.root.winfo_children()[0].winfo_children()[1]
        self.canvas = FigureCanvasTkAgg( self.fig, master=right_frame )
        self.canvas.draw()
        self.toolbar = NavigationToolbar2Tk( self.canvas, right_frame )
        self.toolbar.update()
        
        # Placer le canvas en dessous de la toolbar
        self.canvas.get_tk_widget().pack( side=tk.TOP, fill=tk.BOTH, expand=True )

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
        
        # Displacement
        self.engine.overlays_displacement(
            df=df,
            ax=ax,
            artists=self.artists,
            key='displacement',
            visible_start=self.show_displacement.get()
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

        # Liquidity
        self.engine.overlays_liquidity(
            df=df,
            ax=ax,
            artists=self.artists, 
            key='liquidity', 
            visible_start=self.show_liquidity.get()
        )    

        # FVG
        self.engine.overlays_fvg(
            df=df,
            ax=ax,
            artists=self.artists, 
            key='fvg', 
            visible_start=self.show_fvg.get()
        )    


        # Breakers
        self.engine.overlays_order_blocs(
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
    
    def run(self):
        self.root.mainloop()