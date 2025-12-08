"""
    SMC_Tkinter_UI
"""
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import mplfinance as mpf

from smc_engine import SMC_Engine, SMC_Params

class SMC_Tkinter_UI:
    def __init__( self, df, generate_data=None ):
        self.df_raw = df
        self.generate_data = generate_data
        self.random_seed = 42
        self.df_smc = None

        self.params = SMC_Params()
        self.engine = SMC_Engine(self.params)

        self.root = tk.Tk()
        self.root.title("TradingInPython - SMC Engine PRO")

        # Frame : parameters
        frm_params = ttk.LabelFrame(self.root, text="Paramètres SMC")
        frm_params.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        # Swing width
        ttk.Label(frm_params, text="Swing width :").grid(row=0, column=0, sticky="w")
        self.var_sw = tk.IntVar(value=self.params.swing_width)
        ttk.Spinbox(frm_params, from_=2, to=10, textvariable=self.var_sw, width=5).grid(row=0, column=1)

        # ATR period
        ttk.Label(frm_params, text="ATR period :").grid(row=1, column=0, sticky="w")
        self.var_atr = tk.IntVar(value=self.params.atr_period)
        ttk.Spinbox(frm_params, from_=5, to=50, textvariable=self.var_atr, width=5).grid(row=1, column=1)

        # displacement threshold
        ttk.Label(frm_params, text="Displacement ratio :").grid(row=2, column=0, sticky="w")
        self.var_disp = tk.DoubleVar(value=self.params.displacement_body_ratio)
        ttk.Spinbox( frm_params, from_=0.1, to=0.95, increment=0.05,
                    textvariable=self.var_disp, width=5).grid(row=2, column=1)

        # Frame : overlays
        frm_overlays = ttk.LabelFrame(self.root, text="Overlays")
        frm_overlays.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        self.show_structure = tk.BooleanVar(value=True)
        self.show_bos = tk.BooleanVar(value=True)
        self.show_choch = tk.BooleanVar(value=True)
        self.show_liquidity = tk.BooleanVar(value=True)
        self.show_ob = tk.BooleanVar(value=True)
        self.show_fvg = tk.BooleanVar(value=True)
        self.show_ote = tk.BooleanVar(value=True)
        self.show_labels = tk.BooleanVar(value=True)

        chk = [
            ("Structure", self.show_structure),
            ("BOS", self.show_bos),
            ("CHoCH", self.show_choch),
            ("Liquidity", self.show_liquidity),
            ("Order Blocks", self.show_ob),
            ("FVG", self.show_fvg),
            ("OTE", self.show_ote),
            ("Labels", self.show_labels),
        ]

        for i, (txt, var) in enumerate(chk):
            ttk.Checkbutton( frm_overlays, text=txt, variable=var).grid(row=i, column=0, sticky="w" )

        # Frame: actions
        frm_actions = ttk.Frame( self.root )
        frm_actions.grid( row=2, column=0, pady=10 )

        ttk.Button( frm_actions, text="Exécuter SMC", command=self.run_smc ).grid( row=0, column=0, padx=5 )
        ttk.Button( frm_actions, text="New", command=self.generate_new_data ).grid( row=0, column=1, padx=5 )
        ttk.Button( frm_actions, text="Plot", command=self.plot ).grid( row=0, column=2, padx=5 )

    def update_params(self):
        self.params.swing_width = self.var_sw.get()
        self.params.atr_period = self.var_atr.get()
        self.params.displacement_body_ratio = self.var_disp.get()
        print( f"swing_width: {self.params.swing_width}" )
        print( f"atr_period: {self.params.atr_period}" )
        print( f"body_ratio: {self.params.displacement_body_ratio}" )
        print( f"radom_seed:  {self.random_seed}")
        
    def generate_new_data(self):
        if self.generate_new_data is not None:
            self.random_seed += 1
            self.df_raw = self.generate_data( seed=self.random_seed )
            self.engine = SMC_Engine( self.params )
            self.df_smc = self.engine.apply( self.df_raw )
            print( f"Nouvelles données générés seed: {self.random_seed}." )
            self.plot()
        
    def run_smc(self):
        self.update_params()
        self.engine = SMC_Engine( self.params )
        self.df_smc = self.engine.apply( self.df_raw )
        print( "SMC appliqué." )

    def plot(self):
        if self.df_smc is None:
            print("Veuillez d'abord appliquer SMC.")
            return

        df = self.df_smc

        fig, ax = plt.subplots(figsize=(15, 7))
        mpf.plot( df, type='candle', ax=ax, style='charles', show_nontrading=False )

        self.engine.plot_overlays(
            ax, df,
            show_structure=self.show_structure.get(),
            show_bos=self.show_bos.get(),
            show_choch=self.show_choch.get(),
            show_liquidity=self.show_liquidity.get(),
            show_ob=self.show_ob.get(),
            show_fvg=self.show_fvg.get(),
            show_ote=self.show_ote.get(),
            show_labels=self.show_labels.get(),
        )

        plt.tight_layout()
        plt.show()
        
        self.root.lift()

    def run(self):
        self.root.mainloop()
