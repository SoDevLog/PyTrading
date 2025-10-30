""" 
    Module de gestion de portefeuille d'investissement avec prix en temps réel
    - PortfolioManager
        - update_current_prices
    - PortfolioApp
        - setup_ui
        - create_header
        - create_transactions_tab
            - transactions_tree
        - create_positions_tab
            - positions_tree
        - create_report_tab
"""
import os
import sys
import json
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import yfinance
from datetime import datetime
from tkcalendar import DateEntry
from typing import Dict
from collections import defaultdict
from threading import Thread

from pathlib import Path
base = Path(__file__).resolve().parent.parent
sys.path.append(str(base))

from config.path import BASE_PORTFOLIO_DIR
from tkinterh.helper import Tooltip
import config.func as conf

PORTFOLIO_FILE_PATH = BASE_PORTFOLIO_DIR / "portfolio.json"

class PortfolioManager:

    def __init__( self, filename ):
        self.filename = filename
        self.transactions = []
        self.current_prices = {}
        self.ticker_names = {}  # Cache pour les noms des tickers
        self.load_data()
    
    def load_data( self ):
        """Charge les données depuis le fichier JSON"""
        if os.path.exists(self.filename):
            with open(self.filename, 'r', encoding='utf-8') as f:
                self.transactions = json.load(f)
        else:
            self.transactions = []
            self.current_prices = {}
            
    def save_data( self ):
        """Sauvegarde les données dans le fichier JSON"""
        with open(self.filename, 'w', encoding='utf-8') as f:
            json.dump(self.transactions, f, indent=2, ensure_ascii=False)
    
    def get_ticker_name( self, ticker: str ) -> str:
        """Récupère le nom du ticker depuis yfinance (avec cache)"""
        if ticker in self.ticker_names:
            return self.ticker_names[ticker]
        
        try:
            _ticker = yfinance.Ticker( ticker )
            stock_info = _ticker.get_info()
            _short_name = stock_info.get( 'shortName', 'N/A' )
            self.ticker_names[ticker] = _short_name
            return _short_name
        except Exception as e:
            print(f"Erreur lors de la récupération du nom pour {ticker} : {e}")
            self.ticker_names[ticker] = 'N/A'
            return 'N/A'
    
    def add_transaction( self, type_transaction: str, ticker: str, quantity: float, 
                       price: float, date: str = None, fees: float = 0.0 ):
        """Ajoute une transaction (achat ou vente)"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        if type_transaction.lower() == 'achat':
            total = quantity * price + fees
        else:
            total = quantity * price - fees
        
        # Récupération du nom du ticker
        ticker_upper = ticker.upper()
        name = self.get_ticker_name( ticker_upper )
        
        transaction = {
            'id': len(self.transactions) + 1,
            'type': type_transaction.lower(),
            'ticker': ticker_upper,
            'name': name,
            'quantity': quantity,
            'price': price,
            'date': date,
            'fees': fees,
            'total': total
        }

        self.transactions.append(transaction)
        self.save_data()
        return transaction

    def delete_transaction( self, transaction_id: int ):
        """Supprime une transaction par son ID"""
        self.transactions = [t for t in self.transactions if t['id'] != transaction_id]
        self.save_data()

    # -------------------------------------------------------------------------
    
    def compute_positions( self ) -> Dict[str, Dict[str, float]]:
        """Calcule les positions actuelles du portefeuille"""
        positions = defaultdict(lambda: {
            "quantity": 0.0,
            "total_cost": 0.0,
            "total_sales": 0.0,
            "realized_pnl": 0.0,
            "name": "N/A"
        })

        for t in self.transactions:
            ticker = t["ticker"]
            quantity = float(t["quantity"])
            price = float(t["price"])
            fees = float(t.get("fees", 0.0))
            side = t["type"].lower()
            name = t.get("name", "N/A")

            # Stocker le nom si disponible
            if name != "N/A":
                positions[ticker]["name"] = name

            # Achat : on augmente la position
            if side == "achat":
                positions[ticker]["quantity"] += quantity
                positions[ticker]["total_cost"] += quantity * price + fees

            # Vente : on réduit la position et on réalise le PnL
            elif side == "vente":
                current_qty = positions[ticker]["quantity"]
                if current_qty <= 0:
                    # Vente sans position existante – on ignore ou on log
                    continue

                avg_cost = positions[ticker]["total_cost"] / current_qty
                realized = (price - avg_cost) * quantity - fees
                positions[ticker]["realized_pnl"] += realized

                # Mise à jour de la position restante
                positions[ticker]["quantity"] -= quantity
                positions[ticker]["total_sales"] += quantity * price
                positions[ticker]["total_cost"] -= avg_cost * quantity  # réduction proportionnelle

                # Nettoyage : si quantité ≈ 0, on réinitialise la position
                if abs(positions[ticker]["quantity"]) < 1e-8:
                    positions[ticker]["quantity"] = 0.0
                    positions[ticker]["total_cost"] = 0.0

        # Calcul du coût moyen pour affichage
        for ticker, pos in positions.items():
            if pos["quantity"] > 0:
                pos["avg_cost"] = pos["total_cost"] / pos["quantity"]
            else:
                pos["avg_cost"] = 0.0

            # Si le nom n'est toujours pas disponible, on le récupère
            # Compatibilité ascendante avec les transactions sans le champ 'name'
            #
            if pos["name"] == "N/A":
                pos["name"] = self.get_ticker_name( ticker )            
        return dict( positions )

    # -------------------------------------------------------------------------

    def compute_statistics( self ) -> Dict:
        """Calcule les statistiques globales du portefeuille"""
        total_invested = sum( t['total'] for t in self.transactions if t['type'] == 'achat' )
        total_sold = sum( t['total'] for t in self.transactions if t['type'] == 'vente' )
        total_fees = sum( t['fees'] for t in self.transactions )
        
        positions = self.compute_positions()
        realized_pnl = sum( p['realized_pnl'] for p in positions.values() )
        
        # Calcul de la valeur actuelle et du P&L non réalisé
        current_value = 0.0
        unrealized_pnl = 0.0
        
        for ticker, pos in positions.items():
            current_price = self.current_prices.get( ticker, 0.0 )
            quantity = pos['quantity']
            if current_price > 0 and quantity > 0:
                current_value += quantity * current_price
                avg_cost = pos['total_cost'] / quantity
                unrealized_pnl += (current_price - avg_cost) * quantity

        total_pnl = realized_pnl + unrealized_pnl
        net_invested = total_invested - total_sold
        
        return {
            'total_invested': total_invested,
            'total_sold': total_sold,
            'net_invested': net_invested,
            'total_fees': total_fees,
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_pnl': total_pnl,
            'current_value': current_value,
            'return_percentage': (total_pnl / total_invested * 100) if total_invested > 0 else 0
        }

    # -------------------------------------------------------------------------
    
    def update_current_prices( self, callback ):
        """Met à jour les prix courants"""
        positions = self.compute_positions()
        tickers = [ticker for ticker, pos in positions.items()]

        if not tickers:
            callback()
            return

        def fetch_prices():
            for ticker in tickers:
                try:
                    
                    stock = yfinance.Ticker( ticker )
                    if not stock.fast_info:
                        print(f"⚠️ Le symbole {ticker} n'existe pas ou ne renvoie rien.")
                        self.current_prices[ticker] = 0.0
                        continue

                    hist = stock.history( period="1d" )
                    if not hist.empty:
                        self.current_prices[ticker] = hist['Close'].iloc[-1]
                    else:
                        hist = stock.history(period="5d")
                        if not hist.empty:
                            self.current_prices[ticker] = hist['Close'].iloc[-1]
                        else:
                            self.current_prices[ticker] = 0.0

                except Exception as e:
                    print(f"Erreur lors de la récupération du ticker {ticker} : {e}")
                    self.current_prices[ticker] = 0.0
                    
            if callback:
                callback()

        thread = Thread( target=fetch_prices )
        thread.daemon = True
        thread.start()

# -----------------------------------------------------------------------------

class PortfolioApp:
    """Interface graphique TTK moderne pour le gestionnaire de portefeuille"""
    
    def __init__( self, parent ):
        self.parent = parent
        self.window = tk.Toplevel( parent ) if parent else tk.Tk()
        self.window.title("💰 Gestionnaire de Portefeuille")
        self.window.geometry('+280+65')
        self.window.geometry("1000x850")
        self.initialise = False

        # Bring to front
        self.window.lift()
        self.window.focus_force()
        self.window.attributes('-topmost', True)
        self.window.after(100, lambda: self.window.attributes('-topmost', False))
        
        # Get fetch data configuration and set specific values
        _r = conf.get_fetch_data_configuration()
        if _r != 0:
            self.conf_fetch_data = _r
            if self.conf_fetch_data['PORTFOLIO_FILE_PATH'] == "":
                self.conf_fetch_data['PORTFOLIO_FILE_PATH'] = PORTFOLIO_FILE_PATH
        
        # Portfolio Manager
        self.portfolio = PortfolioManager( Path( self.conf_fetch_data['PORTFOLIO_FILE_PATH'] ) )
        
        # Configuration du style TTK
        self.setup_styles()
        
        self.setup_ui()
        self.update_prices()
    
    # -------------------------------------------------------------------------
    
    def setup_styles( self ):
        
        style = ttk.Style()
        
        # Windows themes:
        #('winnative', 'clam', 'alt', 'default', 'classic', 'vista', 'xpnative')
        # MacOS themes:
        #('aqua', 'clam', 'alt', 'default', 'classic')
        # PythonLinux themes:
        #('clam', 'alt', 'default', 'classic')
        
        style.theme_use( 'vista' )
        
        # Configuration des couleurs
        self.colors = {
            'bg': '#f5f5f5',
            'fg': '#2c3e50',
            'header': "#f1f7fd",
            'primary': '#3498db',
            'success': '#27ae60',
            'danger': '#e74c3c',
            'warning': '#f39c12',
            'light': '#ecf0f1',
            'dark': '#34495e'
        }
        
        # 'flat'	▭ Plat	Aucun relief, sans bordure apparente
        # 'raised'	⧉ Relief sortant	Fait ressortir le widget, comme s'il était au-dessus du plan
        # 'sunken'	⧋ Relief enfoncé	Fait paraître le widget enfoncé dans le plan
        # 'ridge'	⧠ Bord en relief double	Une bordure double légèrement surélevée
        # 'groove'	⧠ Bord en creux double	Une bordure double légèrement enfoncée
        # 'solid'	⧉ Bordure pleine	Une bordure simple, sans effet 3D

        # Style pour les frames
        style.configure( 'Card.TFrame', relief='flat' )
        style.configure( 'CardHeader.TFrame', background=self.colors['header'], relief='solid' )
        style.configure( 'CardTrans.TFrame' ) # Tab Transaction Frame
        style.configure( 'Main.TFrame', background=self.colors['bg'] )
        
        # Style pour les labels
        style.configure(
            'Title.TLabel', 
            font=('Segoe UI', 16, 'bold'),
            foreground=self.colors['primary']
        )
                
        style.configure(
            'Subtitle.TLabel',
            font=('Segoe UI', 11, 'bold'),
            foreground=self.colors['dark']#,
            #background='white'
        )
        
        style.configure(
            'StatName.TLabel',
            font=('Segoe UI', 10),
            foreground='#7f8c8d',
            background=self.colors['header']
        )
        
        style.configure(
            'StatValue.TLabel',
            font=('Segoe UI', 14, 'bold'),
            foreground=self.colors['dark'],
            background=self.colors['header']
        )
        
        style.configure(
            'Info.TLabel',
            font=('Segoe UI', 8),
            foreground='#95a5a6',
        )
        
        # Style pour les boutons
        style.configure(
            'Primary.TButton',
            font=('Segoe UI', 10),
            padding=(20, 8)
        )
        
        style.configure('Danger.TButton',
                       font=('Segoe UI', 10),
                       padding=(20, 8))
        
        # Style pour les entrées
        style.configure('TEntry',
                       fieldbackground='white',
                       borderwidth=1,
                       relief='solid')
        
        # Style pour les radiobuttons
        style.configure(
            'TRadiobutton',
            font=('Segoe UI', 10)
        )
        
        # Style pour le Treeview
        style.configure(
            'Treeview',
            background='white',
            fieldbackground='white',
            foreground=self.colors['dark'],
            font=('Segoe UI', 9)
        )
        
        style.configure(
            'Treeview.Heading',
            font=('Segoe UI', 10, 'bold'),
            foreground=self.colors['dark']
        )
        
        style.map( 'Treeview', background=[('selected', self.colors['primary'])] )
        
        # Style pour le Notebook
        style.configure('TNotebook', background=self.colors['bg'])
        style.configure(
            'TNotebook.Tab', 
            font=('Segoe UI', 10),
            padding=(20, 8)
        )

        # Radiobutton Buy/Sell
        style.configure(
            'Buy.TRadiobutton',
            background='#d4edda',
            padding=(5, 5)
        )        

        style.configure(
            'Sell.TRadiobutton',
            background='#f8d7da',
            padding=(5, 5)
        )
                
    
    # -------------------------------------------------------------------------
    
    def setup_ui( self ):

        main_frame = ttk.Frame( self.window, style='Main.TFrame' )
        main_frame.pack( fill=tk.BOTH, expand=True, padx=15, pady=15 )

        # Control frame with buttons for managing portfolios
        self.create_control( main_frame )
        
        # Header with global statistics on portfolio
        self.create_header( main_frame )

        notebook = ttk.Notebook( main_frame )
        notebook.pack( fill=tk.BOTH, expand=True, pady=(0, 0) )

        # Onglet Transactions
        transactions_frame = ttk.Frame( notebook, style='Main.TFrame' )
        notebook.add( transactions_frame, text="📋 Transactions" )
        self.create_transactions_tab( transactions_frame )

        # Onglet Positions
        positions_frame = ttk.Frame( notebook, style='Main.TFrame' )
        notebook.add( positions_frame, text="📊 Positions" )
        self.create_positions_tab( positions_frame )

        # Onglet Rapport
        report_frame = ttk.Frame( notebook, style='Main.TFrame' )
        notebook.add( report_frame, text="📈 Rapport" )
        self.create_report_tab( report_frame )

    # -------------------------------------------------------------------------
    
    def reload_portfolio_json( self ):
    
        # Ouvre une boîte de dialogue pour choisir un fichier .json
        _file_name = Path( self.portfolio.filename )
        filepath = filedialog.askopenfilename(
            title="Sélectionnez un fichier portefeuille",
            filetypes=[("Fichiers JSON", "*.json")],
            initialdir=_file_name.parent,
            initialfile=os.path.basename( _file_name )
        )
        
        if filepath:
            # Set portfolio path file name
            self.portfolio.filename = Path( filepath )
            
            # Reload data into portfolio
            self.portfolio.load_data()
            
            # Sauver le nouveau choix utilisateur
            self.conf_fetch_data['PORTFOLIO_FILE_PATH'] = str( self.portfolio.filename )
            conf.save_fecth_data_json( self.conf_fetch_data )
            
            # Update title_label with new path.stem
            self.update_title_label()
            
            # Update all 
            self.update_prices()

    # -------------------------------------------------------------------------

    def create_portfolio_json( self ):

        file_path = filedialog.asksaveasfilename(
            title="Créer un nouveau portefeuille",
            defaultextension=".json",
            filetypes=[("Fichiers JSON", "*.json"), ("Tous les fichiers", "*.*")],
            initialfile=os.path.basename( self.portfolio.filename )
        )

        if file_path:
            if not os.path.exists( file_path ):

                try:

                    self.portfolio.filename = file_path
                    
                    # Create empty JSON file
                    with open( self.portfolio.filename, 'w', encoding='utf-8' ) as f:
                        json.dump( [], f, indent=2, ensure_ascii=False )

                    # Reload empty data
                    self.portfolio.load_data()
                    
                    # Update title_label with new path.stem
                    self.update_title_label()
                                
                    # Refresh All
                    self.update_prices()

                except Exception as e:
                    messagebox.showerror(
                        "Erreur",
                        f"Une erreur est survenue lors de la création du fichier :\n{e}"
                    )
                    return

                messagebox.showinfo(
                    "Succès", 
                    f"Fichier JSON créé avec succès!\n"
                    f"Chemin: {file_path}\n"
                )
            else:
                messagebox.showwarning(
                    "Avertissement",
                    f"Le fichier existe déjà !\n"
                    f"Chemin: {file_path}\n"
                    f"Voulez-vous le remplacer ?",
                    icon='warning'
                )

    # -------------------------------------------------------------------------

    def suppress_portfolio_json( self ):
    
        # Ouvre une boîte de dialogue pour choisir un fichier .json
        filepath = filedialog.askopenfilename(
            title="Sélectionnez un fichier portefeuille à supprimer",
            filetypes=[("Fichiers JSON", "*.json")],
            initialfile=os.path.basename( self.portfolio.filename )
        )

        # Si un fichier est sélectionné
        if filepath:
            # Peut-on supprimer ce fichier ?
            if not os.access( filepath, os.W_OK ):
                messagebox.showerror(
                    "Erreur : accès refusé",
                    "Le fichier est en lecture seule.\n"
                    "Veuillez modifier les permissions du fichier avant de le supprimer."
                )
                return
            
            # Confirmation avant suppression
            confirm = messagebox.askyesno( "Confirmation", f"Voulez-vous vraiment supprimer le fichier ?\n\n{filepath}" )
            if confirm:
                try:
                    os.remove( filepath )
                    messagebox.showinfo( "Succès", f"Le fichier : {Path(filepath).absolute().name} a été supprimé avec succès." )
                except Exception as e:
                    messagebox.showerror( "Erreur", f"Une erreur est survenue lors de la suppression :\n{e}" )
            else:
                messagebox.showinfo( "Annulé", "Suppression annulée." )                    

    # -------------------------------------------------------------------------
    
    def create_control( self, parent ):
        """Crée le frame de contrôles pour la gestion des portfolios """
        controls_frame = ttk.LabelFrame( parent, text="Contrôles", padding="5" )
        controls_frame.pack( fill=tk.X, padx=0, pady=0 )

        # Label Titre
        _path = Path( self.portfolio.filename )
        self.title_label = ttk.Label(
            controls_frame,
            text=f"💰 Portefeuille : {_path.stem}",
            style='Title.TLabel'
        )
        self.title_label.pack( side=tk.LEFT, padx=80, pady=(0, 5) )
        
        # Boutons de gestion
        _ttk = ttk.Button( controls_frame, text="Supprimer", command=self.suppress_portfolio_json )
        _ttk.pack( side=tk.RIGHT, padx=( 5, 30 ), pady=( 0, 5 ) )
        Tooltip( _ttk, "Supprimer un fichier de portefeuille" )

        _ttk = ttk.Button( controls_frame, text="Créer", command=self.create_portfolio_json )
        _ttk.pack( side=tk.RIGHT, padx=5, pady=( 0, 5 ) )
        Tooltip( _ttk, "Créer un nouveau portefeuille" )

        _ttk = ttk.Button( controls_frame, text="Ouvrir", command=self.reload_portfolio_json )
        _ttk.pack( side=tk.RIGHT, padx=5, pady=( 0, 5 ) )
        Tooltip( _ttk, "Choisir un fichier de portefeuille" )
            
    # -------------------------------------------------------------------------
    
    def create_header( self, parent ):
        """Crée l'en-tête avec les statistiques globales du portefeuille"""
        header_frame = ttk.Frame( parent, style='Card.TFrame', padding=20 )
        header_frame.pack( fill=tk.X, pady=(0, 5) )
        
        # Frame pour les statistiques
        stats_frame = ttk.Frame( header_frame, style='Card.TFrame' )
        stats_frame.pack( fill=tk.X )

        self.stats_labels = {}
        stats_names = [
            ("Solde Net Investi", "net_invested", "Total des achats - Total des ventes"),
            ("Valeur Actuelle", "current_value", "Quantité des positions en cours x Prix courant"),
            ("P&L Réalisé", "realized_pnl", "P&L (Profit and Loss) Profit ou Perte réalisé sur les positions clôturées"),
            ("P&L Non Réalisé", "unrealized_pnl", "Profit ou Perte non encore réalisé sur les positions en cours"),
            ("P&L Total", "total_pnl", "Profit ou Perte total"),
            ("Rendement", "return_percentage", "Rendement global du portefeuille"),
        ]

        for idx, (name, key, tooltip) in enumerate( stats_names ):
            row = 0 # idx // 3
            col = idx # idx % 3

            stat_container = ttk.Frame( stats_frame, style='CardHeader.TFrame' )
            if tooltip:
                Tooltip( stat_container, tooltip )
            stat_container.grid( row=row, column=col, padx=5, pady=5, sticky="ew")
            stats_frame.grid_columnconfigure( col, weight=1 )

            label_name = ttk.Label( stat_container, text=name, style='StatName.TLabel' )
            label_name.pack( pady=(8, 0) )
            
            label_value = ttk.Label( stat_container, text="0.00 €", style='StatValue.TLabel' )
            label_value.pack( pady=(0, 8) )

            self.stats_labels[key] = label_value
        
        # Bouton de mise à jour des prix
        update_frame = ttk.Frame( header_frame, style='Card.TFrame' )
        update_frame.pack( pady=(15, 0) )
        
        self.update_button = ttk.Button(
            update_frame,
            text="🔄 Actualiser les Prix",
            command=self.update_prices,
            style='Primary.TButton'
        )
        self.update_button.pack()
        self.update_button.config( state=tk.DISABLED, text="⏳ Mise à jour..." )        
        
        self.last_update_label = ttk.Label(
            update_frame,
            text="Dernière mise à jour: Jamais",
            style='Info.TLabel'
        )
        Tooltip( self.last_update_label, "avec les valeurs actuelles du marché" )
        self.last_update_label.pack( pady=(5, 0))
    
    # -------------------------------------------------------------------------
    
    def create_transactions_tab( self, parent ):
        """Crée l'onglet des transactions du portefeuille"""
        form_frame = ttk.LabelFrame(
            parent, 
            text="Nouvelle Transaction",
            padding=15 #,
            # style='CardTrans.TFrame'
        )
        
        form_frame.pack( fill=tk.X, padx=10, pady=5 )

        fields_frame = ttk.Frame( form_frame, style='CardTrans.TFrame' )
        fields_frame.pack()

        for col in range(6):
            fields_frame.grid_columnconfigure( col, weight=1, pad=10 )

        padding_options = {'padx': 5, 'pady': 5}
        
        # Ligne 1
        ttk.Label( fields_frame, text="Type :" ).grid( row=0, column=0, sticky="e", **padding_options )
        self.type_var = tk.StringVar( value="achat" )
        type_frame = ttk.Frame( fields_frame, style='CardTrans.TFrame' )
        type_frame.grid( row=0, column=1, sticky="w", **padding_options )
        ttk.Radiobutton( type_frame, text="Achat", variable=self.type_var, value="achat", style='Buy.TRadiobutton' ).pack( side=tk.LEFT )
        ttk.Radiobutton( type_frame, text="Vente", variable=self.type_var, value="vente", style='Sell.TRadiobutton' ).pack( side=tk.LEFT )

        ttk.Label( fields_frame, text="Ticker :" ).grid( row=0, column=2, sticky="e", **padding_options )
        self.ticker_entry = ttk.Entry( fields_frame, width=15 )
        self.ticker_entry.grid( row=0, column=3, **padding_options )

        ttk.Label( fields_frame, text="Quantité :" ).grid( row=0, column=4, sticky="e", **padding_options )
        self.quantity_entry = ttk.Entry( fields_frame, width=15 )
        self.quantity_entry.grid( row=0, column=5, **padding_options )

        # Ligne 2
        ttk.Label( fields_frame, text="Prix (€) :" ).grid( row=1, column=0, sticky="e", **padding_options )
        self.price_entry = ttk.Entry( fields_frame, width=15 )
        self.price_entry.grid( row=1, column=1, sticky="w", **padding_options )

        ttk.Label( fields_frame, text="Frais (€) :" ).grid( row=1, column=2, sticky="e", **padding_options )
        self.fees_entry = ttk.Entry( fields_frame, width=15 )
        self.fees_entry.insert( 0, "0" )
        self.fees_entry.grid( row=1, column=3, **padding_options )

        ttk.Label( fields_frame, text="Date :" ).grid( row=1, column=4, sticky="e", **padding_options )
        self.date_entry = DateEntry(
            fields_frame, 
            width=12,
            justify='left',
            background='darkblue',
            foreground='white',
            borderwidth=1,
            date_pattern='dd-mm-yyyy',
            locale='fr_FR'
        )
        self.date_entry.grid( row=1, column=5, sticky="w", padx=(10,0), pady=5 ) # **padding_options )

        # Bouton
        add_button = ttk.Button( 
            fields_frame,
            text="➕ Ajouter la Transaction",
            command=self.add_transaction,
            style='Primary.TButton'
        )
        add_button.grid( row=2, column=0, columnspan=6, pady=( 15, 5 ) )
        
        # Liste des transactions
        list_frame = ttk.LabelFrame( 
            parent, 
            text="Historique des Transactions",
            padding=15
        )
        list_frame.pack( fill=tk.BOTH, expand=True, padx=10, pady=5 )
        
        tree_frame = ttk.Frame( list_frame )
        tree_frame.pack( fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar( tree_frame)
        scrollbar.pack( side=tk.RIGHT, fill=tk.Y)

        self.transactions_tree = ttk.Treeview( 
            tree_frame,
            columns=("ID", "Date", "Type", "Ticker", "Nom", "Quantité", "Prix", "Frais", "Total"),
            show="headings",
            yscrollcommand=scrollbar.set,
            height=6
        )
        scrollbar.config( command=self.transactions_tree.yview )

        # Manage select event
        self.transactions_tree.bind( "<<TreeviewSelect>>", self.on_transactions_tree_item_selected )
                
        columns_config = [
            ("ID", 40),
            ("Date", 90),
            ("Type", 80),
            ("Ticker", 70),
            ("Nom", 120),
            ("Quantité", 80),
            ("Prix", 120),
            ("Frais", 80),
            ("Total", 120)
        ]
        
        self.transactions_tree["displaycolumns"] = ("Date", "Type", "Ticker", "Nom", "Quantité", "Prix", "Frais", "Total")
        
        for col, width in columns_config:
            self.transactions_tree.heading( col, text=col)
            self.transactions_tree.column( col, width=width, anchor="center" )

        self.transactions_tree.pack( fill=tk.BOTH, expand=True)

        delete_button = ttk.Button( 
            list_frame,
            text="🗑️ Supprimer la Transaction Sélectionnée",
            command=self.delete_transaction,
            style='Danger.TButton'
        )
        delete_button.pack( pady=( 15, 5 ) )
    
    # -------------------------------------------------------------------------
    
    def create_positions_tab( self, parent ):
        """Crée l'onglet des positions du portefeuille"""
        positions_frame = ttk.Frame( parent, style='Card.TFrame', padding=15 )
        positions_frame.pack( fill=tk.BOTH, expand=True, padx=10, pady=10 )

        # Frame pour aligner le titre et le checkbutton
        header_frame = ttk.Frame( positions_frame, style='Card.TFrame' )
        header_frame.pack( fill=tk.X, pady=( 15, 15) )

        # Frame vide à gauche pour équilibrer
        left_spacer = ttk.Frame( header_frame, style='Card.TFrame' )
        left_spacer.pack( side=tk.LEFT, expand=True )

        title = ttk.Label( 
            header_frame,
            text="📊 Positions Actuelles du Portefeuille",
            style='Subtitle.TLabel'
        )
        title.pack( side=tk.LEFT )

        # Frame à droite contenant le checkbutton
        right_frame = ttk.Frame( header_frame, style='Card.TFrame' )
        right_frame.pack( side=tk.LEFT, expand=True )

        self.var_filter_positions = tk.BooleanVar( False )
        _chk = ttk.Checkbutton( right_frame, text="Positions Ouvertes", variable=self.var_filter_positions, command=self.command_filter_position )
        _chk.pack( side=tk.LEFT )
        Tooltip( _chk, "Filtrer les positions ouvertes")
            
        tree_frame = ttk.Frame( positions_frame)
        tree_frame.pack( fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar( tree_frame)
        scrollbar.pack( side=tk.RIGHT, fill=tk.Y)

        self.positions_tree = ttk.Treeview( 
            tree_frame,
            columns=( "Ticker", "Nom", "Quantité", "Coût Moyen", "Prix Courant",
                    "Valeur Actuelle", "P&L Non Réalisé", "P&L Réalisé"),
            show="headings",
            yscrollcommand=scrollbar.set,
            height=15
        )
        scrollbar.config( command=self.positions_tree.yview)
        
        columns_config = [
            ("Ticker", 70),
            ("Nom", 120),
            ("Quantité", 80),
            ("Coût Moyen", 120),
            ("Prix Courant", 120),
            ("Valeur Actuelle", 130),
            ("P&L Non Réalisé", 140),
            ("P&L Réalisé", 120)
        ]
        
        for col, width in columns_config:
            self.positions_tree.heading( col, text=col )
            self.positions_tree.column( col, width=width, anchor="center" )

        self.positions_tree.pack( fill=tk.BOTH, expand=True )
    
    # -------------------------------------------------------------------------
    
    def create_report_tab( self, parent ):
        """Crée l'onglet du rapport complet"""
        report_frame = ttk.Frame( parent, style='Card.TFrame', padding=15)
        report_frame.pack( fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        title = ttk.Label( report_frame,
                         text="📈 Rapport Complet du Portefeuille",
                         style='Subtitle.TLabel')
        title.pack( pady=(0, 15))
        
        self.report_text = scrolledtext.ScrolledText( 
            report_frame,
            font=("Consolas", 9),
            wrap=tk.WORD,
            height=25
        )
        self.report_text.pack( fill=tk.BOTH, expand=True )
        
        refresh_button = ttk.Button( 
            report_frame,
            text="🔄 Actualiser le Rapport",
            command=self.update_report,
            style='Primary.TButton'
        )
        refresh_button.pack( pady=(10, 0) )
    
    # -------------------------------------------------------------------------
    
    def add_transaction( self ):
        """Ajoute une nouvelle transaction"""
        try:
            type_trans = self.type_var.get()
            ticker = self.ticker_entry.get().strip().upper()
            quantity = float( self.quantity_entry.get() )
            price = float( self.price_entry.get() )
            date = self.date_entry.get().strip()
            fees = float( self.fees_entry.get() )

            if not ticker:
                messagebox.showerror("Erreur", "Le ticker est obligatoire")
                return
            
            if quantity <= 0 or price <= 0:
                messagebox.showerror( "Erreur", "La quantité et le prix doivent être des valeurs positives" )
                return

            self.portfolio.add_transaction( type_trans, ticker, quantity, price, date, fees )

            self.ticker_entry.delete(0, tk.END)
            self.quantity_entry.delete(0, tk.END)
            self.price_entry.delete(0, tk.END)
            self.fees_entry.delete(0, tk.END)
            self.fees_entry.insert(0, "0")

            self.refresh_display()
            
        except ValueError:
            messagebox.showerror( "Erreur", "Veuillez entrer des valeurs numériques valides" )
    
    # -------------------------------------------------------------------------
    
    def delete_transaction(self):
        """Supprime une transaction sélectionnée"""
        selected = self.transactions_tree.selection()
        if not selected:
            messagebox.showwarning( "Attention", "Veuillez sélectionner une transaction à supprimer" )
            return
        
        item = self.transactions_tree.item( selected[0] )
        transaction_id = int( item['values'][0] )
        ticker = item['values'][3]

        if messagebox.askyesno( "Confirmation", f"Voulez-vous vraiment supprimer la transaction #{transaction_id} {ticker} ?" ):
            self.portfolio.delete_transaction(transaction_id)
            self.refresh_display()
    
    # -------------------------------------------------------------------------
    
    def command_filter_position( self ):
        """ Filtre les positions ouvertes """
        
        self.update_positions_list()

    # -------------------------------------------------------------------------
    
    def on_transactions_tree_item_selected( self, event ):
        
        selected = self.transactions_tree.selection()
        
        # Get row's values of selected item
        selected_item = self.transactions_tree.item( selected[0] )["values"]
        
        # Get third column "Ticker"
        ticker = selected_item[ 3 ]
        
        # Put Ticker into ticker_entry
        self.ticker_entry.delete( 0, tk.END )
        self.ticker_entry.insert( 0, ticker )
            
    # -------------------------------------------------------------------------
    
    def update_title_label( self ):
        
        # Update title_label with new path.stem
        _path = Path( self.portfolio.filename )
        self.title_label.config( text=f"💰 Portefeuille : {_path.stem}" )
        
    # -------------------------------------------------------------------------
    
    def update_prices( self ):
        """Met à jour les prix courants"""
        self.update_button.config( state=tk.DISABLED, text="⏳ Mise à jour..." )
        
        def on_complete():
            self.window.after( 0, self._on_prices_updated )
        
        self.portfolio.update_current_prices( callback=on_complete )
    
    # -------------------------------------------------------------------------
    
    def _on_prices_updated( self ):
        """Callback après mise à jour des prix"""
        self.update_button.config( state=tk.NORMAL, text="🔄 Actualiser les Prix" )
        self.last_update_label.config(
            text=f"Dernière mise à jour: {datetime.now().strftime('%H:%M:%S')}"
        )
        self.refresh_display()
    
    # -------------------------------------------------------------------------
    
    def refresh_display(self):
        """Rafraîchit tous les affichages"""
        self.update_statistics()
        self.update_transactions_list()
        self.update_positions_list()
        self.update_report()
    
    # -------------------------------------------------------------------------
    
    def update_statistics( self ):
        """Met à jour les statistiques dans l'en-tête"""
        stats = self.portfolio.compute_statistics()
        
        self.stats_labels['net_invested'].config(text=f"{stats['net_invested']:.2f} €")
        self.stats_labels['current_value'].config(text=f"{stats['current_value']:.2f} €")
        
        for key in ['realized_pnl', 'unrealized_pnl', 'total_pnl']:
            value = stats[key]
            text = f"{'+' if value >= 0 else ''}{value:.2f} €"
            color = self.colors['success'] if value >= 0 else self.colors['danger']
            self.stats_labels[key].config(text=text, foreground=color)
        
        ret = stats['return_percentage']
        ret_text = f"{'+' if ret >= 0 else ''}{ret:.2f} %"
        ret_color = self.colors['success'] if ret >= 0 else self.colors['danger']
        self.stats_labels['return_percentage'].config(text=ret_text, foreground=ret_color)
    
    # -------------------------------------------------------------------------
    
    def update_transactions_list( self ):
        """Met à jour la liste des transactions"""
        for item in self.transactions_tree.get_children():
            self.transactions_tree.delete( item )

        for t in reversed( self.portfolio.transactions ):
            self.transactions_tree.insert(
                '', 
                'end', 
                values=(
                    t['id'],
                    t['date'],
                    t['type'].upper(),
                    t['ticker'],
                    t.get('name', 'N/A'),
                    f"{t['quantity']:.0f}",
                    f"{t['price']:.2f} €",
                    f"{t['fees']:.2f} €",
                    f"{t['total']:.2f} €"
                ),
                tags=('buy' if t['type'] == 'achat' else 'sell',)
            )

            self.transactions_tree.tag_configure( 'buy', background="#d4edda" )
            self.transactions_tree.tag_configure( 'sell', background="#f8d7da" )
            
    # -------------------------------------------------------------------------
    
    def update_positions_list( self ):
        """Met à jour la liste des positions"""
        
        for item in self.positions_tree.get_children():
            self.positions_tree.delete( item )
        
        positions = self.portfolio.compute_positions()

        for ticker, pos in sorted( positions.items() ):
            quantity = pos['quantity']
            avg_cost = pos['total_cost'] / quantity if quantity != 0 else 0.0
            current_price = self.portfolio.current_prices.get( ticker, 0.0 )
            current_value = quantity * current_price if current_price > 0 else 0.0
            unrealized_pnl = ( current_price - avg_cost) * quantity if current_price > 0 else 0.0

            current_price_text = f"{current_price:.2f} €" if current_price > 0 else "N/A"
            current_value_text = f"{current_value:.2f} €" if current_price > 0 else "N/A"
            unrealized_pnl_text = f"{'+' if unrealized_pnl >= 0 else ''}{unrealized_pnl:.2f} €" if current_price > 0 else "N/A"
            realized_pnl_text = f"{'+' if pos['realized_pnl'] >= 0 else ''}{pos['realized_pnl']:.2f} €"
            
            # Récupération du nom depuis la position (déjà mis en cache)
            _short_name = pos.get('name', 'N/A')

            _condition_positive = pos['realized_pnl'] + unrealized_pnl >= 0
            val = (
                    ticker,
                    _short_name,
                    f"{pos['quantity']:.0f}",
                    f"{avg_cost:.2f} €",
                    current_price_text,
                    current_value_text,
                    unrealized_pnl_text,
                    realized_pnl_text
            )
            
            # Filtrer les positions fermée
            #
            if self.var_filter_positions.get() == True:
                
                # N'afficher que les positions avec 'quantity' != 0
                if pos['quantity'] == 0: # quantité
                    continue
                
            self.positions_tree.insert( 
                '', 
                'end',
                values=val, 
                tags=( 'positive' if _condition_positive else 'negative',)
            )
        
        self.positions_tree.tag_configure( 'positive', background='#d4edda' )
        self.positions_tree.tag_configure( 'negative', background='#f8d7da' )
    
    # -------------------------------------------------------------------------
    
    def update_report(self):
        """Met à jour le rapport complet"""
        self.report_text.delete(1.0, tk.END)
        
        report = "="*80 + "\n"
        report += f"RAPPORT DE PORTEFEUILLE - {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
        report += "="*80 + "\n\n"
        
        stats = self.portfolio.compute_statistics()
        report += "💰 STATISTIQUES GLOBALES\n"
        report += "-"*80 + "\n"
        report += f"Capital investi net:     {stats['net_invested']:>20.2f} €\n"
        report += f"Valeur actuelle:         {stats['current_value']:>20.2f} €\n"
        report += f"Total investi:           {stats['total_invested']:>20.2f} €\n"
        report += f"Total vendu:             {stats['total_sold']:>20.2f} €\n"
        report += f"Frais totaux:            {stats['total_fees']:>20.2f} €\n"
        report += f"P&L réalisé:             {stats['realized_pnl']:>20.2f} €\n"
        report += f"P&L non réalisé:         {stats['unrealized_pnl']:>20.2f} €\n"
        report += f"P&L total:               {stats['total_pnl']:>20.2f} €\n"
        report += f"Rendement:               {stats['return_percentage']:>19.2f} %\n\n"

        # Positions
        positions = self.portfolio.compute_positions()
        report += "📊 POSITIONS ACTUELLES\n"
        report += "-"*80 + "\n"
        
        if positions:
            for ticker, pos in sorted( positions.items() ):
                avg_cost = pos['total_cost'] / pos['quantity'] if pos['quantity'] != 0 else 0.0
                report += f"{ticker:10s} | Qté: {pos['quantity']:>8.0f} | "
                report += f"Coût Moy: {avg_cost:>10.2f}€ | "
                report += f"Coût Total: {pos['total_cost']:>9.2f}€ | "
                report += f"P&L: {pos['realized_pnl']:>10.2f}€\n"
        else:
            report += "Aucune position ouverte\n"
        
        report += "\n"
        
        # Transactions
        report += "📋 HISTORIQUE DES TRANSACTIONS\n"
        report += "-"*80 + "\n"
        
        if self.portfolio.transactions:
            for t in self.portfolio.transactions:
                report += f"[{t['date']}] {t['type'].upper():6s} | {t['ticker']:8s} | "
                report += f"Qté: {t['quantity']:>8.0f} | Prix: {t['price']:>10.2f}€ | "
                report += f"Frais: {t['fees']:>8.2f}€ | Total: {t['total']:>9.2f}€\n"
        else:
            report += "Aucune transaction enregistrée\n"
        
        report += "\n" + "="*80 + "\n"
        report += "Fichier de données: " + str(self.portfolio.filename) + "\n"
        report += "="*80 + "\n"
        
        self.report_text.insert(1.0, report)

# -------------------------------------------------------------------------

def main():
    root = tk.Tk()
    app = PortfolioApp( root )
    root.mainloop()

if __name__ == "__main__":
    main()