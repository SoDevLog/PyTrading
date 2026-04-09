""" Tool_Finance - Calendrier des jours d'ouverture des bourses mondiales.

    Données : pandas_market_calendars (calendrier réel des marchés)

    >pip install tkcalendar pandas pandas_market_calendars
    >pip install "pandas_market_calendars==4.1.4" compatible python 3.9
    
"""

import tkinter as tk
import pandas as pd
import pandas_market_calendars as mcal

from tkinter import ttk
from tkcalendar import Calendar
from datetime import date

# -------------------------------------------------------------------------
# MAPPING CENTRALISE DES MARCHES
#
MARKETS = {
    'Paris (CAC40)': {
        'ticker': '^FCHI',
        'calendar': 'XPAR',
        'timezone': 'Europe/Paris',
    },
    'New York (NYSE)': {
        'ticker': '^NYA',
        'calendar': 'XNYS',
        'timezone': 'America/New_York',
    },
    'Nasdaq': {
        'ticker': '^IXIC',
        'calendar': 'NASDAQ',
        'timezone': 'America/New_York',
    },
    'Londres': {
        'ticker': '^FTSE',
        'calendar': 'XLON',
        'timezone': 'Europe/London',
    },
    'Francfort': {
        'ticker': '^GDAXI',
        'calendar': 'XETR',
        'timezone': 'Europe/Berlin',
    },
}

# Pour afficher tous les marchés disponibles :
#print( mcal.get_calendar_names() )

# -------------------------------------------------------------------------

class MarketCalendarApp:

    def __init__(self, root):
        self.root = root
        self.annee = date.today().year
        self.root.title("Calendrier des Bourses")
        self.root.geometry("600x450")
        
        self.setup_ui()

    def setup_ui(self):

        # --- Sélecteur de bourse ---
        frame_ctrl = tk.Frame(self.root)
        frame_ctrl.pack(pady=10)

        tk.Label(frame_ctrl, text='Bourse :').pack(side='left', padx=5)

        self.bourse_var = tk.StringVar(value='Paris (CAC40)')
        combo = ttk.Combobox(
            frame_ctrl,
            textvariable=self.bourse_var,
            values=list( MARKETS.keys() ),
            width=20,
            state='readonly'
        )
        combo.pack(side='left', padx=5)

        self.status_var = tk.StringVar( value="" )
        self.status_label = tk.Label( frame_ctrl, textvariable=self.status_var, fg='gray' )
        self.status_label.pack( side='left', padx=10 )

        # --- Calendrier --- #
        self.cal = Calendar(
            self.root,
            selectmode='day',
            year=self.annee,
            month=date.today().month,
            day=date.today().day,
            locale='fr_FR',
            firstweekday='monday',
        )
        self.cal.pack( padx=20, pady=10, fill='both', expand=True )

        # Tags passé
        self.cal.tag_config('open', background='#c8f7c5', foreground='darkgreen')
        self.cal.tag_config('closed', background='#f7c5c5', foreground='red')

        # Tags futur
        self.cal.tag_config('future_open', background='#e0e0e0', foreground='darkgreen')
        self.cal.tag_config('future_closed', background='#f7c5c5', foreground='red')

        # --- Légende ---
        frame_legende = tk.Frame(self.root)
        frame_legende.pack(pady=(0, 12))

        tk.Label(frame_legende, text='■', fg="#9dd699", font=('Arial', 14)).pack(side='left', padx=5, pady=(0,5))
        tk.Label(frame_legende, text='Ouvert').pack(side='left', padx=(0, 10))

        tk.Label(frame_legende, text='■', fg="#da9f9f", font=('Arial', 14)).pack(side='left', padx=5, pady=(0,5))
        tk.Label(frame_legende, text='Férié / Fermé').pack(side='left', padx=(0, 10))

        # --- Bindings ---
        combo.bind('<<ComboboxSelected>>', self._update_calendar)

        # Chargement initial
        self._update_calendar()

    # ---------------------------------------------------------------------
    # Eviter une erreur de comptatibilité entre 
    # les versions de pandas_market_calendars (4.1.4 vs 5.x)
    #
    def _get_open_days( self, calendar_name ):

        calendar = mcal.get_calendar(calendar_name)

        try:
            start = pd.Timestamp(f'{self.annee}-01-01')
            end   = pd.Timestamp(f'{self.annee}-12-31')
            schedule = calendar.schedule(
                start_date=start,
                end_date=end
            )
            self.status_label.config( fg='grey' )
        except Exception:
            self.status_var.set( "❌ Erreur de chargement du calendrier" )
            self.status_label.config( fg='red' )
            return set()

        return schedule

    # ---------------------------------------------------------------------

    def _update_calendar(self, *args):

        self.status_var.set("⏳ Chargement...")
        self.root.update()
        self.cal.calevent_remove('all')

        market = MARKETS[self.bourse_var.get()]
        schedule = self._get_open_days(market['calendar'])
        open_days = set(schedule.index.date)
        if not open_days:
            return

        today = date.today()
        today_info = schedule.loc[ schedule.index == pd.Timestamp(today) ]

        if today_info.index.tz is None:
            today_info.index = today_info.index.tz_localize('UTC')  # Localiser en UTC

        open_time = None
        close_time = None
        if not today_info.empty:
            open_time = today_info['market_open'].iloc[0]
            close_time = today_info['market_close'].iloc[0]

        all_the_days = pd.date_range(
            f'{self.annee}-01-01',
            f'{self.annee}-12-31',
            freq='D'
        )
        
        nb_feries = len(all_the_days) - len(open_days)

        for day in all_the_days:
            d = day.date()

            if d.weekday() < 5:  # ignorer week-ends

                is_open = d in open_days

                if d >= today:
                    tag = 'future_open' if is_open else 'future_closed'
                else:
                    tag = 'open' if is_open else 'closed'

                label = 'Ouvert' if is_open else 'Fermé'

                self.cal.calevent_create(d, label, tag)

        _status = f"{len(open_days)} jours ouverts / {nb_feries} fériés"
        if open_time and close_time:
            _status += f" | ouverture: {open_time.strftime('%H:%M')} fermeture: {close_time.strftime('%H:%M')}"
        else:
            _status += f" | marché fermé"
        self.status_var.set( _status )

# -------------------------------------------------------------------------

def main():
    root = tk.Tk()
    app = MarketCalendarApp( root )
    root.mainloop()
    
if __name__ == "__main__":
    main()