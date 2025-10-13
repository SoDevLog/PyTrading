""" 
    Indicateur technique complet basé sur le filtre de Kalman
    Compatible Python 3.9+
    
    Utilise yfinance pour les données
    period : 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max Either Use period parameter or use start and end
    interval : 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo Intraday data cannot extend last 60 days
        
"""
from matplotlib import dates, gridspec
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional, Union

import yfinance
warnings.filterwarnings('ignore')

# Paramètres pour l'extraction des données
company = {"name": "SAFRAN", "symbol": "SAF.PA"}
company = {'symbol': 'STMPA.PA', 'name' : 'STMICROELECTRONICS'}
#company = {"name": "STELLANTIS", "symbol": "STLAP.PA"}
#company = {'symbol': 'HO.PA', 'name' : 'THALES'}

date_start = '2025-09-26'
date_end = datetime.now()
interval_dates = '30m'

class IndicateurTendanceKalman:
    """
    Indicateur technique complet basé sur le filtre de Kalman
    Compatible Python 3.9+
    """
    
    def __init__(self, 
                 process_variance: Optional[float] = None,
                 measurement_variance: Optional[float] = None,
                 lookback: int = 50,
                 volatility_window: int = 20):
        """
        Initialisation du filtre de Kalman
        
        Args:
            process_variance: Variance du processus (Q) - adaptatif si None
            measurement_variance: Variance de mesure (R) - adaptatif si None
            lookback: Fenêtre de lookback pour les signaux
            volatility_window: Fenêtre pour calcul volatilité
        """
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.lookback = lookback
        self.volatility_window = volatility_window
        self.historique: Dict = {}
        self.length = 0
        
    def kalman_filter(self, 
                     y: Union[List, np.ndarray],
                     Q: Optional[float] = None,
                     R: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filtre de Kalman pour extraction de tendance
        
        Args:
            y: Série temporelle des prix
            Q: Variance du processus (bruit système)
            R: Variance de mesure (bruit observation)
            
        Returns:
            tendance: Série filtrée (tendance)
            velocite: Dérivée première (vitesse de changement)
            variance: Variance estimée
        """
        y = np.asarray(y, dtype=np.float64).flatten()
        n = len(y)
        
        if n < 3:
            return y.copy(), np.zeros_like(y), np.ones_like(y)
        
        # Initialisation adaptative des variances
        if Q is None:
            Q = np.var( np.diff(y) ) * 0.01  # 1% de la variance des différences
        if R is None:
            R = np.var( y ) * 0.1  # 10% de la variance totale
            
        # État: [position, vélocité]
        # Modèle: x(t) = F * x(t-1) + w(t) où w ~ N(0, Q)
        # Mesure: y(t) = H * x(t) + v(t) où v ~ N(0, R)
        
        # Matrices du système
        dt = 1.0  # Pas de temps unitaire
        F = np.array([[1.0, dt],   # Matrice de transition
                     [0.0, 1.0]], dtype=np.float64)
        
        H = np.array([[1.0, 0.0]], dtype=np.float64)  # Matrice d'observation
        
        # Covariances
        Q_matrix = Q * np.array([[dt**4/4, dt**3/2],
                                [dt**3/2, dt**2]], dtype=np.float64)
        R_matrix = np.array([[R]], dtype=np.float64)
        
        # Initialisation
        x = np.array([y[0], 0.0], dtype=np.float64)  # [position, vélocité]
        P = np.eye(2, dtype=np.float64) * 100.0  # Covariance initiale
        
        # Stockage résultats
        tendance = np.zeros(n, dtype=np.float64)
        velocite = np.zeros(n, dtype=np.float64)
        variance = np.zeros(n, dtype=np.float64)
        
        # Filtrage de Kalman
        for i in range(n):
            # Prédiction
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q_matrix
            
            # Innovation
            y_obs = np.array([y[i]], dtype=np.float64)
            y_pred = H @ x_pred
            innovation = y_obs - y_pred
            
            # Covariance de l'innovation
            S = H @ P_pred @ H.T + R_matrix
            
            # Gain de Kalman
            K = P_pred @ H.T / S[0, 0]
            
            # Mise à jour
            x = x_pred + K.flatten() * innovation[0]
            P = (np.eye(2) - np.outer(K, H)) @ P_pred
            
            # Stockage
            tendance[i] = x[0]
            velocite[i] = x[1]
            variance[i] = P[0, 0]
        
        # Cycle = résidu
        cycle = y - tendance
        
        return tendance, velocite, cycle, variance
    
    def calculer_variances_adaptatives(
        self, 
        prix: np.ndarray,
        volatility: np.ndarray
    ) -> Tuple[float, float]:
        
        """
        Calcul adaptatif des variances Q et R basé sur la volatilité
        
        Args:
            prix: Série de prix
            volatility: Volatilité calculée
            
        Returns:
            Q: Variance du processus
            R: Variance de mesure
        """
        # Variance des différences (mouvement du système)
        diff_var = np.var( np.diff(prix) )
        
        # Adaptation à la volatilité actuelle
        vol_mean = np.mean( volatility )
        if vol_mean == 0:
            vol_factor = 1.0
        else:
            vol_factor = np.clip( volatility[-1] / vol_mean, 0.3, 3.0 )

        # Q: variance du processus (plus élevée = plus réactif)
        Q = diff_var * 0.01 * vol_factor
        
        # R: variance de mesure (plus élevée = plus de lissage)
        R = np.var( prix ) * 0.05 / vol_factor
        
        return Q, R
    
    def calculer_signaux_tendance(
        self,
        prix: np.ndarray,
        tendance: np.ndarray,
        velocite: np.ndarray,
        cycle: np.ndarray,
        variance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        """
        Génère les signaux de trading basés sur le filtre de Kalman
        
        Returns:
            signaux: Array de signaux (-1: vente, 0: neutre, 1: achat)
            force_tendance: Force de la tendance
            position_relative: Position relative au trend
            acceleration: Accélération (dérivée de la vélocité)
        """
        n = len(prix)
        signaux = np.zeros(n, dtype=np.int8)
        force_tendance = np.zeros(n, dtype=np.float64)
        
        # 1. Accélération (dérivée de la vélocité)
        acceleration = np.gradient(velocite)
        
        # Lissage de l'accélération
        kernel = np.ones(5) / 5
        acceleration_smooth = np.convolve(
            np.pad(acceleration, (2, 2), mode='edge'),
            kernel,
            mode='valid'
        )
        
        # 2. Position relative par rapport à la tendance
        position_relative = np.where(
            tendance != 0,
            cycle / tendance * 100,
            0
        )
        
        # 3. Incertitude (écart-type de Kalman)
        uncertainty = np.sqrt(variance)
        uncertainty_norm = uncertainty / np.mean(uncertainty) if np.mean(uncertainty) > 0 else np.ones_like(uncertainty)
        
        # 4. Force de la tendance (combinaison vélocité + stabilité)
        window_size = min(self.lookback // 5, 15)
        for i in range(window_size, n):
            start_idx = i - window_size
            
            # Vélocité moyenne
            vel_window = velocite[start_idx:i]
            vel_mean = np.mean(vel_window)
            
            # Consistance (faible écart-type = tendance forte)
            vel_std = np.std(vel_window)
            vel_consistency = 1.0 / (1.0 + vel_std)
            
            # Facteur de confiance (faible incertitude = forte confiance)
            confidence = 1.0 / (1.0 + uncertainty_norm[i])
            
            force_tendance[i] = vel_mean * vel_consistency * confidence
        
        # 5. Génération des signaux avec seuils adaptatifs
        signal_lookback = min(self.lookback, n)
        for i in range(signal_lookback, n):
            # Calcul des percentiles dynamiques
            force_window = force_tendance[max(0, i-signal_lookback):i]
            if len(force_window) > 0:
                percentile_65 = np.percentile(force_window, 65)
                percentile_35 = np.percentile(force_window, 35)
            else:
                percentile_65 = percentile_35 = 0
            
            # Conditions pour signal haussier
            condition_haussiere = (
                velocite[i] > 0.05 and  # Vélocité positive
                acceleration_smooth[i] > 0 and  # Accélération positive
                position_relative[i] > 0.5 and  # Prix au-dessus de la tendance
                force_tendance[i] > percentile_65 and  # Force significative
                uncertainty_norm[i] < 1.5  # Incertitude raisonnable
            )
            
            # Conditions pour signal baissier
            condition_baissiere = (
                velocite[i] < -0.05 and  # Vélocité négative
                acceleration_smooth[i] < 0 and  # Accélération négative
                position_relative[i] < -0.5 and  # Prix en-dessous de la tendance
                force_tendance[i] < percentile_35 and  # Force faible
                uncertainty_norm[i] < 1.5  # Incertitude raisonnable
            )
            
            if condition_haussiere:
                signaux[i] = 1
            elif condition_baissiere:
                signaux[i] = -1
            else:
                signaux[i] = 0
        
        return signaux, force_tendance, position_relative, acceleration_smooth
    
    def calculer_niveaux_support_resistance(
        self,
        prix: np.ndarray,
        tendance: np.ndarray,
        cycle: np.ndarray,
        variance: np.ndarray,
        window: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        """
        Calcule les niveaux de support/résistance avec bandes de confiance Kalman
        """
        if window is None:
            window = min(self.lookback // 2, 30)
        
        n = len(prix)
        supports = np.full(n, np.nan, dtype=np.float64)
        resistances = np.full(n, np.nan, dtype=np.float64)
        
        # Écart-type de l'incertitude
        std_dev = np.sqrt(variance)
        
        for i in range(window, n):
            cycle_window = cycle[i-window:i]
            
            # Bandes basées sur l'incertitude de Kalman (2 sigma)
            confidence_band = 2.0 * std_dev[i]
            
            # Support: tendance - incertitude + cycle négatif
            cycles_negatifs = cycle_window[cycle_window < 0]
            if len(cycles_negatifs) > 0:
                support_cycle = np.percentile(cycles_negatifs, 5)
            else:
                support_cycle = -np.std(cycle_window)
            
            supports[i] = tendance[i] + support_cycle - confidence_band
            
            # Résistance: tendance + incertitude + cycle positif
            cycles_positifs = cycle_window[cycle_window > 0]
            if len(cycles_positifs) > 0:
                resistance_cycle = np.percentile(cycles_positifs, 95)
            else:
                resistance_cycle = np.std(cycle_window)
            
            resistances[i] = tendance[i] + resistance_cycle + confidence_band
        
        return supports, resistances
    
    def calculer_indicateur_complet(
        self,
        prix: Union[List, np.ndarray],
        dates: Optional[pd.DatetimeIndex] = None,
        adaptive_variance: bool = True
    ) -> Dict:
        
        """
        Calcule l'indicateur technique complet avec filtre de Kalman
        """
        prix = np.asarray(prix, dtype=np.float64)
        self.length = len(prix)

        if self.length < self.volatility_window:
            raise ValueError( f"Pas assez de données. Minimum: {self.volatility_window}" )
        
        if dates is None:
            dates = pd.date_range(start='2020-01-01', periods=self.length, freq='D')

        # 1. Calcul volatilité
        log_returns = np.diff(np.log(np.maximum(prix, 1e-10)))
        returns_series = pd.Series(log_returns)
        volatility_series = returns_series.rolling(
            window=self.volatility_window,
            min_periods=1
        ).std()
        
        volatility = volatility_series.bfill().values
        volatility = np.concatenate([[volatility[0]], volatility])
        
        # 2. Choix des variances Q et R
        if adaptive_variance:
            Q, R = self.calculer_variances_adaptatives(prix, volatility)
        else:
            Q = self.process_variance or np.var(np.diff(prix)) * 0.01
            R = self.measurement_variance or np.var(prix) * 0.1
        
        # 3. Application du filtre de Kalman
        try:
            tendance, velocite, cycle, variance = self.kalman_filter(prix, Q, R)
        except Exception as e:
            print( f"Erreur filtre Kalman, utilisation fallback: {e}" )
            # Fallback: moyenne mobile
            tendance = self._moyenne_mobile_centree(prix, 20)
            velocite = np.gradient(tendance)
            cycle = prix - tendance
            variance = np.ones_like(prix) * np.var(cycle)
        
        # 4. Calcul des signaux
        signaux, force_tendance, position_relative, acceleration = self.calculer_signaux_tendance(
            prix, tendance, velocite, cycle, variance
        )
        
        # 5. Support/Résistance
        supports, resistances = self.calculer_niveaux_support_resistance(
            prix, tendance, cycle, variance
        )
        
        # 6. Métriques qualité
        trend_quality = self.evaluer_qualite_tendance(tendance, cycle, prix)
        
        # 7. Stockage
        self.historique = {
            'dates': dates,
            'prix': prix,
            'tendance': tendance,
            'velocite': velocite,
            'cycle': cycle,
            'variance': variance,
            'signaux': signaux,
            'force_tendance': force_tendance,
            'position_relative': position_relative,
            'acceleration': acceleration,
            'supports': supports,
            'resistances': resistances,
            'Q': Q,
            'R': R,
            'trend_quality': trend_quality,
            'volatility': volatility
        }
        
        return self.historique
    
    def _moyenne_mobile_centree( self, y: np.ndarray, window: int ) -> np.ndarray:
        """Moyenne mobile centrée comme fallback"""
        n = len(y)
        trend = np.zeros_like(y)
        half_window = window // 2
        
        for i in range(n):
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)
            trend[i] = np.mean(y[start:end])
        
        return trend
    
    def evaluer_qualite_tendance(
        self,
        tendance: np.ndarray,
        cycle: np.ndarray,
        prix: np.ndarray
    ) -> Dict[str, float]:
        
        """Évalue la qualité du filtrage de Kalman"""
        
        # R² avec gestion des cas limites
        prix_mean = np.mean(prix)
        ss_total = np.sum((prix - prix_mean)**2)
        ss_residual = np.sum(cycle**2)
        
        r_squared = 1.0 - (ss_residual / ss_total) if ss_total > 1e-10 else 0.0
        r_squared = np.clip(r_squared, 0.0, 1.0)
        
        # Lissage
        trend_diff2 = np.diff(tendance, n=2)
        smoothness = 1.0 / (1.0 + np.std(trend_diff2)) if len(trend_diff2) > 0 else 0.0
        
        # Signal/Noise ratio
        var_trend = np.var(tendance)
        var_cycle = np.var(cycle)
        signal_noise_ratio = var_trend / var_cycle if var_cycle > 1e-10 else 1.0
        
        return {
            'r_squared': float(r_squared),
            'smoothness': float(smoothness),
            'signal_noise_ratio': float(signal_noise_ratio),
            'trend_strength': float(r_squared * smoothness)
        }
    
    def generer_rapport_performance( self ) -> Dict:
        
        """Rapport performance des signaux"""
        if not self.historique:
            raise ValueError( "Calculer d'abord l'indicateur" )
        
        signaux = self.historique['signaux']
        prix = self.historique['prix']
        
        positions = np.where(signaux != 0)[0]
        
        if len(positions) < 2:
            return {"message": "Pas assez de signaux pour évaluer la performance"}
        
        rendements_signals = []
        
        for i in range(len(positions) - 1):
            pos_entree = positions[i]
            pos_sortie = positions[i + 1]
            
            if signaux[pos_entree] == 1:
                rendement = (prix[pos_sortie] - prix[pos_entree]) / prix[pos_entree] * 100
                rendements_signals.append(rendement)
        
        if not rendements_signals:
            return {"message": "Aucun cycle d'achat/vente complet"}
        
        rendements_array = np.array(rendements_signals)
        
        return {
            'nb_signaux': len(rendements_signals),
            'rendement_moyen': float(np.mean(rendements_array)),
            'rendement_total': float(np.sum(rendements_array)),
            'rendement_std': float(np.std(rendements_array)),
            'taux_reussite': float(np.mean(rendements_array > 0) * 100),
            'sharpe_ratio': float(np.mean(rendements_array) / np.std(rendements_array)) if np.std(rendements_array) > 0 else 0.0,
            'max_drawdown': float(np.min(rendements_array)),
            'max_gain': float(np.max(rendements_array))
        }
    
    def visualiser_indicateur(
        self,
        ax1: Optional[plt.Axes] = None,
        ax2: Optional[plt.Axes] = None,
        ax3: Optional[plt.Axes] = None,
        ax4: Optional[plt.Axes] = None,
        tick_positions: Optional[np.ndarray] = None,
        tick_labels: Optional[List[str]] = None,
        lines: Optional[List] = None
    ) -> None:
        
        """Visualisation complète de l'indicateur Kalman"""
        if not self.historique:
            raise ValueError( "Calculer d'abord l'indicateur" )
        
        hist = self.historique
       
        # 1. Prix et tendance avec bandes de confiance
        _lines, = ax1.plot( hist['dates'], hist['prix'], 'b-', alpha=0.7,
                label='Prix', linewidth=1)
        lines.append(_lines)
        
        ax1.plot( hist['dates'], hist['tendance'], 'r-',
                linewidth=2, label='Tendance Kalman')
        
        # Bandes de confiance (±2 sigma)
        std_dev = np.sqrt(hist['variance'])
        upper_band = hist['tendance'] + 2 * std_dev
        lower_band = hist['tendance'] - 2 * std_dev
        
        ax1.fill_between(hist['dates'], lower_band, upper_band,
                        alpha=0.2, color='red', label='Bande confiance (95%)')
        
        # Support/Résistance
        mask_support = ~np.isnan(hist['supports'])
        mask_resistance = ~np.isnan(hist['resistances'])
        
        if np.any(mask_support):
            ax1.plot(
                np.array(hist['dates'])[mask_support],
                hist['supports'][mask_support],
                'g--', alpha=0.6, label='Support'
            )

        if np.any(mask_resistance):
            ax1.plot(
                np.array(hist['dates'])[mask_resistance],
                hist['resistances'][mask_resistance],
                'orange', linestyle='--', alpha=0.6, label='Résistance'
            )
        
        # Signaux
        signaux_achat = hist['signaux'] == 1
        signaux_vente = hist['signaux'] == -1
        
        if np.any(signaux_achat):
            ax1.scatter(
                np.array(hist['dates'])[signaux_achat],
                hist['prix'][signaux_achat],
                color='green', marker='^', s=100,
                label='Achat', zorder=5
            )
        
        if np.any(signaux_vente):
            ax1.scatter(
                np.array(hist['dates'])[signaux_vente],
                hist['prix'][signaux_vente],
                color='red', marker='v', s=100,
                label='Vente', zorder=5
            )

        if __name__ == "__main__":
            _title = f"Prix et Tendance Kalman {company['name']} [{self.length}] {interval_dates}"
        else:
            _title = f"Prix et Tendance Kalman"
        ax1.set_title( _title )
        ax1.legend(loc='best', fontsize=8)
        ax1.grid( True, alpha=0.8 )

        # 2. Cycle et incertitude
        ax2_twin = ax2.twinx()
        
        ax2.plot( hist['dates'], hist['cycle'], 'purple', linewidth=1, label='Cycle' )
        ax2.axhline( y=0, color='black', linestyle='-', alpha=0.5 )
        
        positive_mask = hist['cycle'] > 0
        negative_mask = hist['cycle'] < 0
        
        ax2.fill_between(hist['dates'], hist['cycle'], 0,
                        where=positive_mask, alpha=0.3, color='green')
        ax2.fill_between(hist['dates'], hist['cycle'], 0,
                        where=negative_mask, alpha=0.3, color='red')
        
        # Incertitude de Kalman
        uncertainty = np.sqrt(hist['variance'])
        ax2_twin.plot( hist['dates'], uncertainty, 'orange',
                     linewidth=1.5, label='Incertitude σ' )
        ax2_twin.fill_between( hist['dates'], uncertainty, 0,
                             alpha=0.2, color='orange' )
        
        ax2.set_title('Composante Cyclique et Incertitude Kalman')
        ax2.set_ylabel('Cycle', color='purple')
        ax2_twin.set_ylabel('Incertitude', color='orange')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # 3. Vélocité et accélération
        ax3_twin = ax3.twinx()
        
        ax3.plot(hist['dates'], hist['velocite'], 'darkblue',
                linewidth=1.5, label='Vélocité')
        ax3_twin.plot(hist['dates'], hist['acceleration'], 'darkorange',
                     linewidth=1, label='Accélération')
        
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3_twin.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Zones de vélocité
        vel_pos = hist['velocite'] > 0
        vel_neg = hist['velocite'] < 0
        
        ax3.fill_between(hist['dates'], hist['velocite'], 0,
                        where=vel_pos, alpha=0.2, color='green')
        ax3.fill_between(hist['dates'], hist['velocite'], 0,
                        where=vel_neg, alpha=0.2, color='red')
        
        ax3.set_title('Vélocité et Accélération de la Tendance')
        ax3.set_ylabel('Vélocité', color='darkblue')
        ax3_twin.set_ylabel('Accélération', color='darkorange')
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        # 4. Volatilité et confiance
        ax4_twin = ax4.twinx()
        
        volatility = hist.get('volatility', np.ones_like(hist['prix']))
        ax4.plot(hist['dates'], volatility * 100, 'darkred',
                linewidth=1, label='Volatilité (%)')
        ax4.fill_between(hist['dates'], volatility * 100, 0,
                        alpha=0.2, color='red')
        
        # Confiance des signaux
        confiance = np.abs(hist['force_tendance'])
        if np.max(confiance) > np.min(confiance):
            confiance_norm = (confiance - np.min(confiance)) / (np.max(confiance) - np.min(confiance))
        else:
            confiance_norm = np.ones_like(confiance) * 0.5
            
        ax4_twin.plot(hist['dates'], confiance_norm * 100, 'darkgreen',
                     linewidth=2, label='Confiance Signal (%)')
        
        # Zones de confiance
        high_conf = confiance_norm > 0.7
        med_conf = (confiance_norm > 0.3) & (confiance_norm <= 0.7)
        low_conf = confiance_norm <= 0.3
        
        ax4_twin.fill_between(hist['dates'], 0, 100, where=high_conf,
                             alpha=0.1, color='green', label='Zone Forte')
        ax4_twin.fill_between(hist['dates'], 0, 100, where=med_conf,
                             alpha=0.1, color='yellow', label='Zone Modérée')
        ax4_twin.fill_between(hist['dates'], 0, 100, where=low_conf,
                             alpha=0.1, color='red', label='Zone Faible')
        
        ax4.set_title('Volatilité du Marché et Confiance des Signaux')
        ax4.set_ylabel('Volatilité (%)', color='darkred')
        ax4_twin.set_ylabel('Confiance (%)', color='darkgreen')
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='lower left', fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # Gestion des ticks
        if __name__ == "__main__":
            ax1.set_xticks( tick_positions )
            ax1.set_xticklabels( tick_labels )
        else:
            ax4.xaxis.set_major_formatter( dates.DateFormatter('%d-%m-%y') )
           
        # plt.setp( ax1.get_xticklabels(), visible=True )
        # plt.setp( ax2.get_xticklabels(), visible=False )
        # plt.setp( ax3.get_xticklabels(), visible=False )
        # plt.setp( ax4.get_xticklabels(), visible=False )

        #plt.tight_layout()

        if __name__ == "__main__":
            plt.show()
        
        self._afficher_rapport()
    
    def _afficher_rapport( self ) -> None:
        """Affichage du rapport de performance"""
        print( "\n" + "="*50)
        print( "MÉTRIQUES DE QUALITÉ - FILTRE DE KALMAN" )
        print( "="*50)
        
        quality = self.historique['trend_quality']
        print( f"R² (variance expliquée): {quality['r_squared']:.3f}" )
        print( f"Lissage: {quality['smoothness']:.3f}" )
        print( f"Ratio Signal/Bruit: {quality['signal_noise_ratio']:.3f}" )
        print( f"Force tendance: {quality['trend_strength']:.3f}" )
        print( f"Q (variance processus): {self.historique['Q']:.6f}" )
        print( f"R (variance mesure): {self.historique['R']:.6f}" )
        
        # Performance
        try:
            perf = self.generer_rapport_performance()
            if 'nb_signaux' in perf:
                print( f"\nPERFORMANCE:" )
                print( f"Signaux: {perf['nb_signaux']}" )
                print( f"Rendement moyen: {perf['rendement_moyen']:.2f}%" )
                print( f"Taux réussite: {perf['taux_reussite']:.1f}%" )
                print( f"Sharpe: {perf['sharpe_ratio']:.2f}" )
            else:
                print( f"\n{perf['message']}" )
        except Exception as e:
            print( f"\nErreur calcul performance: {e}" )


def tester_indicateur_kalman():
    """Test complet du filtre de Kalman"""

    ticker = yfinance.Ticker( company['symbol'] )
    data = ticker.history(
        start=date_start,
        end=date_end,
        interval=interval_dates,
        prepost=True
    )
    
    # Index continu pour le graphique
    data['index_continuous'] = np.arange(len(data))
    x_axis = data['index_continuous'].values
    
    # Configuration des ticks pour l'axe x
    n_ticks = min(10, len(data))
    tick_indices = np.linspace(0, len(data)-1, n_ticks, dtype=int)
    tick_positions = data['index_continuous'].values[tick_indices]
    tick_labels = [data.index[i].strftime('%Y-%m-%d') for i in tick_indices]
    
    prix = data['Close'].values
    dates = x_axis
    
    # Test indicateur
    print( "="*60 )
    print( "TEST INDICATEUR KALMAN - Python 3.9 Compatible" )
    print( "="*60 )
    print( f"Action: {company['name']} ({company['symbol']})" )
    print( f"Période: {date_start} à {date_end.strftime('%Y-%m-%d')}" )
    print( f"Nombre de points: {len(prix)}" )
    print( "-"*60 )
    
    try:
        # Création de l'indicateur avec paramètres optimisés
        indicateur = IndicateurTendanceKalman(
            lookback=40,
            volatility_window=15
        )
        
        # Calcul complet avec variances adaptatives
        resultats = indicateur.calculer_indicateur_complet(
            prix, 
            dates,
            adaptive_variance=True
        )
        
        print( "✓ Calcul réussi!" )
        print( f"✓ Points traités: {len(prix)}" )
        print( f"✓ Signaux générés: {np.sum(resultats['signaux'] != 0)}" )
        print( f"  - Signaux achat: {np.sum(resultats['signaux'] == 1)}" )
        print( f"  - Signaux vente: {np.sum(resultats['signaux'] == -1)}" )
        
        # Statistiques sur la vélocité
        print( f"\nStatistiques Kalman:" )
        print( f"  - Vélocité moyenne: {np.mean(resultats['velocite']):.4f}" )
        print( f"  - Vélocité max: {np.max(resultats['velocite']):.4f}" )
        print( f"  - Vélocité min: {np.min(resultats['velocite']):.4f}" )
        print( f"  - Incertitude moyenne: {np.mean(np.sqrt(resultats['variance'])):.4f}" )
        
        # Visualisation
        print( "\n" + "-"*60)
        print( "Génération des graphiques..." )
        
        fig = plt.figure( figsize=(16, 9) )
        plt.style.use('default')

        params = {"left": 0.05, "bottom": 0.05, "right": 0.95, "top": 0.93, "hspace": 0.3}
        grid_spec = gridspec.GridSpec( 4, 1, height_ratios=[3, 2, 2, 2], **params )
        
        ax1 = fig.add_subplot( grid_spec[0] )
        ax2 = fig.add_subplot( grid_spec[1], sharex=ax1 )
        ax3 = fig.add_subplot( grid_spec[2], sharex=ax1 )
        ax4 = fig.add_subplot( grid_spec[3], sharex=ax1 )

        indicateur.visualiser_indicateur(
            ax1, ax2, ax3, ax4,
            tick_positions=tick_positions,
            tick_labels=tick_labels
        )
        
        return indicateur, resultats
        
    except Exception as e:
        print( f"✗ Erreur: {e}" )
        import traceback
        traceback.print_exc()
        raise


def comparer_dernier_signal(indicateur: IndicateurTendanceKalman) -> None:
    """Analyse du dernier signal généré"""
    if not indicateur.historique:
        print( "Aucun historique disponible" )
        return
    
    hist = indicateur.historique
    signaux = hist['signaux']
    
    # Trouver le dernier signal non-nul
    signaux_indices = np.where(signaux != 0)[0]
    
    if len(signaux_indices) == 0:
        print( "\nAucun signal généré sur la période" )
        return
    
    last_signal_idx = signaux_indices[-1]
    last_signal = signaux[last_signal_idx]
    
    print( "\n" + "="*60)
    print( "ANALYSE DU DERNIER SIGNAL" )
    print( "="*60)
    
    signal_type = "ACHAT" if last_signal == 1 else "VENTE"
    print( f"Type: {signal_type}" )
    print( f"Position: {last_signal_idx} / {len(signaux)}" )
    
    print( f"\nConditions au moment du signal:" )
    print( f"  - Prix: {hist['prix'][last_signal_idx]:.2f}" )
    print( f"  - Tendance: {hist['tendance'][last_signal_idx]:.2f}" )
    print( f"  - Vélocité: {hist['velocite'][last_signal_idx]:.4f}" )
    print( f"  - Accélération: {hist['acceleration'][last_signal_idx]:.4f}" )
    print( f"  - Position relative: {hist['position_relative'][last_signal_idx]:.2f}%" )
    print( f"  - Force tendance: {hist['force_tendance'][last_signal_idx]:.4f}" )
    print( f"  - Incertitude: {np.sqrt(hist['variance'][last_signal_idx]):.4f}" )
    
    # État actuel (dernier point)
    current_idx = len(signaux) - 1
    print( f"\nÉtat actuel (dernier point):" )
    print( f"  - Prix: {hist['prix'][current_idx]:.2f}" )
    print( f"  - Tendance: {hist['tendance'][current_idx]:.2f}" )
    print( f"  - Vélocité: {hist['velocite'][current_idx]:.4f}" )
    print( f"  - Accélération: {hist['acceleration'][current_idx]:.4f}" )
    print( f"  - Support: {hist['supports'][current_idx]:.2f}" )
    print( f"  - Résistance: {hist['resistances'][current_idx]:.2f}" )
    
    # Recommandation
    print( f"\nRECOMMANDATION:" )
    if hist['velocite'][current_idx] > 0 and hist['acceleration'][current_idx] > 0:
        print( "  → Momentum HAUSSIER détecté" )
    elif hist['velocite'][current_idx] < 0 and hist['acceleration'][current_idx] < 0:
        print( "  → Momentum BAISSIER détecté" )
    else:
        print( "  → Momentum INCERTAIN - Prudence recommandée" )
    
    # Distance aux supports/résistances
    dist_support = ((hist['prix'][current_idx] - hist['supports'][current_idx]) / 
                   hist['prix'][current_idx] * 100)
    dist_resistance = ((hist['resistances'][current_idx] - hist['prix'][current_idx]) / 
                      hist['prix'][current_idx] * 100)
    
    print( f"\nNiveaux clés:" )
    print( f"  - Distance au support: {dist_support:.2f}%" )
    print( f"  - Distance à la résistance: {dist_resistance:.2f}%" )


if __name__ == "__main__":
    # Exécution du test
    indicateur, resultats = tester_indicateur_kalman()
    
    # Analyse du dernier signal
    comparer_dernier_signal(indicateur)
    
    print( "\n" + "="*60)
    print( "Test terminé avec succès!" )
    print( "="*60)