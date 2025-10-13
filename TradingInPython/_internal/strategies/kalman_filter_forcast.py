""" 
    Indicateur technique complet basé sur le filtre de Kalman
    Compatible Python 3.9+
    
    Utilise yfinance pour les données
    period : 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max Either Use period parameter or use start and end
    interval : 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo Intraday data cannot extend last 60 days
    
    Ajout de prévision à court terme au filtre de Kalman
    
    - calculer_signaux_achat_vente : signaux d'achat/vente avec anticipation
    - calculer_signaux_achat_vente_adaptatifs : threshold adaptatif
    - calculer_niveaux_support_resistance : niveaux dynamiques avec bandes de confiance
    - calculer_indicateur_complet : calcul complet avec prévision
        - kalman_forecast
    - visualiser_indicateur
        - visualiser_avec_prevision
        
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance

if __name__ != "__main__":
    import debug.func as debug
#import warnings

from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
from matplotlib import gridspec

#warnings.filterwarnings('ignore')

# Paramètres pour l'extraction des données
company = {"name": "SAFRAN", "symbol": "SAF.PA"}
company = {'symbol': 'STMPA.PA', 'name' : 'STMICROELECTRONICS'}
#company = {"name": "STELLANTIS", "symbol": "STLAP.PA"}
#company = {'symbol': 'HO.PA', 'name' : 'THALES'}

date_start = '2025-09-26'
date_end = datetime.now()
interval_dates = '30m'

WINDOW_TREND_SIZE_MAX = 500 # self.lookback // 5
WINDOW_SUPPORT_RESISTANCE_MAX = 250 # self.lookback // 2

class IndicateurTendanceKalman:
    """
    Indicateur technique complet basé sur le filtre de Kalman
    Compatible Python 3.9+
    """
    
    def __init__(self, 
                 process_variance: Optional[float] = None,
                 measurement_variance: Optional[float] = None,
                 process_noise_factor: float = 0.01,  # Q scaling
                 measurement_noise_factor: float = 0.05,  # R scaling
                 lookback: int = 50,
                 volatility_window: int = 20,
                 signal_buy_sell_velocity_threshold: float = 0.03,
                 signal_buy_sell_veloc_adapt_thresh: bool = False,
                 n_forecast: int = 10):
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
        self.process_noise_factor = process_noise_factor
        self.measurement_noise_factor = measurement_noise_factor
        self.lookback = lookback
        self.volatility_window = volatility_window
        self.historique: Dict = {}
        self.length = 0
        self.n_forecast = n_forecast
        self.signal_buy_sell_veloc_adapt_thresh = signal_buy_sell_veloc_adapt_thresh
        self.signal_buy_sell_velocity_threshold = signal_buy_sell_velocity_threshold
        
        if self.lookback != 0:
            self.window_trend_size = self.lookback // 5
            self.window_support_resistance_size = self.lookback // 2

    def kalman_forecast(
        self, 
        y: Union[List, np.ndarray],
        Q: Optional[float] = None,
        R: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        
        """
        Filtre de Kalman avec prévision à court terme
        
        Args:
            y: Série temporelle des prix
            Q: Variance du processus
            R: Variance de mesure
            
        Returns:
            tendance: Série filtrée (tendance)
            velocite: Dérivée première
            cycle: Résidu
            variance: Variance estimée
            forecast_data: Dictionnaire avec les prévisions
        """
        y = np.asarray(y, dtype=np.float64).flatten()
        n = len(y)
        
        if n < 3:
            return y.copy(), np.zeros_like(y), np.zeros_like(y), np.ones_like(y), {}
        
        # Initialisation adaptative des variances
        if Q is None:
            Q = np.var(np.diff(y)) * 0.01
        if R is None:
            R = np.var(y) * 0.1
        
        # Matrices du système
        dt = 1.0
        F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        H = np.array([[1.0, 0.0]], dtype=np.float64)
        
        Q_matrix = Q * np.array([[dt**4/4, dt**3/2],
                                [dt**3/2, dt**2]], dtype=np.float64)
        R_matrix = np.array([[R]], dtype=np.float64)
        
        # Initialisation
        x = np.array([y[0], 0.0], dtype=np.float64)
        P = np.eye(2, dtype=np.float64) * 100.0
        
        # Stockage résultats (données historiques)
        tendance = np.zeros(n, dtype=np.float64)
        velocite = np.zeros(n, dtype=np.float64)
        variance = np.zeros(n, dtype=np.float64)
        
        # === PHASE 1: FILTRAGE (sur données historiques) ===
        for i in range(n):
            # Prédiction
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q_matrix
            
            # Observation et correction
            y_obs = np.array([y[i]], dtype=np.float64)
            y_pred = H @ x_pred
            innovation = y_obs - y_pred
            
            S = H @ P_pred @ H.T + R_matrix
            K = P_pred @ H.T / S[0, 0]
            
            x = x_pred + K.flatten() * innovation[0]
            P = (np.eye(2) - np.outer(K, H)) @ P_pred
            
            tendance[i] = x[0]
            velocite[i] = x[1]
            variance[i] = P[0, 0]
        
        # === PHASE 2: PRÉVISION (sans observations) ===
        forecast_tendance = np.zeros(self.n_forecast, dtype=np.float64)
        forecast_velocite = np.zeros(self.n_forecast, dtype=np.float64)
        forecast_variance = np.zeros(self.n_forecast, dtype=np.float64)
        forecast_upper = np.zeros(self.n_forecast, dtype=np.float64)
        forecast_lower = np.zeros(self.n_forecast, dtype=np.float64)
        
        # État initial = dernier état filtré
        x_forecast = x.copy()
        P_forecast = P.copy()
        
        for i in range(self.n_forecast):
            # Prédiction pure (pas de correction)
            x_forecast = F @ x_forecast
            P_forecast = F @ P_forecast @ F.T + Q_matrix
            
            # Stockage
            forecast_tendance[i] = x_forecast[0]
            forecast_velocite[i] = x_forecast[1]
            forecast_variance[i] = P_forecast[0, 0]
            
            # Intervalles de confiance (95%)
            std_dev = np.sqrt(P_forecast[0, 0])
            forecast_upper[i] = x_forecast[0] + 1.96 * std_dev
            forecast_lower[i] = x_forecast[0] - 1.96 * std_dev
        
        # Calcul du cycle
        cycle = y - tendance
        
        # Données de prévision
        forecast_data = {
            'n_forecast': self.n_forecast,
            'tendance': forecast_tendance,
            'velocite': forecast_velocite,
            'variance': forecast_variance,
            'upper_bound': forecast_upper,
            'lower_bound': forecast_lower,
            'confidence_width': forecast_upper - forecast_lower
        }
        
        return tendance, velocite, cycle, variance, forecast_data
    
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
        Q = diff_var * self.process_noise_factor * vol_factor
        
        # R: variance de mesure (plus élevée = plus de lissage)
        R = np.var( prix ) * self.measurement_noise_factor / vol_factor
        
        return Q, R
    
    # -------------------------------------------------------------------------
    
    def calculer_signaux_achat_vente(
            self,
            prix: np.ndarray,
            tendance: np.ndarray,
            velocite: np.ndarray,
            cycle: np.ndarray,
            variance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        """
        Génère les signaux en intégrant les prévisions
        """
        n = len(prix)
        signaux = np.zeros(n, dtype=np.int8)
        force_tendance = np.zeros(n, dtype=np.float64)
        
        # Calcul de l'accélération
        acceleration = np.gradient( velocite )
        kernel = np.ones( 5 ) / 5
        acceleration_smooth = np.convolve(
            np.pad( acceleration, (2, 2), mode='edge'),
            kernel,
            mode='valid'
        )
        
        # Position relative
        position_relative = np.where(
            tendance != 0,
            cycle / tendance * 100,
            0
        )
        
        # Incertitude normalisée
        uncertainty = np.sqrt( variance )
        uncertainty_norm = uncertainty / np.mean(uncertainty) if np.mean(uncertainty) > 0 else np.ones_like(uncertainty)
        
        # Force de la tendance
        window_size = min( self.window_trend_size, WINDOW_TREND_SIZE_MAX )
        if __name__ != "__main__":
            debug.print( f"self.lookback: {self.lookback}" )
            debug.print( f"window_trend_size: {window_size}" )
            debug.print( f"window_support_resistance_size: {self.window_support_resistance_size}" )
        
        for i in range( window_size, n ):
            start_idx = i - window_size
            vel_window = velocite[start_idx:i]
            vel_mean = np.mean(vel_window)
            vel_std = np.std(vel_window)
            vel_consistency = 1.0 / (1.0 + vel_std)
            confidence = 1.0 / (1.0 + uncertainty_norm[i])
            force_tendance[i] = vel_mean * vel_consistency * confidence
        
        # === SIGNAUX AVEC ANTICIPATION ===
        signal_lookback = min( self.lookback, n )
        
        for i in range( signal_lookback, n ):
            # Percentiles dynamiques
            force_window = force_tendance[ max(0, i-signal_lookback):i ]
            if len( force_window ) > 0:
                percentile_65 = np.percentile(force_window, 65)
                percentile_35 = np.percentile(force_window, 35)
            else:
                percentile_65 = percentile_35 = 0
            
            condition_haussiere = (
                velocite[i] > self.signal_buy_sell_velocity_threshold and
                acceleration_smooth[i] > -0.01 and
                position_relative[i] > 0.2 and
                force_tendance[i] > percentile_65 and
                uncertainty_norm[i] < 1.5
            )

            condition_baissiere = (
                velocite[i] < -self.signal_buy_sell_velocity_threshold and
                acceleration_smooth[i] < 0.01 and
                position_relative[i] < -0.2 and
                force_tendance[i] < percentile_35 and
                uncertainty_norm[i] < 1.5 
            )
            
            if condition_haussiere:
                signaux[i] = 1
            elif condition_baissiere:
                signaux[i] = -1
            else:
                signaux[i] = 0
        
        return signaux, force_tendance, position_relative, acceleration_smooth

    # -------------------------------------------------------------------------
    
    def _evaluer_signal_actuel_avec_prevision(self, forecast_data: Dict) -> Dict:
        """
        Évalue la robustesse du signal ACTUEL en regardant les prévisions
        """
        if len(forecast_data.get('velocite', [])) < 3:
            return {'confiance_prospective': 0.5}
        
        # Analyse des 3-5 prochains points prévus
        future_vel = forecast_data['velocite'][:5]
        future_trend = forecast_data['tendance'][:5]
        
        # Cohérence de la tendance future
        vel_consistent = np.std(future_vel) < np.abs(np.mean(future_vel)) * 0.5
        trend_monotone = np.all(np.diff(future_trend) > 0) or np.all(np.diff(future_trend) < 0)
        
        # Élargissement de l'intervalle de confiance (signe d'incertitude)
        conf_widening = forecast_data['confidence_width'][-1] / forecast_data['confidence_width'][0]
        
        confiance_score = 0.0
        if vel_consistent:
            confiance_score += 0.4
        if trend_monotone:
            confiance_score += 0.3
        if conf_widening < 1.5:  # IC ne s'élargit pas trop
            confiance_score += 0.3
        
        return {
            'confiance_prospective': confiance_score,
            'tendance_future': 'haussiere' if np.mean(future_vel) > 0 else 'baissiere',
            'coherence': vel_consistent and trend_monotone
        }
    
    def calculer_signaux_achat_vente_adaptatifs(
        self,
        prix: np.ndarray,
        tendance: np.ndarray,
        velocite: np.ndarray,
        cycle: np.ndarray,
        variance: np.ndarray,
        forecast_data: Dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcule les signaux avec des seuils adaptatifs basés sur :
        - La volatilité historique
        - L'incertitude du filtre de Kalman
        - La cohérence des prévisions (pour les derniers points seulement)
        """
        n = len(prix)
        signaux = np.zeros(n, dtype=np.int8)
        force_tendance = np.zeros(n, dtype=np.float64)
        
        # Calculs standard
        acceleration = np.gradient(velocite)
        kernel = np.ones(5) / 5
        acceleration_smooth = np.convolve(
            np.pad(acceleration, (2, 2), mode='edge'),
            kernel,
            mode='valid'
        )
        
        position_relative = np.where(tendance != 0, cycle / tendance * 100, 0)
        uncertainty = np.sqrt(variance)
        uncertainty_norm = uncertainty / np.mean(uncertainty) if np.mean(uncertainty) > 0 else np.ones_like(uncertainty)
        
        # Force de tendance (inchangé)
        window_size = min(self.window_trend_size, WINDOW_TREND_SIZE_MAX)
        for i in range(window_size, n):
            start_idx = i - window_size
            vel_window = velocite[start_idx:i]
            vel_mean = np.mean(vel_window)
            vel_std = np.std(vel_window)
            vel_consistency = 1.0 / (1.0 + vel_std)
            confidence = 1.0 / (1.0 + uncertainty_norm[i])
            force_tendance[i] = vel_mean * vel_consistency * confidence
        
        # === CALCUL DES SEUILS ADAPTATIFS ===
        signal_lookback = min(self.lookback, n)
        
        # Seuil de vélocité adaptatif basé sur la volatilité récente
        recent_vol = np.std( velocite[max(0, n-signal_lookback):n])
        vel_threshold_adaptive = max(0.02, min(0.05, recent_vol * 0.5))
        
        # === GÉNÉRATION DES SIGNAUX ===
        for i in range(signal_lookback, n):
            force_window = force_tendance[max(0, i-signal_lookback):i]
            if len(force_window) > 0:
                percentile_65 = np.percentile(force_window, 65)
                percentile_35 = np.percentile(force_window, 35)
            else:
                percentile_65 = percentile_35 = 0
            
            # Ajustement pour les derniers points avec prévision
            is_recent = (i >= n - 10)  # 10 derniers points
            adjustment_factor = 1.0
            
            if is_recent and forecast_data:
                eval_prospective = self._evaluer_signal_actuel_avec_prevision( forecast_data )
                # Si prévision cohérente, on peut être plus agressif
                if eval_prospective['coherence']:
                    adjustment_factor = 0.85  # Réduit les seuils de 15%
            
            # Conditions avec seuils adaptatifs
            condition_haussiere = (
                velocite[i] > vel_threshold_adaptive * adjustment_factor and
                acceleration_smooth[i] > -0.01 and
                position_relative[i] > 0.2 and
                force_tendance[i] > percentile_65 and
                uncertainty_norm[i] < 1.5
            )
            
            condition_baissiere = (
                velocite[i] < -vel_threshold_adaptive * adjustment_factor and
                acceleration_smooth[i] < 0.01 and
                position_relative[i] < -0.2 and
                force_tendance[i] < percentile_35 and
                uncertainty_norm[i] < 1.5
            )
            
            if condition_haussiere:
                signaux[i] = 1
            elif condition_baissiere:
                signaux[i] = -1
    
        return signaux, force_tendance, position_relative, acceleration_smooth
    
    # -------------------------------------------------------------------------
    
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
            window = min( self.window_support_resistance_size, WINDOW_SUPPORT_RESISTANCE_MAX )
        
        n = len( prix )
        supports = np.full( n, np.nan, dtype=np.float64 )
        resistances = np.full( n, np.nan, dtype=np.float64 )
        
        # Écart-type de l'incertitude
        std_dev = np.sqrt( variance )
        
        for i in range( window, n ):
            cycle_window = cycle[ i-window:i ]
            
            # Bandes basées sur l'incertitude de Kalman (2 sigma)
            confidence_band = 2.0 * std_dev[ i ]
            
            # Support: tendance - incertitude + cycle négatif
            cycles_negatifs = cycle_window[cycle_window < 0]
            if len(cycles_negatifs) > 0:
                support_cycle = np.percentile( cycles_negatifs, 5 )
            else:
                support_cycle = -np.std( cycle_window )
            
            supports[i] = tendance[i] + support_cycle - confidence_band
            
            # Résistance: tendance + incertitude + cycle positif
            cycles_positifs = cycle_window[cycle_window > 0]
            if len(cycles_positifs) > 0:
                resistance_cycle = np.percentile(cycles_positifs, 95)
            else:
                resistance_cycle = np.std( cycle_window )
            
            resistances[i] = tendance[i] + resistance_cycle + confidence_band
        
        return supports, resistances
    
    def calculer_indicateur_complet(
        self,
        prix: Union[List, np.ndarray],
        dates: Optional[pd.DatetimeIndex] = None,
        adaptive_variance: bool = True,
    ) -> Dict:
        
        """
        Calcule l'indicateur technique complet avec filtre de Kalman
        """
        prix = np.asarray( prix, dtype=np.float64 )
        self.length = len( prix )

        if self.lookback == 0:
            self.window_trend_size = self.length // 5
            self.window_support_resistance_size = self.length // 2
            
        if self.length < self.volatility_window:
            raise ValueError( f"Pas assez de données. Minimum: {self.volatility_window}" )
        
        if dates is None:
            dates = pd.date_range(start='2020-01-01', periods=self.length, freq='D')

        # 1. Calcul volatilité
        log_returns = np.diff( np.log( np.maximum( prix, 1e-10 ) ) )
        returns_series = pd.Series( log_returns )
        volatility_series = returns_series.rolling(
            window=self.volatility_window,
            min_periods=1
        ).std()
        
        volatility = volatility_series.bfill().values
        volatility = np.concatenate([[volatility[0]], volatility])
        
        # 2. Choix des variances Q et R
        if adaptive_variance:
            Q, R = self.calculer_variances_adaptatives( prix, volatility )
        else:
            Q = self.process_variance or np.var( np.diff( prix ) ) * 0.01
            R = self.measurement_variance or np.var( prix ) * 0.1

        # Convertir Q et R en float Python natif pour le Cyhton
        Q = float(Q)
        R = float(R)
        
        # 3. Application du filtre de Kalman
        try:
            tendance, velocite, cycle, variance, forecast_data = self.kalman_forecast(
                prix, Q, R
            )

            # Créer les dates de prévision
            if isinstance( dates, pd.Series ) and pd.api.types.is_datetime64_any_dtype( dates ):
                last_date = pd.Timestamp( dates.iloc[-1] )
                # Calculer la fréquence à partir des données
                time_diff = pd.Timestamp(dates.iloc[-1]) - pd.Timestamp(dates.iloc[-2])
                # Générer les dates de prévision
                forecast_dates = [last_date + time_diff * (i + 1) for i in range(forecast_data['n_forecast'])]
                forecast_dates = pd.DatetimeIndex(forecast_dates)
            else:
                if isinstance( dates, np.ndarray ):
                    forecast_dates = np.arange(
                        dates[-1], 
                        dates[-1] + forecast_data['n_forecast']
                    )
                if isinstance( dates, pd.Series ): 
                    forecast_dates = np.arange(
                        dates.iloc[-1], 
                        dates.iloc[-1] + forecast_data['n_forecast']
                    )
        except Exception as e:
            print( f"Erreur filtre Kalman, utilisation fallback: {e}" )
            # Fallback: moyenne mobile
            tendance = self._moyenne_mobile_centree(prix, 20)
            velocite = np.gradient( tendance )
            cycle = prix - tendance
            variance = np.ones_like(prix) * np.var(cycle)
        
        # 4. Calcul des signaux
        if self.signal_buy_sell_veloc_adapt_thresh == False:
            signaux, force_tendance, position_relative, acceleration = self.calculer_signaux_achat_vente(
                prix, tendance, velocite, cycle, variance
            )
        else:
            signaux, force_tendance, position_relative, acceleration = self.calculer_signaux_achat_vente_adaptatifs(
                prix, tendance, velocite, cycle, variance, forecast_data
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
            'volatility': volatility,
            'forecast_data': forecast_data,
            'forecast_dates': forecast_dates
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
        
        for i in range( len(positions) - 1 ):
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
            'rendement_moyen': float( np.mean(rendements_array) ),
            'rendement_total': float( np.sum(rendements_array) ),
            'rendement_std': float( np.std(rendements_array) ),
            'taux_reussite': float( np.mean(rendements_array > 0) * 100 ),
            'sharpe_ratio': float( np.mean(rendements_array) / np.std(rendements_array) ) if np.std(rendements_array) > 0 else 0.0,
            'max_drawdown': float( np.min(rendements_array) ),
            'max_gain': float( np.max(rendements_array) )
        }

    # -------------------------------------------------------------------------
    
    def visualiser_avec_prevision( self, ax1 ):
        """
        Ajoute la visualisation des prévisions sur le graphique principal.
        """
        if not self.historique or 'forecast_data' not in self.historique:
            return

        hist = self.historique
        forecast = hist['forecast_data']
        forecast_dates = hist.get('forecast_dates', [])

        # Vérifie la présence des données nécessaires
        if not forecast or 'tendance' not in forecast or len(forecast['tendance']) == 0 or len(forecast_dates) == 0:
            return

        # Ligne de prévision
        ax1.plot(
            forecast_dates,
            forecast['tendance'],
            linestyle='-',
            linewidth=1.2, 
            alpha=0.8, 
            label='Prévision Kalman', 
            color='navy'
        )

        # Intervalle de confiance
        if 'lower_bound' in forecast and 'upper_bound' in forecast:
            ax1.fill_between(
                forecast_dates,
                forecast['lower_bound'],
                forecast['upper_bound'],
                alpha=0.15, color='orange', label='IC 95%'
            )

        # Point de jonction entre historique et prévision
        _len = len( hist['dates'] )
        last_date = hist['dates'][_len-1]
        _len = len( hist['tendance'] )
        last_price = hist['tendance'][_len-1]

        first_forecast_date = forecast_dates[0]
        first_forecast_price = forecast['tendance'][0]

        ax1.plot(
            [last_date, first_forecast_date],
            [last_price, first_forecast_price],
            'r:', linewidth=1.5, alpha=0.6, color='navy'
        )
        
        _title = ax1.get_title()
        _tendance = "HAUSSIÈRE" if np.mean(forecast['velocite']) > 0 else "BAISSIÈRE"
        ax1.set_title( _title + f" - forecast: {self.n_forecast} - trend: {self.window_trend_size} - supp/resist: {self.window_support_resistance_size} -> {_tendance} : {forecast['tendance'][-1]:.2f}" )
    
    # -------------------------------------------------------------------------
    
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
        # --------------------------------------------
        #
        _lines, = ax1.plot( hist['dates'], hist['prix'], 'b-', alpha=0.7, color='green',
                label='Prix', linewidth=1)
        lines.append( _lines )
        
        ax1.plot( hist['dates'], hist['tendance'], 'r-', color='darkblue',
                linewidth=1.5, label='Tendance Kalman')
        
        # Bandes de confiance (±2 sigma)
        std_dev = np.sqrt(hist['variance'])
        upper_band = hist['tendance'] + 2 * std_dev
        lower_band = hist['tendance'] - 2 * std_dev
        
        ax1.fill_between( hist['dates'], lower_band, upper_band,
                        alpha=0.3, color='grey', label='Bande confiance (95%)' )
        
        # Support/Résistance
        mask_support = ~np.isnan(hist['supports'])
        mask_resistance = ~np.isnan(hist['resistances'])
        
        if np.any(mask_support):
            ax1.plot(
                np.array(hist['dates'])[mask_support],
                hist['supports'][mask_support],
                'red', linestyle='--', alpha=1, label='Support'
            )

        if np.any(mask_resistance):
            ax1.plot(
                np.array(hist['dates'])[mask_resistance],
                hist['resistances'][mask_resistance],
                'green', linestyle='--', alpha=1, label='Résistance'
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
        ax1.legend( loc='upper left', fontsize=10 )
        ax1.grid( True, alpha=0.8 )

        # Afficher la prédiction
        # ----------------------
        #
        if self.n_forecast > 0:
            self.visualiser_avec_prevision( ax1 )
            
        # 2. Composante Cyclique et ...
        # -----------------------------------------------
        def plot_cycle_and_rsi( ax2, hist ):
            
            # Incertitude de Kalman
            uncertainty = np.sqrt( hist['variance'] )
            _max = np.max(uncertainty)
            _min = np.min(uncertainty)
            
            """
            Affiche le cycle avec son RSI pour identifier les zones de surachat/survente
            """
            ax2_twin = ax2.twinx()
            
            # Cycle principal
            ax2.plot( hist['dates'], hist['cycle'], 'purple', linewidth=1.2, label='Cycle' )
            ax2.axhline( y=0, color='black', linestyle='-', alpha=0.5 )
            
            positive_mask = hist['cycle'] > 0
            negative_mask = hist['cycle'] < 0
            
            ax2.fill_between( hist['dates'], hist['cycle'], 0,
                            where=positive_mask, alpha=0.3, color='green' )
            ax2.fill_between( hist['dates'], hist['cycle'], 0,
                            where=negative_mask, alpha=0.3, color='red' )
            
            # Calcul du RSI du cycle
            window = 14
            cycle_diff = np.diff( hist['cycle'], prepend=hist['cycle'][0] )
            gains = np.where(cycle_diff > 0, cycle_diff, 0)
            losses = np.where(cycle_diff < 0, -cycle_diff, 0)
            
            avg_gains = pd.Series(gains).rolling( window=window, min_periods=1).mean()
            avg_losses = pd.Series(losses).rolling( window=window, min_periods=1 ).mean()
            
            rs = avg_gains / (avg_losses + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            
            # RSI sur l'axe secondaire
            ax2_twin.plot( hist['dates'], rsi, 'darkorange', linewidth=1, label='RSI Cycle' )
            ax2_twin.axhline( y=70, color='red', linestyle='--', alpha=0.4, label='Surachat' )
            ax2_twin.axhline( y=30, color='green', linestyle='--', alpha=0.4, label='Survente' )
            #ax2_twin.fill_between( hist['dates'], 30, 70, alpha=0.1, color='gray' )
            
            # Mise en forme
            ax2.set_title(f'Composante Cyclique et RSI (Momentum du cycle) Incertitude Kalman σ: {_max:.5f} / {_min:.5f}')
            ax2.set_ylabel('Cycle', color='purple')
            ax2_twin.set_ylabel('RSI Cycle', color='darkorange')
            ax2_twin.set_ylim(0, 100)
            
            ax2.legend( loc='upper left', fontsize=10 )
            ax2_twin.legend( loc='lower left', fontsize=10 )
            ax2.grid(True, alpha=0.9)

        def plot_cycle_percentiles( ax2, hist ):
            """
            Affiche le cycle normalisé avec ses percentiles historiques
            """
            ax2_twin = ax2.twinx()
            
            # Cycle principal
            ax2.plot(hist['dates'], hist['cycle'], 'purple', linewidth=1.2, label='Cycle')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # Calcul des percentiles glissants
            window = 50  # Fenêtre pour les percentiles
            cycle_series = pd.Series(hist['cycle'])
            
            percentile_90 = cycle_series.rolling(window=window, min_periods=20).quantile(0.9)
            percentile_75 = cycle_series.rolling(window=window, min_periods=20).quantile(0.75)
            percentile_25 = cycle_series.rolling(window=window, min_periods=20).quantile(0.25)
            percentile_10 = cycle_series.rolling(window=window, min_periods=20).quantile(0.1)
            
            # Bandes de percentiles
            ax2.fill_between(hist['dates'], percentile_10, percentile_90,
                            alpha=0.1, color='blue', label='P10-P90')
            ax2.fill_between(hist['dates'], percentile_25, percentile_75,
                            alpha=0.2, color='blue', label='P25-P75')
            
            ax2.plot(hist['dates'], percentile_90, '--', color='red', alpha=0.5, linewidth=1)
            ax2.plot(hist['dates'], percentile_10, '--', color='green', alpha=0.5, linewidth=1)
            
            # Position percentile actuelle (0-100)
            cycle_position = np.zeros_like(hist['cycle'])
            for i in range(window, len(hist['cycle'])):
                window_data = hist['cycle'][max(0, i-window):i]
                cycle_position[i] = (np.sum(window_data <= hist['cycle'][i]) / len(window_data)) * 100
            
            ax2_twin.plot(hist['dates'], cycle_position, 'darkorange', 
                        linewidth=1.5, label='Position Percentile')
            ax2_twin.axhline(y=80, color='red', linestyle=':', alpha=0.7)
            ax2_twin.axhline(y=20, color='green', linestyle=':', alpha=0.7)
            
            # Mise en forme
            ax2.set_title('Cycle avec Percentiles Historiques')
            ax2.set_ylabel('Cycle', color='purple')
            ax2_twin.set_ylabel('Position Percentile (%)', color='darkorange')
            ax2_twin.set_ylim(0, 100)
            ax2.legend(loc='upper left', fontsize=10)
            ax2_twin.legend(loc='upper right', fontsize=10)
            ax2.grid(True, alpha=0.9)
            
        plot_cycle_and_rsi( ax2, hist )
        #plot_cycle_percentiles( ax2, hist )
        
        # 3. Vélocité et accélération
        # ---------------------------
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

        # Alignement des zéros
        # --------------------
        # Récupérer les limites actuelles
        vel_min, vel_max = ax3.get_ylim()
        acc_min, acc_max = ax3_twin.get_ylim()

        # Calculer les limites symétriques pour centrer sur zéro
        vel_limit = max(abs(vel_min), abs(vel_max))
        acc_limit = max(abs(acc_min), abs(acc_max))

        # Appliquer les limites symétriques
        ax3.set_ylim(-vel_limit, vel_limit)
        ax3_twin.set_ylim(-acc_limit, acc_limit)

        _v_max = np.max( self.historique['velocite'] )
        _v_min = np.min( self.historique['velocite'] )
        _v_mean = np.mean( self.historique['velocite'] )
        _title = f"Vélocité max: {_v_max:.4f} min: {_v_min:.4f} moy: {_v_mean:.4f} et Accélération de la Tendance"
        ax3.set_title( _title )
        ax3.set_ylabel('Vélocité', color='darkblue')
        ax3_twin.set_ylabel('Accélération', color='darkorange')
        ax3.legend(loc='upper left', fontsize=10)
        ax3_twin.legend(loc='lower left', fontsize=10)
        ax3.grid(True, alpha=0.9)
        
        # 4. Volatilité et confiance
        # --------------------------
        #
        ax4_twin = ax4.twinx()
        
        volatility = hist.get('volatility', np.ones_like(hist['prix']))
        ax4.plot( hist['dates'], volatility * 100, 'darkred',
                linewidth=1, label='Volatilité (%)' )
        ax4.fill_between( hist['dates'], volatility * 100, 0,
                        alpha=0.2, color='red' )
        
        # Confiance des signaux
        confiance = np.abs( hist['force_tendance'] )
        if np.max(confiance) > np.min(confiance):
            confiance_norm = (confiance - np.min(confiance)) / (np.max(confiance) - np.min(confiance))
        else:
            confiance_norm = np.ones_like(confiance) * 0.5
            
        ax4_twin.plot( hist['dates'], confiance_norm * 100, 'darkgreen',
                     linewidth=2, label='Confiance Signal (%)')
        
        # Zones de confiance
        high_conf = confiance_norm > 0.7
        med_conf = (confiance_norm > 0.3) & (confiance_norm <= 0.7)
        low_conf = confiance_norm <= 0.3
        
        ax4_twin.fill_between( hist['dates'], 0, 100, where=high_conf,
                             alpha=0.1, color='green', label='Zone Forte' )
        ax4_twin.fill_between( hist['dates'], 0, 100, where=med_conf,
                             alpha=0.1, color='yellow', label='Zone Modérée' )
        ax4_twin.fill_between( hist['dates'], 0, 100, where=low_conf,
                             alpha=0.1, color='red', label='Zone Faible' )
        
        ax4.set_title( 'Volatilité du Marché et Confiance des Signaux' )
        ax4.set_ylabel( 'Volatilité (%)', color='darkred' )
        ax4_twin.set_ylabel( 'Confiance (%)', color='darkgreen' )
        ax4.legend( loc='upper left', fontsize=10 )
        ax4_twin.legend( loc='lower left', fontsize=10 )
        ax4.grid( True, alpha=0.9 )

        # Gestion des ticks
        if __name__ == "__main__":
            ax1.set_xticks( tick_positions )
            ax1.set_xticklabels( tick_labels )
        else:
            ax4.set_xticks( tick_positions )
            ax4.set_xticklabels( tick_labels )
                       
        # plt.setp( ax1.get_xticklabels(), visible=True )
        # plt.setp( ax2.get_xticklabels(), visible=False )
        # plt.setp( ax3.get_xticklabels(), visible=False )
        # plt.setp( ax4.get_xticklabels(), visible=False )

        #plt.tight_layout()

        if __name__ == "__main__":
            plt.show()
        
        self._afficher_rapport()


    def _afficher_previsions(self):
        if 'forecast_data' not in self.historique:
            return
        
        forecast = self.historique['forecast_data']
        print( "\n" + "="*40)
        print(f"PRÉVISIONS À COURT TERME ({forecast['n_forecast']} points):")
        print( "="*40)
        print(f"Tendance prévue: {forecast['tendance'][-1]:.2f}")
        print(f"Vélocité moyenne: {np.mean(forecast['velocite']):.4f}")
        print(f"Intervalle confiance: [{forecast['lower_bound'][-1]:.2f}, {forecast['upper_bound'][-1]:.2f}]")
        print(f"Largeur IC finale: {forecast['confidence_width'][-1]:.2f}")
        
        direction = "HAUSSIÈRE" if np.mean(forecast['velocite']) > 0 else "BAISSIÈRE"
        print(f"→ Direction anticipée: {direction}")
            
    def _afficher_rapport( self ) -> None:
        """Affichage du rapport de performance"""
        print( "\n" + "="*40)
        print( "FILTRE DE KALMAN - MÉTRIQUES QUALITÉ" )
        print( "="*40)
        
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
                print( "\n" + "="*25)
                print( f"PERFORMANCE ACHAT/VENTE :" )
                print( "="*25)
                print( f"Nombre de Signaux: {perf['nb_signaux']}" )
                print( f"Rendement moyen: {perf['rendement_moyen']:.2f} %" )
                print( f"Rendement total: {perf['rendement_total']:.2f} %" )
                print( f"Drawdown max: {perf['max_drawdown']:.2f} %" )
                print( f"Gain max: {perf['max_gain']:.2f} %" )
                print( f"Taux réussite: {perf['taux_reussite']:.1f} %" )
                print( f"Sharpe: {perf['sharpe_ratio']:.2f}" )
            else:
                print( f"\n{perf['message']}" )
        except Exception as e:
            print( f"\nErreur calcul performance: {e}" )
            
        self._afficher_stats_velocite()
        
        # Afficher la prévision
        if self.n_forecast > 0:
            self._afficher_previsions()

    def _afficher_stats_velocite( self ) -> None:
        print( "\n" + "="*30)
        print( "STATISTIQUES SUR LA VÉLOCITÉ :" )
        print( "="*30)
        print( f"- Vélocité moyenne: {np.mean(self.historique['velocite']):.4f}" )
        print( f"- Vélocité max: {np.max(self.historique['velocite']):.4f}" )
        print( f"- Vélocité min: {np.min(self.historique['velocite']):.4f}" )
        print( f"- Incertitude moyenne: {np.mean(np.sqrt(self.historique['variance'])):.4f}" )
        
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
    data['index_continuous'] = np.arange( len(data) )
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
        # print( f"\nStatistiques Kalman:" )
        # print( f"  - Vélocité moyenne: {np.mean(resultats['velocite']):.4f}" )
        # print( f"  - Vélocité max: {np.max(resultats['velocite']):.4f}" )
        # print( f"  - Vélocité min: {np.min(resultats['velocite']):.4f}" )
        # print( f"  - Incertitude moyenne: {np.mean(np.sqrt(resultats['variance'])):.4f}" )
        
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

        lines = []
        indicateur.visualiser_indicateur(
            ax1, ax2, ax3, ax4,
            tick_positions=tick_positions,
            tick_labels=tick_labels,
            lines=lines
        )
        
        return indicateur, resultats
        
    except Exception as e:
        print( f"✗ Erreur: {e}" )
        import traceback
        traceback.print_exc()
        raise


def comparer_dernier_signal( indicateur: IndicateurTendanceKalman ) -> None:
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
    comparer_dernier_signal( indicateur )
    
    print( "\n" + "="*60 )
    print( "Test terminé avec succès!" )
    print( "="*60 )