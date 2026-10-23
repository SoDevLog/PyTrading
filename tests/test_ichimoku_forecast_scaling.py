"""Le modèle de prévision doit être appliqué avec l'échelle de son entraînement.

`train_predictive_model` normalise ses features avec un `StandardScaler`, mais
`apply_model` prédit sur les features brutes : le modèle est utilisé hors de
son domaine et ne sort pratiquement plus qu'une seule classe. Constaté sur des
données de synthèse : aucune prédiction d'achat (+1), alors que la classe
existe dans les données.

Ces tests vérifient le comportement visible : la ligne « Forcasted Signal »
doit encore pouvoir annoncer un achat.
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

INTERNAL = Path(__file__).resolve().parents[1] / "TradingInPython" / "_internal"
if str(INTERNAL) not in sys.path:
    sys.path.insert(0, str(INTERNAL))

from digitsignalprocessing import ichimoku_kinko_hyo  # noqa: E402


FEATURES = [
    "Tenkan_sen", "Kijun_sen", "Senkou_span_A", "Senkou_span_B",
    "ATR", "Kijun_sen_upper", "Kijun_sen_lower",
]


def _signalled_data(n=300, seed=1):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2024-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.013, n)))
    data = pd.DataFrame({
        "Open": close,
        "High": close * (1 + np.abs(rng.normal(0, 0.006, n))),
        "Low": close * (1 - np.abs(rng.normal(0, 0.006, n))),
        "Close": close,
        "Adj Close": close,
    }, index=index)
    data = ichimoku_kinko_hyo.ichimoku_modernise(data, 9, 26, 52)
    return ichimoku_kinko_hyo.generate_signals(data)


def _train(data):
    with contextlib.redirect_stdout(io.StringIO()):
        return ichimoku_kinko_hyo.train_predictive_model(data)


def _predicted_classes(data, forcasted):
    mean = data["Close"].mean()
    return set((forcasted.dropna() - mean).round().astype(int))


def test_apply_model_can_still_predict_a_buy():
    data = _signalled_data()
    model = _train(data)

    out = ichimoku_kinko_hyo.apply_model(model, data.copy())

    classes = _predicted_classes(data, out["Signal_Forcasted"])
    assert 1 in classes, (
        "le modèle appliqué ne prédit jamais d'achat : il est utilisé avec des "
        f"features non normalisées (classes observées : {sorted(classes)})"
    )


def test_apply_model_matches_predictions_on_normalised_features():
    data = _signalled_data()
    model = _train(data)

    out = ichimoku_kinko_hyo.apply_model(model, data.copy())

    features = data[FEATURES].dropna()
    scaled = pd.DataFrame(model.scaler_.transform(features),
                          columns=features.columns, index=features.index)
    expected = model.predict(scaled)
    got = (out.loc[features.index, "Signal_Forcasted"]
           - data["Close"].mean()).to_numpy()

    assert np.array_equal(got, expected), (
        "apply_model ne prédit pas sur les mêmes features que l'entraînement"
    )
