"""Model Zoo registry — maps string keys to model classes."""
from models.gbm               import GBMModel
from models.monte_carlo       import MonteCarloModel
from models.arima_model       import ARIMAModel
from models.garch_model       import GARCHModel
from models.ou_model          import OUModel
from models.jump_diffusion    import JumpDiffusionModel
from models.heston            import HestonModel
from models.linear_regression import LinearRegressionModel
from models.xgboost_model     import XGBoostModel

# Optional heavy models — gracefully excluded if deps aren't installed
try:
    from models.prophet_model import ProphetModel
    _PROPHET_OK = True
except Exception:
    ProphetModel = None  # type: ignore[assignment,misc]
    _PROPHET_OK = False

try:
    from models.lstm_model import LSTMModel
    _LSTM_OK = True
except Exception:
    LSTMModel = None  # type: ignore[assignment,misc]
    _LSTM_OK = False

REGISTRY: dict = {
    "gbm":               GBMModel,
    "monte_carlo":       MonteCarloModel,
    "arima":             ARIMAModel,
    "garch":             GARCHModel,
    "ou":                OUModel,
    "jump_diffusion":    JumpDiffusionModel,
    "heston":            HestonModel,
    "linear_regression": LinearRegressionModel,
    "xgboost":           XGBoostModel,
}

if _PROPHET_OK:
    REGISTRY["prophet"] = ProphetModel
if _LSTM_OK:
    REGISTRY["lstm"] = LSTMModel

# Which models are unavailable in this environment
UNAVAILABLE: set[str] = set()
if not _PROPHET_OK:
    UNAVAILABLE.add("prophet")
if not _LSTM_OK:
    UNAVAILABLE.add("lstm")
