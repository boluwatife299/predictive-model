"""Model Zoo registry — maps string keys to model classes."""
from models.gbm              import GBMModel
from models.monte_carlo      import MonteCarloModel
from models.arima_model      import ARIMAModel
from models.garch_model      import GARCHModel
from models.ou_model         import OUModel
from models.jump_diffusion   import JumpDiffusionModel
from models.heston           import HestonModel
from models.linear_regression import LinearRegressionModel
from models.xgboost_model    import XGBoostModel
from models.prophet_model    import ProphetModel
from models.lstm_model       import LSTMModel

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
    "prophet":           ProphetModel,
    "lstm":              LSTMModel,
}
