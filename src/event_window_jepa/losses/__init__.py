from event_window_jepa.losses.latent_prediction import latent_prediction_loss
from event_window_jepa.losses.variance_regularization import (
    covariance_regularization,
    variance_regularization,
)

__all__ = ["covariance_regularization", "latent_prediction_loss", "variance_regularization"]
