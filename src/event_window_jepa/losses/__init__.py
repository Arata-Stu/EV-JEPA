from event_window_jepa.losses.latent_prediction import (
    BalancedLatentPredictionOutput,
    balanced_event_support_latent_prediction_loss,
    latent_prediction_loss,
)
from event_window_jepa.losses.sigreg import (
    ProjectedSIGReg,
    SIGRegOutput,
    SIGRegProjector,
    SlicedEppsPulleySIGReg,
)
from event_window_jepa.losses.variance_regularization import (
    covariance_regularization,
    variance_regularization,
)

__all__ = [
    "BalancedLatentPredictionOutput",
    "ProjectedSIGReg",
    "SIGRegOutput",
    "SIGRegProjector",
    "SlicedEppsPulleySIGReg",
    "balanced_event_support_latent_prediction_loss",
    "covariance_regularization",
    "latent_prediction_loss",
    "variance_regularization",
]
