from event_window_jepa.losses.cmax import (
    CMaxOutput,
    TamingCMaxLoss,
    average_timestamp_image,
    bilinear_splat_iwe,
    charbonnier_spatial_smoothness,
    sample_patch_flow_at_events,
    taming_focus_loss,
    warp_events_to_reference,
)
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
    "CMaxOutput",
    "ProjectedSIGReg",
    "SIGRegOutput",
    "SIGRegProjector",
    "SlicedEppsPulleySIGReg",
    "TamingCMaxLoss",
    "average_timestamp_image",
    "balanced_event_support_latent_prediction_loss",
    "bilinear_splat_iwe",
    "charbonnier_spatial_smoothness",
    "covariance_regularization",
    "latent_prediction_loss",
    "sample_patch_flow_at_events",
    "taming_focus_loss",
    "variance_regularization",
    "warp_events_to_reference",
]
