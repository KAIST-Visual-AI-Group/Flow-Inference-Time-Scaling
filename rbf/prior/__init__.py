from .sd import (
    StableDiffusionPrior,
)

from .flux import FluxPrior
from .instaflow import InstaFlowPrior
from .flux_fill import FluxFillPrior

from .sd2 import SD2Prior
PRIORs = {
    "sd": StableDiffusionPrior,
    "flux": FluxPrior,
    "flux_fill": FluxFillPrior,
    "instaflow": InstaFlowPrior,
    "sd2": SD2Prior,
}
