from .base import DDIMCorrector
from .rgrp_sampler import RGRPSampler
from .adaptive_sampler import AdaptiveSampler
from .dps import DiffRewardCorrector

CORRECTORs = {
    "ddim": DDIMCorrector,
    "particle": RGRPSampler,
    "adaptive": AdaptiveSampler,
    "diff": DiffRewardCorrector,
}

CORRECTOR_REQUIRING_GRADIENT = ["diff"]