from .base import LinearAnnealingTimeSampler, FluxTimeSampler, SDTimeSampler

TIME_SAMPLERs = {
    "linear_annealing": LinearAnnealingTimeSampler,
    "flux_scheduler": FluxTimeSampler,
    "sd_scheduler": SDTimeSampler
}