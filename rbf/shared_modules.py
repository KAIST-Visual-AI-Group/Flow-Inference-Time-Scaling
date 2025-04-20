dataset = None
background = None
model = None
prior = None
logger = None
time_sampler = None
noise_sampler = None
corrector = None

OFF_LOG = False
DO_NOT_SAVE_INTERMEDIATE_IMAGES = False

def assert_initialized():
    assert (
        dataset is not None
        and background is not None
        and model is not None
        and prior is not None
        and logger is not None
        and time_sampler is not None
        and noise_sampler is not None
    ), "Please initialize the shared modules before using them."