from typing import Any, Optional
from omegaconf import OmegaConf, DictConfig


def load_config(*yamls: str, cli_args: Optional[list] = None, from_string=False, **kwargs) -> Any:
    if from_string:
        yaml_confs = [OmegaConf.create(s) for s in yamls]
    else:
        yaml_confs = [OmegaConf.load(f) for f in yamls]
    cli_conf = OmegaConf.from_cli(cli_args)
    cfg = OmegaConf.merge(*yaml_confs, cli_conf, kwargs)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)

    return cfg

def fetch_config(self, cfg):
    """
    Fetch dataclass variables to local variables
    self: any class object
    cfg: any dataclass object
    """

    for key, value in cfg.items():
        setattr(self, key, value)
    return self
