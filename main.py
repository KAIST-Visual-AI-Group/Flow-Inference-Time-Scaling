import os
import argparse
from time import time
from datetime import datetime
import math 
from dataclasses import dataclass

from omegaconf import OmegaConf

import rbf.shared_modules as sm
sm.OFF_LOG = False
sm.DO_NOT_SAVE_INTERMEDIATE_IMAGES = False

from rbf.utils.config_utils import load_config
from rbf.general_trainer import GeneralTrainer
from rbf.rewind_trainer import RewindTrainer
from rbf.rbf import RBF

# Differentiable reward
# from rbf.dps_trainer import DPSTrainer
# from rbf.rbf_dps import RBFDPS
# from rbf.svdd_dps import SVDDDPS

from rbf.utils.extra_utils import ignore_kwargs
from rbf.utils.print_utils import print_with_box, print_info, print_note
from rbf.utils.random_utils import seed_everything

GENERAL_METHODS = ["bon", "code", "smc", "svdd"]
REWIND_METHODS = ["sop"]

@ignore_kwargs
@dataclass
class Config:
    root_dir: str = "./results/default"
    save_source: bool = False
    seed: int = 1
    tag: str = ""
    save_now: bool = False

    max_nfe: int = 500
    n_particles: int = 1
    batch_size: int = 1
    max_steps: int = 30
    tau_norm: float = 0.0
    filtering_method: str = None

    ode_step: int = 4
    forward_step: float = 780.0
    backward_step: float = 810.0
    ode_seed_t: float = 120.0
    corrector: str = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    parser.add_argument(
        "-t",
        "--trainer_type",
        default="general",
        choices=["general", "rewind"],
        help="type of trainer to use",
    )
    args, extras = parser.parse_known_args()
    cfg = load_config(args.config, cli_args=extras)
    
    now = datetime.now()
    strnow = now.strftime("%Y%m%d_%H%M%S")
    
    if cfg.save_now:
        cfg.tag = cfg.tag + "_" + strnow
    cfg.root_dir = os.path.join(cfg.root_dir.replace(" ", "_"), cfg.tag)


    print_with_box(
        f"Config loaded from {args.config} with the following content:\n{OmegaConf.to_yaml(cfg)}",
        title="Config",
        max_len=88,
    ) if not sm.OFF_LOG else None
    main_cfg = Config(**cfg)
    

    if main_cfg.filtering_method == "sop":
        assert main_cfg.backward_step > main_cfg.forward_step, f"backward_step: {main_cfg.backward_step} <= forward_step: {main_cfg.forward_step}"
        estimated_nfe = (main_cfg.ode_step * main_cfg.batch_size * main_cfg.n_particles) * (math.ceil(main_cfg.ode_seed_t / (main_cfg.backward_step - main_cfg.forward_step)) + 1)
        main_cfg.max_steps = int(main_cfg.ode_seed_t / (main_cfg.backward_step - main_cfg.forward_step) + 1)

    else:
        if main_cfg.tau_norm is not None and main_cfg.tau_norm != 0:
            estimated_nfe = (main_cfg.n_particles * main_cfg.batch_size) \
                * (2 * main_cfg.max_steps)

        else:
            estimated_nfe = (main_cfg.n_particles * main_cfg.batch_size) \
                * (main_cfg.max_steps)

    print_note("Estimated NFE: ", estimated_nfe) if not sm.OFF_LOG else None

    if estimated_nfe > cfg.max_nfe:
        print(f"Estimated NFE: {estimated_nfe} > Max NFE: {cfg.max_nfe}") if not sm.OFF_LOG else None
        raise ValueError("Estimated NFE is greater than max NFE")

    
    seed_everything(main_cfg.seed)
    print_info("Seed set to ", main_cfg.seed) if not sm.OFF_LOG else None

    # save the config to a file
    os.makedirs(main_cfg.root_dir, exist_ok=True)
    print(os.path.join(main_cfg.root_dir, "config.yaml")) if not sm.OFF_LOG else None
    with open(os.path.join(main_cfg.root_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    print_info("filtering_method", main_cfg.filtering_method) if not sm.OFF_LOG else None


    if main_cfg.filtering_method.lower() in GENERAL_METHODS:
        trainer = GeneralTrainer(cfg)

    elif main_cfg.filtering_method.lower() in REWIND_METHODS:
        trainer = RewindTrainer(cfg)

    elif main_cfg.filtering_method == "rbf":
        trainer = RBF(cfg)

    # elif main_cfg.filtering_method == "dps":
    #     trainer = DPSTrainer(cfg)

    # elif main_cfg.filtering_method == "rbf_dps":
    #     trainer = RBFDPS(cfg)

    # elif main_cfg.filtering_method == "svdd_dps":
    #     trainer = SVDDDPS(cfg)
        
    else:
        raise ValueError(f"Unknown trainer type: {args.trainer}")

    start_time = time()
    output_filename = trainer.train()
    collapse_time = time() - start_time
    print_with_box(
        (
            f"Training finished in {collapse_time:.2f} seconds.\n"
            f"Output saved to {output_filename}"
        ),
        title="Results",
    ) if not sm.OFF_LOG else None


if __name__ == "__main__":
    main()
