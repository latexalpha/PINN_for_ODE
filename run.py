"""
@author: Shangyu ZHAO
"""

import os
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from forward_solution import system_training as forward_runner
from inverse_identification import system_training as inverse_runner


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./config", config_name="config.yaml")
def run(cfg: DictConfig) -> None:
    print(f"Working directory : {os.getcwd()}")
    output_dir = HydraConfig.get().runtime.output_dir
    print(f"Output directory  : {output_dir}")
    # Set up logging
    logger.info("INFO LEVEL MESSAGE: \n")
    logger.info(f"CONFIG: \n{OmegaConf.to_yaml(cfg)}")
    # training
    if cfg.training.type == "pinn":
        forward_runner(cfg, output_dir, logger)
    elif cfg.training.type == "ipinn":
        inverse_runner(cfg, output_dir, logger)


if __name__ == "__main__":
    run()
