"""
Runnable script with hydra capabilities
"""

import os
import pickle
import random
import sys
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import open_dict

from gflownet.utils.common import gflownet_from_config


@hydra.main(config_path="./config", config_name="train", version_base="1.1")
def main(config):

    # Set and print working and logging directory
    with open_dict(config):
        config.logger.logdir.path = (
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        )
    print(f"\nWorking directory of this run: {os.getcwd()}")
    print(f"Logging directory of this run: {config.logger.logdir.path}\n")

    # Reset seed for job-name generation in multirun jobs
    random.seed(None)
    # Set other random seeds
    set_seeds(config.seed)

    # Initialize a GFlowNet agent from the configuration file
    gflownet = gflownet_from_config(config)

    save_path = f"env_{gflownet.env.__class__.__name__}_proxy_{gflownet.proxy.__class__.__name__}_gfn"

    # Train GFlowNet
    gflownet.train()

    

    # Sample from trained GFlowNet
    # TODO: move to method in GFlowNet agent, like sample_and_log()
    if config.n_samples > 0 and config.n_samples <= 1e5:
        batch, times = gflownet.sample_batch(n_forward=config.n_samples, train=False)
        x_sampled = batch.get_terminating_states(proxy=True)
        energies = gflownet.proxy(x_sampled)
        x_sampled = batch.get_terminating_states()
        df = pd.DataFrame(
            {
                "readable": [gflownet.env.state2readable(x) for x in x_sampled],
                "energies": energies.tolist(),
            }
        )
        samples_dir = Path("./samples/")
        samples_dir.mkdir(parents=True, exist_ok=True)
        csv_path = samples_dir /  (save_path + ".csv")
        df.to_csv(csv_path)
        dct = {"x": x_sampled, "energy": energies}
        pkl_path = samples_dir /  (save_path + ".pkl")
        pickle.dump(dct, open(pkl_path, "wb"))

    # Print replay buffer
    if len(gflownet.buffer.replay) > 0:
        print("\nReplay buffer:")
        print(gflownet.buffer.replay)

    # Close logger
    # TODO: make it gflownet.end() - perhaps there are other things to end
    gflownet.logger.end()


def set_seeds(seed):
    import numpy as np
    import torch

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()
    sys.exit()
