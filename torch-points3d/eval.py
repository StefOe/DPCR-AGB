import hydra
import numpy as np
import torch.random
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, open_dict

from torch_points3d.trainer import Trainer


@hydra.main(config_path="conf", config_name="eval")
def main(cfg):
    rs = cfg.get("random_seed", 21)

    # disable random shuffling and dropping of last batch in training loader
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg["shuffle"] = False
        cfg["drop_last"] = False

    np.random.default_rng(rs)
    torch.random.manual_seed(rs)

    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    if cfg.pretty_print:
        print(OmegaConf.to_yaml(cfg))
    
    trainer = Trainer(cfg)
    eval_stages = cfg.get("eval_stages", [""])
    for stage in eval_stages:
        trainer.eval(stage)
    #
    # # https://github.com/facebookresearch/hydra/issues/440
    GlobalHydra.get_state().clear()
    return 0


if __name__ == "__main__":
    main()
