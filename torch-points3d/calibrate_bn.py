import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from torch_points3d.trainer import Trainer


@hydra.main(config_path="conf", config_name="calibrate_bn")
def main(cfg):
    if cfg.pretty_print:
        print(OmegaConf.to_yaml(cfg))

    epochs = cfg["epochs"]

    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    trainer = Trainer(cfg)
    trainer.iterate_epochs(epochs)

    # # https://github.com/facebookresearch/hydra/issues/440
    GlobalHydra.get_state().clear()
    return 0


if __name__ == "__main__":
    main()
