from mmengine.registry import Registry
from omegaconf import OmegaConf

MODELS = Registry('models')


def build_model(config, class_names):
    model = MODELS.build(OmegaConf.to_container(config, resolve=True),  # type: ignore[arg-type]
                         default_args={'class_names': class_names})
    return model
