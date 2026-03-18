from .backprojection import compute_kp_dists_features as compute_kp_dists_features, features_from_views as features_from_views
from .model_wrappers import CLIPWrapper as CLIPWrapper, DINOWrapper as DINOWrapper, EffNetWrapper as EffNetWrapper, SAMWrapper as SAMWrapper

models = {
    "clip": CLIPWrapper,
    "dino": DINOWrapper,
    "effnet": EffNetWrapper,
    "sam": SAMWrapper
}
