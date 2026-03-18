import os
import json
import numpy as np
import pandas as pd


def sample_split_shapes(dataframe, n_shapes_per_cat, random_seed):
    # Group by 'mesh_class' and sample n entries from each group.
    # Explicit loop is compatible with pandas 3.x (groupby key no longer passed to apply).
    groups = [
        group.sample(min(len(group), n_shapes_per_cat), random_state=random_seed)
        for _, group in dataframe.groupby('mesh_class')
    ]
    return pd.concat(groups).reset_index(drop=True)

def sample_shapes(args, save=False):
    ret = {}
    
    for split in args.splits:
        ret[split] = []
        
        split_shapes_df = pd.read_csv(os.path.join(args.keypointnet_dir, 'splits', f'{split}.txt'), sep='-',
                                      names=('mesh_class', 'mesh_id'), dtype=str)
            
        n_cats = len(list(split_shapes_df.mesh_class.unique()))
        n_shapes_per_cat = args.max_shapes_per_split // n_cats
        
        ret[split] = sample_split_shapes(split_shapes_df, n_shapes_per_cat, args.seed)
        
    if save:
        os.makedirs(args.save_dir, exist_ok=True)  
        for split, sampled_shapes_df in ret.items():
            save_path = os.path.join(args.save_dir, f'{split}_shapes.csv')
            sampled_shapes_df.to_csv(save_path, index=False)
    
    return ret


