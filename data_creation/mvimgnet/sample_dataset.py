import os
import json
import argparse
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description='Sampling shapes from keypointnet dataset.')
    parser.add_argument('--max_shapes_per_split', type=int, default=1000,
                        help='Number of pairs per shape (default: 1000)')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save the dataset. should be the same for the subsequent scripts')
    parser.add_argument('--mvimgnet_dir', type=str, required=True,
                        help='Directory where the MVImgNet dataset is stored')
    parser.add_argument('--splits_dir', type=str, required=True,
                        help='official splits dir')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'],
                        help='List of dataset splits (default: ["train", "val"])')
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed for reproducibility (default: 2024)')
    return parser.parse_args()


def sample_shapes(args, random_state, save=False):
    ret = {}
    
    for split in args.splits:
        ret[split] = []
        
        with open(os.path.join(args.splits_dir, f'{split}.json')) as fin:
            split_shapes = json.load(fin)
        
        sampled_split_shapes = list(random_state.choice(split_shapes, min(len(split_shapes), args.max_shapes_per_split), replace=False))
        sampled_split_shapes = [el.replace('/', '__') for el in sampled_split_shapes]
        ret[split] = sampled_split_shapes
        
    if save:
        os.makedirs(args.save_dir, exist_ok=True)  
        for split, sampled_split_shapes in ret.items():
            save_path = os.path.join(args.save_dir, f'{split}_shapes.json')
            with open(save_path, 'w') as fout:
                json.dump(sampled_split_shapes, fout)
    
    return ret


if __name__ == '__main__':
    # '/datawaha/cggroup/abdeas0a/datasets/MVImgNet/data/MVImgNet_by_categories/'
    # '/datawaha/cggroup/abdeas0a/datasets/MVImgNet/MVPNet/split/val.json'
    # Parse arguments
    args = parse_arguments()

    # Read the training and the validation shapes
    random_state = np.random.RandomState(args.seed)
    sampled_shapes_dict = sample_shapes(args, random_state, save=True)