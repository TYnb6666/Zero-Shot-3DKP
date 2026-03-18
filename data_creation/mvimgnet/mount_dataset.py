import argparse
import os
import subprocess
from collections import UserList
from pathlib import Path

class MergedPath(UserList):
    def __init__(self, paths):
        super().__init__(paths)

    def joinpath(self, *other):
        mvi_dir, = (mvi_path for a in self if (mvi_path := a.joinpath(*other)).exists())
        return mvi_dir

def parse_arguments():
    parser = argparse.ArgumentParser(description='Mount the MVImgNet dataset.')
    parser.add_argument('--mvimgnet_dir', type=str, required=True,
                        help='Directory where the MVImgNet dataset is stored')
    return parser.parse_args()


def main(args):
    mvi_zips = sorted(Path(args.mvimgnet_dir).glob('mvi_*.zip'))
    overlay2 = Path(args.mvimgnet_dir, 'overlay2')
    overlay2.mkdir(exist_ok=True)
    for mvi_zip in mvi_zips:
        mount_point = overlay2 / mvi_zip.name
        mount_point.mkdir(exist_ok=True)
        subprocess.run(('mount-zip', os.fspath(mvi_zip), os.fspath(mount_point)), check=True)


if __name__ == '__main__':
    main(parse_arguments())