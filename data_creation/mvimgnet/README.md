# MVImgNet Dataset Preparation

Here is a summary of what these scripts do, these scripts should be run sequentially:
1. sample_dataset.py: sample shapes from shapenet datasets for the training, val, and test sets.
2. prepare_mvimgnet_data.py: Extracts the camera viewpoint parameters, get the pixel coordinates for the sparse points in each captured image. 
3. convert_to_spair_format.py: This script converts the dataset to the SPair71K dataset so that we can use it in the telling left to right code without major modifications in the data loading side in their code.

## 1. sample_dataset.py
The default values for the arguments are already defined, feel free to modify them whenever necessary. To run the code please run the following command:

```bash
cd data_creation/mvimgnet
python sample_dataset.py --save_dir DATASET_SAVE_DIR --keypointnet_dir KEYPOINT_DIR --max_shapes_per_split 160
```
```DATASET_SAVE_DIR``` should be the output folder that will contain the renderings, and all other exported dat.
```KEYPOINT_DIR``` should be the path of the downloaded KeypointNetDataset. 

## 2. prepare_mvimgnet_data.py
This code provides functionality to process sparse point clouds and filter out points that belong to a dense object point cloud within a specified distance threshold.
```bash
cd data_creation/mvimgnet
python prepare_mvimgnet_data.py --mvimgnet_dir /path/to/dataset --pc_dir /path/to/pointclouds --distance_threshold 0.02 --overwrite
--dataset_dir: Path to the directory containing the dataset.
--pc_dir: Path to the directory containing the point cloud data.
--distance_threshold: Distance threshold for filtering point clouds.
--overwrite: Optional flag to overwrite existing data.
```
Note that the save_dir should be the same for the sample_dataset.py script. 

Here is an example:
```bash
python main.py --mvimgnet_dir /data/MVImgNet/data/MVImgNet_by_categories/ --pc_dir /data/MVImgNet/mvpnet/MVPNet/ --distance_threshold 0.02 --overwrite
```
This command will process the dataset located at /data/MVImgNet/data/MVImgNet_by_categories/, using the point cloud data from /data/MVImgNet/mvpnet/MVPNet/, with a distance threshold of 0.02, and overwrite existing data.

## 3. convert_to_spair_format.py
```bash
cd data_creation/mvimgnet
python convert_to_spair_format.py --save_dir DATASET_SAVE_DIR --mvimgnet_dir MVIMGNET_DIR
```
Note that the save_dir should be the same for the sample_dataset.py and render_dataset.py scripts.


