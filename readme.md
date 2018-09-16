still under construction...

## Prepare dataset

### 1. s3dis 

1) build [PointUtil](https://github.com/liuyuan-pal/PointUtil) and `ln -s /path/to/libPointUtil.so data/`

2) `python dataset/s3dis_util --raw_dir=path/to/raw/s3dis_dataset --pkl_dir=path/to/output/intermediate/pkl/files --dataset_dir=path/to/output/prepared/dataset/files`

