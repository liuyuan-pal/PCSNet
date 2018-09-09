#!/usr/bin/env bash
python train_s3dis.py --restore=1 --restore_epoch=25 --restore_model=model/pointnet_13_dilated_embed_test/model24.ckpt
python train_s3dis_v2.py