import tensorflow as tf
import os
path=os.path.split(os.path.realpath(__file__))[0]
pcs_ops=tf.load_op_library(os.path.join(path,'operators','build','PCSOps.so'))
import sys
sys.path.append(path)
from tensorflow.python.framework import ops

@ops.RegisterGradient("SumGather")
def _sum_gather_gradients(op,dfeats_gather):
    dfeats=tf.gather(dfeats_gather,op.inputs[3])
    return [dfeats,None,None,None]

@ops.RegisterGradient("MaxGather")
def _sum_gather_gradients(op,dfeats_gather,didxs_gather):
    dfeats=pcs_ops.max_scatter(dfeats_gather,op.outputs[1],op.inputs[1],op.inputs[2],op.inputs[3])
    return [dfeats,None,None,None]

def sum_gather(feats,nlens,nbegs,ncens):
    return pcs_ops.sum_gather(feats,nlens,nbegs,ncens)

def max_gather(feats,nlens,nbegs,ncens):
    feats_gather,idxs_gather=pcs_ops.max_gather(feats, nlens, nbegs, ncens)
    return feats_gather,idxs_gather

def compute_voxel_idx(xyzs,voxel_len):
    return pcs_ops.compute_voxel_index(xyzs, tf.reduce_min(xyzs,axis=0), voxel_len=voxel_len)

def points_pooling_two_layers(xyzs,feats,labels,voxel_size1,voxel_size2):
    # permutation 1
    vidxs1 = compute_voxel_idx(xyzs, voxel_len=voxel_size1)
    o2p_idxs1, vlens1, vbegs1, vcens1 = pcs_ops.compute_permutation_info(vidxs1)

    pts1 = tf.gather(xyzs, o2p_idxs1)
    feats = tf.gather(feats, o2p_idxs1)
    labels = tf.gather(labels, o2p_idxs1)

    # compute xyz 2
    pts2 = sum_gather(pts1, vlens1, vbegs1, vcens1)
    pts2 = pts2 / tf.expand_dims(tf.cast(vlens1, tf.float32), axis=1)

    # compute diff xyz 1
    repeated_pts2=tf.gather(pts2,vcens1)
    dpts1=pts1-repeated_pts2

    # permutation 2
    vidxs2 = compute_voxel_idx(pts2, voxel_len=voxel_size2)
    o2p_idxs2, vlens2, vbegs2, vcens2 = pcs_ops.compute_permutation_info(vidxs2)
    pts2=tf.gather(pts2,o2p_idxs2)

    # compute xyz 3
    pts3 = sum_gather(pts2, vlens2, vbegs2, vcens2)
    pts3 = pts3 / tf.expand_dims(tf.cast(vlens2, tf.float32), axis=1)

    # compute diff xyz 2
    repeated_pts3=tf.gather(pts3,vcens2)
    dpts2=pts2-repeated_pts3

    # repermutate 1
    reper_o2p_idxs1, vlens1, vbegs1, vcens1 = \
        pcs_ops.compute_repermutation_info(o2p_idxs2, vlens1, vbegs1, vcens1)
    pts1 = tf.gather(pts1, reper_o2p_idxs1)
    dpts1 = tf.gather(dpts1, reper_o2p_idxs1)
    feats = tf.gather(feats, reper_o2p_idxs1)
    labels = tf.gather(labels, reper_o2p_idxs1)

    return [pts1,pts2,pts3],[dpts1,dpts2],feats,labels,[vlens1,vlens2],[vbegs1,vbegs2],[vcens1,vcens2]