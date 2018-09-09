from pcs_ops import max_gather,sum_gather,idxs_scatter,repeat_scatter,\
    search_neighborhood,search_neighborhood_range,points_pooling_two_layers
import tensorflow as tf

use_tf_gather=True

def concat_neighboring_feats(feats, nidxs, nlens, nbegs, ncens):
    if use_tf_gather:
        centering_feats = tf.gather(feats,ncens)
        neighboring_feats = tf.gather(feats,nidxs)
    else:
        centering_feats = repeat_scatter(feats,nlens,nbegs,ncens)
        neighboring_feats = idxs_scatter(feats,nidxs,nlens)
    concat_feats = tf.concat([centering_feats, neighboring_feats], axis=1)
    return concat_feats

def diff_neighboring_feats(feats, nidxs, nlens, nbegs, ncens):
    if use_tf_gather:
        centering_feats = tf.gather(feats,ncens)
        neighboring_feats = tf.gather(feats,nidxs)
    else:
        centering_feats = repeat_scatter(feats,nlens,nbegs,ncens)
        neighboring_feats = idxs_scatter(feats,nidxs,nlens)
    diff_feats = neighboring_feats-centering_feats
    return diff_feats

def average_pool(feats,vlens,vbegs,vcens):
    feats=sum_gather(feats,vlens,vbegs,vcens)
    feats/=tf.expand_dims(tf.cast(vlens,tf.float32),axis=1)
    return feats

def max_pool(feats,vlens,vbegs,vcens):
    feats=max_gather(feats,vlens,vbegs,vcens)
    return feats

def connected_mlp(feats, fc_dims, final_dim, name, reuse=None):
    for idx,fd in enumerate(fc_dims):
        cfeats=tf.contrib.layers.fully_connected(feats, num_outputs=fd, scope='{}_fc_{}'.format(name,idx),
                                                 activation_fn=tf.nn.relu, reuse=reuse)
        feats=tf.concat([cfeats,feats],axis=1)

    feats=tf.contrib.layers.fully_connected(feats, num_outputs=final_dim, scope='{}_fc_out'.format(name),
                                            activation_fn=None, reuse=reuse)

    return feats


def connected_pointnet(sxyzs, feats, fc_dims, final_dim, name, nidxs, nlens, nbegs, ncens, reuse=None):
    with tf.name_scope(name):
        sfeats = concat_neighboring_feats(feats, nidxs, nlens, nbegs, ncens)
        sfeats = tf.concat([sfeats,sxyzs],axis=1)
        sfeats=connected_mlp(sfeats, fc_dims, final_dim, name, reuse)
        feats=max_pool(sfeats,nlens,nbegs,ncens)

    return feats


def connected_pointnet_nofeats(sxyzs, fc_dims, final_dim, name, nidxs, nlens, nbegs, ncens, reuse=None):
    with tf.name_scope(name):
        sfeats=connected_mlp(sxyzs, fc_dims, final_dim, name, reuse)
        feats=max_pool(sfeats,nlens,nbegs,ncens)

    return feats


def fc_embed(feats, name, embed_dim, reuse):
    ofeats=tf.contrib.layers.fully_connected(feats, num_outputs=embed_dim, scope='{}_fc_embed'.format(name),
                                             activation_fn=tf.nn.leaky_relu, reuse=reuse)
    return ofeats

