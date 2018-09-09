from test_util import *
import sys
sys.path.append('..')
from pcs_ops import pcs_ops
import tensorflow as tf
neighbor_ops=tf.load_op_library('/home/liuyuan/project/Segmentation/tf_ops/build/libTFNeighborOps.so')

from tensorflow.python.framework import ops

@ops.RegisterGradient("NeighborScatter")
def _neighbor_scatter_gradient(op,dsfeats):
    use_diff=op.get_attr('use_diff')
    difeats=neighbor_ops.neighbor_gather(dsfeats, op.inputs[1], op.inputs[2], op.inputs[3], use_diff=use_diff)
    return [difeats,None,None,None]


@ops.RegisterGradient("LocationWeightFeatSum")
def _location_weight_feat_sum_gradient(op,dtfeats_sum):
    dlw,dfeats=neighbor_ops.location_weight_feat_sum_backward(op.inputs[0], op.inputs[1], dtfeats_sum, op.inputs[2], op.inputs[3])
    return [dlw,dfeats,None,None]


@ops.RegisterGradient("NeighborSumFeatGather")
def _neighbor_sum_feat_gather_gradient(op, dgfeats):
    difeats=neighbor_ops.neighbor_sum_feat_scatter(dgfeats, op.inputs[1], op.inputs[2], op.inputs[3])
    return [difeats,None,None,None]


@ops.RegisterGradient("NeighborSumFeatScatter")
def _neighbor_sum_feat_scatter_gradient(op,dsfeats):
    difeats=neighbor_ops.neighbor_sum_feat_gather(dsfeats, op.inputs[1], op.inputs[2], op.inputs[3])
    return [difeats,None,None,None]

@ops.RegisterGradient("NeighborMaxFeatGather")
def _neighbor_max_feat_gather_gradient(op,dgfeats,dmax_idxs):
    difeats=neighbor_ops.neighbor_max_feat_scatter(dgfeats,op.inputs[0],op.outputs[1],op.inputs[2],op.inputs[1])
    return [difeats,None,None]


def eval_val_scatter_compare(feats, dfeats, idxs, lens, begs, cens, sess):
    feats_pl=tf.placeholder(tf.float32,[None,None])
    idxs_pl=tf.placeholder(tf.int32,[None])
    lens_pl=tf.placeholder(tf.int32,[None])
    begs_pl=tf.placeholder(tf.int32,[None])
    cens_pl=tf.placeholder(tf.int32,[None])
    dfeats_scatter_pl=tf.placeholder(tf.float32,[None,None])

    nsf1=pcs_ops.idxs_scatter(feats_pl,idxs_pl,lens_pl)
    ndf1=tf.gradients(nsf1,feats_pl,dfeats_scatter_pl)[0]
    osf1=neighbor_ops.neighbor_scatter(feats_pl,idxs_pl,lens_pl,begs_pl,use_diff=False)
    odf1=tf.gradients(osf1,feats_pl,dfeats_scatter_pl)[0]

    nsf2=pcs_ops.repeat_scatter(feats_pl,lens_pl,begs_pl,cens_pl)
    ndf2=tf.gradients(nsf2,feats_pl,dfeats_scatter_pl)[0]
    osf2=neighbor_ops.neighbor_sum_feat_scatter(feats_pl,cens_pl,lens_pl,begs_pl)
    odf2=tf.gradients(osf2,feats_pl,dfeats_scatter_pl)[0]

    nsf1, ndf1, osf1, odf1, nsf2, ndf2, osf2, odf2=\
        sess.run([nsf1,ndf1,osf1,odf1,nsf2,ndf2,osf2,odf2],feed_dict={
            feats_pl:feats,
            idxs_pl:idxs,
            lens_pl:lens,
            begs_pl:begs,
            cens_pl:cens,
            dfeats_scatter_pl:dfeats,
        })

    df1=np.abs(nsf1-osf1)
    db1=np.abs(ndf1-odf1)
    df2=np.abs(nsf2-osf2)
    db2=np.abs(ndf2-odf2)

    return df1,df2,db1,db2

def eval_val_gather_compare(feats, dfeats, idxs, lens, begs, cens, sess):
    feats_pl=tf.placeholder(tf.float32,[None,None])
    idxs_pl=tf.placeholder(tf.int32,[None])
    lens_pl=tf.placeholder(tf.int32,[None])
    begs_pl=tf.placeholder(tf.int32,[None])
    cens_pl=tf.placeholder(tf.int32,[None])
    dfeats_gather_pl=tf.placeholder(tf.float32,[None,None])

    nsf1=pcs_ops.sum_gather(feats_pl,lens_pl,begs_pl,cens_pl)
    ndf1=tf.gradients(nsf1,feats_pl,dfeats_gather_pl)[0]
    osf1=neighbor_ops.neighbor_sum_feat_gather(feats_pl,cens_pl,lens_pl,begs_pl)
    odf1=tf.gradients(osf1,feats_pl,dfeats_gather_pl)[0]

    nsf2,_=pcs_ops.max_gather(feats_pl,lens_pl,begs_pl,cens_pl)
    ndf2=tf.gradients(nsf2,feats_pl,dfeats_gather_pl)[0]
    osf2,_=neighbor_ops.neighbor_max_feat_gather(feats_pl,lens_pl,begs_pl)
    odf2=tf.gradients(osf2,feats_pl,dfeats_gather_pl)[0]

    nsf1, ndf1, osf1, odf1, nsf2, ndf2, osf2, odf2=\
        sess.run([nsf1,ndf1,osf1,odf1,nsf2,ndf2,osf2,odf2],feed_dict={
            feats_pl:feats,
            idxs_pl:idxs,
            lens_pl:lens,
            begs_pl:begs,
            cens_pl:cens,
            dfeats_gather_pl:dfeats,
        })

    df1=np.abs(nsf1-osf1)
    db1=np.abs(ndf1-odf1)
    df2=np.abs(nsf2-osf2)
    db2=np.abs(ndf2-odf2)

    return df1,df2,db1,db2

def test_val(val,pn,fd,message):
    if np.mean(val)>1e-5 or np.max(val)>1e-4:
        print val
        print pn,fd,np.max(val),np.mean(val)
        print '{} error'.format(message)
        exit(0)


def test_scatter_single(pn,fd,sess):
    feats=np.random.uniform(-1,1,[pn,fd])
    nidxs=[]
    nlens=[]
    for i in xrange(pn):
        num=np.random.randint(0,10)
        nlens.append(num)
        if num>0:
            idxs=np.random.choice(pn,num,False)
            nidxs.append(idxs)

    nlens=np.asarray(nlens)
    nbegs=compute_nbegs(nlens)
    ncens=compute_ncens(nlens)
    nidxs=np.concatenate(nidxs,axis=0)

    dfeats=np.random.uniform(-1,1,[ncens.shape[0],fd])
    df1, df2, db1, db2=eval_val_scatter_compare(feats,dfeats,nidxs,nlens,nbegs,ncens,sess)

    test_val(df1,pn,fd,'idxs scatter forward')
    test_val(df2,pn,fd,'repeat scatter forward')
    test_val(db1,pn,fd,'idxs scatter backward')
    test_val(db2,pn,fd,'repeat scatter backward')

def test_gather_single(pn,fd,sess):
    nidxs=[]
    nlens=[]
    for i in xrange(pn):
        num=np.random.randint(0,10)
        nlens.append(num)
        if num>0:
            idxs=np.random.choice(pn,num,False)
            nidxs.append(idxs)

    nlens=np.asarray(nlens)
    nbegs=compute_nbegs(nlens)
    ncens=compute_ncens(nlens)
    nidxs=np.concatenate(nidxs,axis=0)
    feats=np.random.uniform(-1,1,[ncens.shape[0],fd])

    dfeats=np.random.uniform(-1,1,[pn,fd])
    df1, df2, db1, db2=eval_val_gather_compare(feats,dfeats,nidxs,nlens,nbegs,ncens,sess)

    test_val(df1,pn,fd,'sum gather forward')
    test_val(df2,pn,fd,'max gather forward')
    test_val(db1,pn,fd,'sum gather backward')
    test_val(db2,pn,fd,'max gather backward')

def test(fn=test_scatter_single):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess=tf.Session(config=config)


    for _ in xrange(100):
        pn=np.random.randint(30,1030)
        fd=np.random.randint(30,1030)
        fn(pn,fd,sess)

    for _ in xrange(100):
        pn=np.random.randint(1020,1030)
        fd=np.random.randint(100,200)
        fn(pn,fd,sess)

    for _ in xrange(100):
        pn=np.random.randint(100,200)
        fd=np.random.randint(1020,1030)
        fn(pn,fd,sess)


if __name__=="__main__":
    test(test_gather_single)