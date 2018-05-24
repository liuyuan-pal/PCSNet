import numpy as np
import sys
sys.path.append('..')
from pcs_ops import sum_gather
from test_util import *
import tensorflow as tf


def sum_gather_forward_np(feats,lens,begs,cens):
    s,f=feats.shape
    m=lens.shape[0]
    feats_gather=np.empty([m,f],np.float64)
    for i,l in enumerate(lens):
        feats_gather[i]=np.sum(feats[begs[i]:begs[i]+l],axis=0)

    return feats_gather

def sum_gather_backward_np(feats, dfeats_gather, lens, begs, cens):
    dfeats=dfeats_gather[cens]
    return dfeats


def test_np_ops():
    m=64
    f=16
    lens = np.random.randint(0, 5, [m])
    begs = compute_nbegs(lens)
    cens = compute_ncens(lens)
    s = np.sum(lens)

    feats = np.random.uniform(-1, 1, [s, f])
    dfeats_gather = np.random.uniform(-1, 1, [m, f])

    _=sum_gather_forward_np(feats,lens,begs,cens)
    dfeats=sum_gather_backward_np(feats,dfeats_gather,lens,begs,cens)

    fn=lambda feats: sum_gather_forward_np(feats,lens,begs,cens)
    dfeats_num=eval_numerical_gradient_array(fn,feats,dfeats_gather)

    print np.mean(np.abs(dfeats_num-dfeats))
    print np.max(np.abs(dfeats_num-dfeats))

def eval_val(feats, dfeats_gather, lens, begs, cens, sess):
    feats_pl=tf.placeholder(tf.float32,[None,None])
    cens_pl=tf.placeholder(tf.int32,[None])
    begs_pl=tf.placeholder(tf.int32,[None])
    lens_pl=tf.placeholder(tf.int32,[None])
    dfeats_gather_pl=tf.placeholder(tf.float32,[None,None])

    feats_gather=sum_gather(feats_pl,lens_pl,begs_pl,cens_pl)
    dfeats=tf.gradients(feats_gather,feats_pl,dfeats_gather_pl)[0]

    feats_gather_val,dfeats_val=sess.run([feats_gather,dfeats],feed_dict={
        feats_pl:feats,
        cens_pl:cens,
        begs_pl:begs,
        lens_pl:lens,
        dfeats_gather_pl:dfeats_gather,
    })

    return feats_gather_val,dfeats_val

def test_single(m,f,sess):
    lens = np.random.randint(0, 5, [m])
    begs = compute_nbegs(lens)
    cens = compute_ncens(lens)
    s = np.sum(lens)

    feats = np.random.uniform(-1, 1, [s, f])
    dfeats_gather = np.random.uniform(-1, 1, [m, f])

    feats_gather = sum_gather_forward_np(feats, lens, begs, cens)
    dfeats = sum_gather_backward_np(feats, dfeats_gather, lens, begs, cens)
    feats_gather_val, dfeats_val = eval_val(feats, dfeats_gather, lens, begs, cens, sess)

    diff_abs = np.abs(feats_gather - feats_gather_val)
    if np.mean(diff_abs) > 1e-5 or np.max(diff_abs) > 1e-4:
        print 'error!'
        print m, f
        exit(0)

    diff_abs = np.abs(dfeats - dfeats_val)
    if np.mean(diff_abs) > 1e-5 or np.max(diff_abs) > 1e-4:
        print 'error!'
        print m, f
        exit(0)

def test():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess=tf.Session(config=config)

    for _ in xrange(100):
        pn=np.random.randint(30,1030)
        fd=np.random.randint(30,1030)
        test_single(pn,fd,sess)

    for _ in xrange(100):
        pn=np.random.randint(1020,1030)
        fd=np.random.randint(100,200)
        test_single(pn,fd,sess)

    for _ in xrange(100):
        pn=np.random.randint(100,200)
        fd=np.random.randint(1020,1030)
        test_single(pn,fd,sess)


if __name__=="__main__":
    test()
