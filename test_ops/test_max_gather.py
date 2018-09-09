from test_util import *
import sys
sys.path.append('..')
from pcs_ops import max_gather
import tensorflow as tf

def max_gather_np(feats, nlens, nbegs):
    s,f=feats.shape
    m=nlens.shape[0]

    feats_gather=np.zeros([m,f],np.float64)
    idxs_gather=np.zeros([m,f],np.int)
    for i in xrange(m):
        if nlens[i]==0: continue
        cfeats= feats[nbegs[i]:nbegs[i] + nlens[i]]
        feats_gather[i]=np.max(cfeats,axis=0)
        idxs_gather[i]=np.argmax(cfeats,axis=0)

    return feats_gather,idxs_gather


def max_scatter_backward(feats, idxs_gather, dfeats_gather, nlens, nbegs):
    s,f=feats.shape
    m=idxs_gather.shape[0]

    dfeats=np.zeros([s,f],np.float64)
    for i in xrange(m):
        if nlens[i]==0: continue
        cur_dfeats=np.zeros([nlens[i],f])
        cur_dfeats[idxs_gather[i],np.arange(f)]=dfeats_gather[i]
        dfeats[nbegs[i]:nbegs[i]+nlens[i]]=cur_dfeats

    return dfeats


def test_np_ops():
    m=64
    f=8
    lens = np.random.randint(0, 5, [m])
    begs = compute_nbegs(lens)
    cens = compute_ncens(lens)
    s = np.sum(lens)

    feats = np.random.uniform(-1, 1, [s, f])
    dfeats_gather = np.random.uniform(-1, 1, [m, f])

    _,idxs=max_gather_np(feats,lens,begs)
    dfeats=max_scatter_backward(feats,idxs,dfeats_gather,lens,begs)

    fn=lambda feats: max_gather_np(feats,lens,begs)[0]
    dfeats_num=eval_numerical_gradient_array(fn,feats,dfeats_gather)

    print dfeats_num
    print np.mean(np.abs(dfeats_num-dfeats))
    print np.max(np.abs(dfeats_num-dfeats))

def eval_val(feats, dfeats_gather, lens, begs, cens, sess):
    feats_pl=tf.placeholder(tf.float32,[None,None])
    cens_pl=tf.placeholder(tf.int32,[None])
    begs_pl=tf.placeholder(tf.int32,[None])
    lens_pl=tf.placeholder(tf.int32,[None])
    dfeats_gather_pl=tf.placeholder(tf.float32,[None,None])

    feats_gather=max_gather(feats_pl,lens_pl,begs_pl,cens_pl)
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

    feats_gather,idxs_gather= max_gather_np(feats, lens, begs)
    dfeats = max_scatter_backward(feats, idxs_gather, dfeats_gather, lens, begs)
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

    # idxs_diff=np.sum(idxs_gather-idxs_gather_val)
    # if idxs_diff>0:
    #     xs,ys=np.nonzero(np.not_equal(idxs_gather,idxs_gather_val))
    #     error=False
    #     for k in xrange(len(xs)):
    #         diff_val=feats[begs[xs[k]]+idxs_gather[xs[k],ys[k]],ys[k]]-feats[begs[xs[k]]+idxs_gather_val[xs[k],ys[k]],ys[k]]
    #         if abs(diff_val)>1e-5:
    #             error=True
    #             break
    #
    #     if error:
    #         print 'idxs error!'
    #         print m,f
    #         exit(0)
    #     else:
    #         return

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
    test_old()
