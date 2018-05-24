import sys
sys.path.append('..')
from pcs_ops import points_pooling_two_layers,compute_voxel_idx
import tensorflow as tf
import numpy as np
from test_util import output_points


def eval(pts, feats, labels, sess):
    xyzs_pl=tf.placeholder(tf.float32,[None,3])
    feats_pl=tf.placeholder(tf.float32,[None,3])
    labels_pl=tf.placeholder(tf.int32,[None])

    [xyzs1,xyzs2,xyzs3], [dxyzs1,dxyzs2], pfeats, plabels, [vlens1,vlens2], [vbegs1,vbegs2], [vcens1,vcens2]=\
        points_pooling_two_layers(xyzs_pl, feats_pl, labels_pl, 0.15, 0.5)

    p1,p2,p3,d1,d2,f,l,vl1,vl2,vb1,vb2,vc1,vc2=sess.run(
        [xyzs1,xyzs2,xyzs3,dxyzs1,dxyzs2,pfeats,plabels,vlens1,vlens2,vbegs1,vbegs2,vcens1,vcens2],
        feed_dict={
            xyzs_pl:pts,
            feats_pl:feats,
            labels_pl:labels,
        })

    return p1,p2,p3,d1,d2,f,l,vl1,vl2,vb1,vb2,vc1,vc2

def eval_compute_idxs(pts, sess):
    xyzs_pl=tf.placeholder(tf.float32,[None,3])
    idxs=compute_voxel_idx(xyzs_pl,0.3)
    vidxs=sess.run(
        idxs,feed_dict={
            xyzs_pl:pts,
        })
    return vidxs

def test_compute_idxs():
    pt_num=40960
    xyzs=np.random.uniform(-1.5,1.5,[pt_num,3])

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess=tf.Session(config=config)
    idxs=eval_compute_idxs(xyzs,sess)

    flatten_idxs=[]
    for i in xrange(len(idxs)):
        flatten_idxs.append(idxs[i][0]*11*11+idxs[i][1]*11+idxs[i][2])
    flatten_idxs=np.asarray(flatten_idxs,np.int32)

    colors=np.random.randint(0,256,[np.max(flatten_idxs)+1,3])
    print colors[flatten_idxs]
    output_points('test_result/xyzs.txt',xyzs,colors[flatten_idxs])


def check_vidxs(max_cens,max_len,lens,begs,cens):
    nbegs=np.cumsum(lens)
    assert np.sum(nbegs[:-1]!=begs[1:])==0
    assert begs[0]==0

    assert np.sum(cens>=max_cens)==0
    assert max_len==lens[-1]+begs[-1]


def output_hierarchy(pts1,pts2,cens,name):
    colors=np.random.randint(0,256,[len(pts2),3])
    output_points('test_result/{}_dense.txt'.format(name),pts1,colors[cens,:])
    output_points('test_result/{}_sparse.txt'.format(name),pts2,colors)


def check_dxyzs(pts1,pts2,dpts1,vcens):
    pn1=pts1.shape[0]
    tmp_dpts1=np.copy(dpts1)
    for i in xrange(pn1):
        tmp_dpts1[i]+=pts2[vcens[i]]

    print np.mean(np.abs(tmp_dpts1-pts1),axis=0),np.max(np.abs(tmp_dpts1-pts1),axis=0)

def test():
    xyzs=np.random.uniform(-1.5,1.5,[10240,3])
    rbgs=np.random.uniform(-1.0,1.0,[10240,3])
    lbls=np.random.randint(0,5,[10240])

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess=tf.Session(config=config)
    p1, p2, p3, d1, d2, f, l, vl1, vl2, vb1, vb2, vc1, vc2=eval(xyzs,rbgs,lbls,sess)

    check_vidxs(len(p2),len(p1),vl1,vb1,vc1)
    check_vidxs(len(p3),len(p2),vl2,vb2,vc2)

    output_hierarchy(p1,p2,vc1,'12')
    output_hierarchy(p2,p3,vc2,'23')

def test_data():
    import cPickle
    with open('test_data/251_Area_5_conferenceRoom_1.pkl','r') as f:
        xyzs, rgbs, covars, lbls, block_mins=cPickle.load(f)

    xyzs=xyzs[0]
    rgbs=rgbs[0]
    lbls=lbls[0]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess=tf.Session(config=config)
    p1, p2, p3, d1, d2, f, l, vl1, vl2, vb1, vb2, vc1, vc2=eval(xyzs,rgbs,lbls,sess)

    check_vidxs(len(p2),len(p1),vl1,vb1,vc1)
    check_vidxs(len(p3),len(p2),vl2,vb2,vc2)

    output_hierarchy(p1,p2,vc1,'12')
    output_hierarchy(p2,p3,vc2,'23')

    output_points('test_result/rgbs.txt',p1,f*127+128)


if __name__=="__main__":
    test_data()
