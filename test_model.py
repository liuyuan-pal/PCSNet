from model import *
import os
from dataset.data_util import read_pkl
import numpy as np

def build_network(xyzs, feats, labels, is_training, reuse=False):
    xyzs, dxyzs, feats, labels, vlens, vbegs, vcens = \
        points_pooling_two_layers(xyzs,feats,labels,voxel_size1=0.15,voxel_size2=0.45)
    global_feats, local_feats = pointnet_13_dilated_embed(xyzs,dxyzs,feats,vlens,vbegs,vcens,reuse)

    global_feats=tf.expand_dims(global_feats,axis=0)
    local_feats=tf.expand_dims(local_feats,axis=0)
    logits=classifier_mlp(global_feats, local_feats, is_training, 13, False, use_bn=False)  # [1,pn,num_classes]

    flatten_logits=tf.reshape(logits,[-1,13])  # [pn,num_classes]
    probs=tf.nn.softmax(flatten_logits)
    preds=tf.argmax(flatten_logits,axis=1)
    ops={}
    ops['feats']=feats
    ops['probs']=probs
    ops['logits']=flatten_logits
    ops['preds']=preds
    ops['labels']=labels
    ops['xyzs']=xyzs[0]
    return ops

def build_pls():
    xyzs_pl=tf.placeholder(tf.float32,[None,3])
    feats_pl=tf.placeholder(tf.float32,[None,3])
    labels_pl=tf.placeholder(tf.int32,[None])

    return xyzs_pl,feats_pl,labels_pl

def build_session():
    xyzs_pl, feats_pl, labels_pl=build_pls()
    ops=build_network(xyzs_pl,feats_pl,labels_pl,tf.constant(False))

    feed_dict=dict()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)
    saver = tf.train.Saver(max_to_keep=500)
    model_path='/home/liuyuan/project/Segmentation/model/pointnet_13_dilated_embed/model40.ckpt'
    print 'restoring {}'.format(model_path)
    saver.restore(sess,model_path)
    return sess, [xyzs_pl, feats_pl, labels_pl], ops, feed_dict

def get_block_train_test_split(test_area=5):
    '''
    :param test_area: default use area 5 as testset
    :return:
    '''
    path = os.path.split(os.path.realpath(__file__))[0]
    f = open(path + '/dataset/cache/room_block_stems.txt', 'r')
    file_stems = [line.strip('\n') for line in f.readlines()]
    f.close()

    train, test = [], []
    for fs in file_stems:
        if fs.split('_')[2] == str(test_area):
            test.append(fs)
        else:
            train.append(fs)

    return train, test

def eval():
    train_list,test_list=get_block_train_test_split()
    sess, pls, ops, feed_dict=build_session()
    xyzs, rgbs, _, lbls, block_mins=read_pkl('/home/liuyuan/data/S3DIS/sampled_test_nolimits/'+test_list[0])
    all_preds,all_labels=[],[]
    for k in xrange(len(xyzs)):
        logits,labels=sess.run([ops['logits'],ops['labels']],feed_dict={
            pls[0]:xyzs[k],
            pls[1]:rgbs[k],
            pls[2]:lbls[k],
        })
        preds=np.argmax(logits,axis=1)
        all_preds.append(preds)
        all_labels.append(labels)

    all_preds=np.concatenate(all_preds,axis=0)
    all_labels=np.concatenate(all_labels,axis=0)
    print np.sum(all_preds==all_labels)/float(len(all_preds))

if __name__=="__main__":
    eval()