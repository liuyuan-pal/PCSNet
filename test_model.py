from draw_utils import output_points, get_s3dis_class_colors
from model import *
from dataset.data_util import read_pkl
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path',type=str,default='model/s3dis/model99.ckpt')
parser.add_argument('--data_path',type=str,default='data/s3dis/188_Area_5_office_22.pkl')
args = parser.parse_args()


def build_network(xyzs, feats, labels, is_training, reuse=False):
    xyzs, dxyzs, feats, labels, vlens, vbegs, vcens = \
        points_pooling_two_layers(xyzs,feats,labels,voxel_size1=0.15,voxel_size2=0.45)
    global_feats, local_feats = pointnet_13_dilated_embed(xyzs, dxyzs, feats, vlens, vbegs, vcens, reuse)
    logits=classifier(global_feats, local_feats, is_training, 13, False)

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
    model_path=args.model_path
    print('restoring {}'.format(model_path))
    saver.restore(sess,model_path)
    return sess, [xyzs_pl, feats_pl, labels_pl], ops, feed_dict


def eval():
    xyzs, rgbs, lbls, block_mins=read_pkl(args.data_path)

    sess, pls, ops, feed_dict=build_session()

    all_preds, all_labels, all_xyzs, all_rgbs=[], [], [] ,[]
    for k in range(len(xyzs)):
        perm_xyzs,perm_rgbs,logits,labels=\
        sess.run([ops['xyzs'],ops['feats'],ops['logits'],ops['labels']],feed_dict={
            pls[0]:xyzs[k],
            pls[1]:rgbs[k],
            pls[2]:lbls[k],
        })
        preds=np.argmax(logits,axis=1)
        all_preds.append(preds)
        all_labels.append(labels)
        all_xyzs.append(perm_xyzs)
        all_rgbs.append(perm_rgbs)

    all_preds=np.concatenate(all_preds,axis=0)
    all_labels=np.concatenate(all_labels,axis=0)

    for xyz_i in range(len(all_xyzs)):
        all_xyzs[xyz_i]+=block_mins[xyz_i]
    all_xyzs=np.concatenate(all_xyzs,axis=0)
    all_rgbs=np.asarray(np.concatenate(all_rgbs,axis=0)*127+128).astype(np.uint8)

    print('accuracy {:.2f}%'.format(np.sum(all_preds==all_labels)/float(len(all_preds))*100))
    print('output labels ...')

    colors=get_s3dis_class_colors()
    name='_'.join(args.data_path.split('/')[-1].split('.')[0].split('_')[1:])
    output_points('result/{}_rgb.txt'.format(name),all_xyzs,all_rgbs)
    output_points('result/{}_pr.txt'.format(name),all_xyzs,colors[all_preds])
    output_points('result/{}_gt.txt'.format(name),all_xyzs,colors[all_labels])

if __name__=="__main__":
    eval()