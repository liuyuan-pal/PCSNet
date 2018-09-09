from layer import *
from tensorflow.contrib import framework


def classifier(feats, pfeats, is_training, num_classes, reuse=False):
    '''

    :param feats: k,f
    :param pfeats:
    :param is_training:
    :param num_classes:
    :param reuse:
    :return:
    '''
    with tf.name_scope('segmentation_classifier'):
        class_mlp1 = tf.contrib.layers.fully_connected(feats, num_outputs=512, scope='classifier_mlp1',
                                                       activation_fn=tf.nn.relu, reuse=reuse)
        class_mlp1 = tf.concat([class_mlp1, pfeats], axis=1)
        class_mlp1 = tf.cond(is_training, lambda: tf.nn.dropout(class_mlp1, 0.7), lambda: class_mlp1)

        class_mlp2 = tf.contrib.layers.fully_connected(class_mlp1, num_outputs=256, scope='classifier_mlp2',
                                                       activation_fn=tf.nn.relu, reuse=reuse)
        class_mlp2 = tf.concat([class_mlp2, pfeats], axis=1)
        class_mlp2 = tf.cond(is_training,lambda:tf.nn.dropout(class_mlp2,0.7),lambda:class_mlp2)

        logits = tf.contrib.layers.fully_connected(class_mlp2, num_outputs=num_classes, scope='classifier_mlp3',
                                                   activation_fn=None, reuse=reuse)

    return logits

def classifier_mlp(feats, pfeats, is_training, num_classes, reuse=False, use_bn=False):
    '''
    :param feats: n,k,f
    :param pfeats:
    :param is_training:
    :param num_classes:
    :param reuse:
    :return:
    '''
    normalizer_params={'scale':False,'is_training':is_training,'reuse':reuse}
    feats=tf.expand_dims(feats,axis=2)     # n,k,1,2048+6
    pfeats=tf.expand_dims(pfeats, axis=2)  # n,k,1,6
    bn=tf.contrib.layers.batch_norm if use_bn else None
    with tf.name_scope('segmentation_classifier'):
        with framework.arg_scope([tf.contrib.layers.conv2d],kernel_size=[1,1],stride=1,
                                 padding='VALID',activation_fn=tf.nn.relu,reuse=reuse,
                                 normalizer_fn=bn):

            # feats = tf.cond(is_training, lambda: tf.nn.dropout(feats, 0.5), lambda: feats)
            normalizer_params['scope']='class_mlp1_bn'
            class_mlp1 = tf.contrib.layers.conv2d(
                feats, num_outputs=512, scope='class_mlp1',normalizer_params=normalizer_params)
            class_mlp1=tf.concat([class_mlp1, pfeats], axis=3)
            class_mlp1 = tf.cond(is_training, lambda: tf.nn.dropout(class_mlp1, 0.7), lambda: class_mlp1)

            normalizer_params['scope']='class_mlp2_bn'
            class_mlp2 = tf.contrib.layers.conv2d(
                class_mlp1, num_outputs=256, scope='class_mlp2',normalizer_params=normalizer_params)
            class_mlp2=tf.concat([class_mlp2, pfeats], axis=3)
            class_mlp2=tf.cond(is_training,lambda:tf.nn.dropout(class_mlp2,0.7),lambda:class_mlp2)

            logits = tf.contrib.layers.conv2d(
                class_mlp2, num_outputs=num_classes, scope='class_mlp3',activation_fn=None,normalizer_fn=None)

        logits=tf.squeeze(logits,axis=2,name='logits')

    return logits

def pointnet_13_dilated_embed(xyzs, dxyzs, feats, vlens, vbegs, vcens, reuse=False):
    with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
        feats1=average_pool(feats,vlens[0],vbegs[0],vcens[0])
        feats2=average_pool(feats1,vlens[1],vbegs[1],vcens[1])

        with tf.name_scope('stage0'):
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.15) # 29
            sxyzs = diff_neighboring_feats(xyzs[0], nidxs, nlens, nbegs, ncens)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats_pn=connected_pointnet(sxyzs,feats,[8,8,16],32,'feats0',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood_range(xyzs[0], 0.15, 0.2) # 22
            sxyzs = diff_neighboring_feats(xyzs[0], nidxs, nlens, nbegs, ncens)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats_pn=connected_pointnet(sxyzs,feats,[8,8,16],32,'feats1',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood_range(xyzs[0], 0.1, 0.15) # 16
            sxyzs = diff_neighboring_feats(xyzs[0], nidxs, nlens, nbegs, ncens)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats_pn=connected_pointnet(sxyzs,feats,[8,8,16],32,'feats2',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.1)  # 12
            sxyzs = diff_neighboring_feats(xyzs[0], nidxs, nlens, nbegs, ncens)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats_ed=fc_embed(feats,'embed3',32,reuse)
            feats_pn=connected_pointnet(sxyzs,feats_ed,[8,8,16],32,'feats3',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage0=tf.concat([feats,feats_pn],axis=1)

            voxel_stage0_fc=connected_mlp(tf.concat([dxyzs[0],feats_stage0],axis=1),[8,8,16],32,'pool0',reuse)
            voxel_stage0_pool=max_pool(voxel_stage0_fc,vlens[0],vbegs[0],vcens[0])
            feats_stage0_pool=max_pool(feats_stage0,vlens[0],vbegs[0],vcens[0])
            feats_stage0_pool=tf.concat([feats1,feats_stage0_pool,voxel_stage0_pool],axis=1)

        with tf.name_scope('stage1'):
            feats=feats_stage0_pool
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.45) # 30
            sxyzs = diff_neighboring_feats(xyzs[1], nidxs, nlens, nbegs, ncens)  # [en,ifn]
            sxyzs /= 0.45  # rescale
            feats_ed=fc_embed(feats,'embed4',64,reuse)
            feats_pn=connected_pointnet(sxyzs,feats_ed,[16,16,32],64,'feats4',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood_range(xyzs[1], 0.45, 0.6) # 24
            sxyzs = diff_neighboring_feats(xyzs[1], nidxs, nlens, nbegs, ncens)  # [en,ifn]
            sxyzs /= 0.45  # rescale
            feats_ed=fc_embed(feats,'embed5',48,reuse)
            feats_pn=connected_pointnet(sxyzs,feats_ed,[16,16,16],48,'feats5',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_ed=fc_embed(feats,'embed6',48,reuse)
            feats_pn=connected_pointnet(sxyzs,feats_ed,[16,16,16],48,'feats6',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood_range(xyzs[1], 0.3, 0.45) # 16
            sxyzs = diff_neighboring_feats(xyzs[1], nidxs, nlens, nbegs, ncens)  # [en,ifn]
            sxyzs /= 0.45  # rescale

            feats_ed=fc_embed(feats,'embed7',64,reuse)
            feats_pn=connected_pointnet(sxyzs,feats_ed,[16,16,16],48,'feats7',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_ed=fc_embed(feats,'embed8',64,reuse)
            feats_pn=connected_pointnet(sxyzs,feats_ed,[16,16,16],48,'feats8',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.3) # 12
            sxyzs = diff_neighboring_feats(xyzs[1], nidxs, nlens, nbegs, ncens)  # [en,ifn]
            sxyzs /= 0.45  # rescale

            feats_ed=fc_embed(feats,'embed9',96,reuse)
            feats_pn=connected_pointnet(sxyzs,feats_ed,[16,16,16],48,'feats9',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_ed=fc_embed(feats,'embed10',96,reuse)
            feats_pn=connected_pointnet(sxyzs,feats_ed,[16,16,16],48,'feats10',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage1=tf.concat([feats,feats_pn],axis=1)

            voxel_stage1_fc=connected_mlp(tf.concat([dxyzs[1],feats_stage1],axis=1),[16,16,16],48,'pool1',reuse)
            voxel_stage1_pool=max_pool(voxel_stage1_fc,vlens[1],vbegs[1],vcens[1])
            feats_stage1_pool=max_pool(feats_stage1,vlens[1],vbegs[1],vcens[1])
            feats_stage1_pool=tf.concat([feats2,feats_stage1_pool,voxel_stage1_pool],axis=1)

        with tf.name_scope('stage2'):
            feats=feats_stage1_pool

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[2], 0.9)
            sxyzs = diff_neighboring_feats(xyzs[2], nidxs, nlens, nbegs, ncens)  # [en,ifn]
            sxyzs /= 0.9  # rescale

            feats_ed=fc_embed(feats,'embed11',128,reuse)
            feats_pn=connected_pointnet(sxyzs,feats_ed,[16,16,32],64,'feats11',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_ed=fc_embed(feats,'embed12',128,reuse)
            feats_pn=connected_pointnet(sxyzs,feats_ed,[16,16,32],64,'feats12',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage2=tf.concat([feats,feats_pn],axis=1)

            feats=tf.concat([xyzs[2],feats],axis=1)
            feats_stage2_fc=connected_mlp(feats,[32,32,48],128,'global',reuse)

        with tf.name_scope('unpool'):
            lf2=tf.concat([feats_stage2,feats_stage2_fc],axis=1)
            lf2_up=tf.gather(lf2,vcens[1])
            lf1=tf.concat([lf2_up,feats_stage1],axis=1)
            lf1_up=tf.gather(lf1,vcens[0])
            lf0=tf.concat([lf1_up,feats_stage0],axis=1)

        return lf0, feats_stage0