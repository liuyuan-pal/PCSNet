from data_util import read_pkl,save_pkl, sample_block, normalize_block
import time
import os
import glob
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def get_class_names():
    names=[]
    path=os.path.split(os.path.realpath(__file__))[0]
    with open(os.path.join(path,'cache','s3dis_class_names.txt'),'r') as f:
        for line in f.readlines():
            names.append(line.strip('\n'))

    return names


def read_room_dirs():
    dirs=[]
    path=os.path.split(os.path.realpath(__file__))[0]
    with open(os.path.join(path,'cache','s3dis_dir_names.txt'),'r') as f:
        for line in f.readlines():
            line=line.strip('\n')
            dirs.append(os.path.join(*line.split(' ')))
    return dirs


def read_room(room_dir,class_names):
    points=[]
    labels=[]
    for f in glob.glob(os.path.join(room_dir,'*.txt')):
        class_str=os.path.basename(f).split('_')[0]
        class_index=class_names.index(class_str)
        xyz_rgb=np.loadtxt(f,dtype=np.float32)
        label=np.ones([xyz_rgb.shape[0],1],dtype=np.uint8)*class_index
        points.append(xyz_rgb)
        labels.append(label)

    points=np.concatenate(points,axis=0)
    labels=np.concatenate(labels,axis=0)

    return points,labels


def read_train_test_stems(test_area=5):
    '''
    :param test_area: default use area 5 as testset
    :return:
    '''
    path = os.path.split(os.path.realpath(__file__))[0]
    f = open(os.path.join(path,'cache','s3dis_room_stems.txt'), 'r')
    file_stems = [line.strip('\n') for line in f.readlines()]
    f.close()

    train, test = [], []
    for fs in file_stems:
        if fs.split('_')[2] == str(test_area):
            test.append(fs)
        else:
            train.append(fs)

    return train, test


def raw2pkl(raw_dir, pkl_dir):
    class_names=get_class_names()
    room_dirs=read_room_dirs()
    if not os.path.exists(pkl_dir):
        os.mkdir(pkl_dir)

    path=os.path.split(os.path.realpath(__file__))[0]
    f=open(os.path.join(path,'cache','s3dis_room_names.txt'),'w')
    for dir_index,dir in enumerate(room_dirs):
        points,labels=read_room(os.path.join(raw_dir, dir), class_names)
        fn=str(dir[:-12]).replace(os.path.sep, '_')
        fn="{}_{}.pkl".format(dir_index,fn)
        labels[labels>=13]=12  # mark points of stairs as clutter
        save_pkl(os.path.join(pkl_dir, fn), [points, labels[:, 0]])
        f.write('{}\n'.format(fn))
        print fn
    f.close()


block_size=3.0
block_stride=1.5
ds_stride=0.05
min_pn=512


def prepare_data(fn,use_rescale,use_swap,use_flip_x,use_flip_y,resample,jitter_color,cur_min_pn=min_pn):
    points, labels = read_pkl(fn)
    xyzs, rgbs, lbls = sample_block(points, labels, ds_stride, block_size, block_stride, min_pn=cur_min_pn, rescale=use_rescale,
                                    swap=use_swap, flip_x=use_flip_x, flip_y=use_flip_y, rotation=False)

    xyzs, rgbs, lbls, block_mins = normalize_block(xyzs, rgbs, lbls, block_size, resample=resample,
                                                   resample_low=0.8, resample_high=1.0,
                                                   max_sample=False, jitter_color=jitter_color, jitter_val=2.5)

    return xyzs, rgbs, lbls, block_mins


def prepare_s3dis_train_single_file(pkl_dir, output_dir, fn):
    room_fn=os.path.join(pkl_dir, fn)
    all_data=[[] for _ in xrange(4)]
    bg=time.time()

    data = prepare_data(room_fn, True, True, False, False, True, True)
    for t in xrange(4):
        all_data[t]+=data[t]
    data = prepare_data(room_fn, True, True, True, False, True, True)
    for t in xrange(4):
        all_data[t]+=data[t]
    data = prepare_data(room_fn, True, True, False, True, True, True)
    for t in xrange(4):
        all_data[t]+=data[t]
    data = prepare_data(room_fn, True, True, True, True, True, True)
    for t in xrange(4):
        all_data[t]+=data[t]

    data = prepare_data(room_fn, True, False, False, False, True, True)
    for t in xrange(4):
        all_data[t]+=data[t]
    data = prepare_data(room_fn, True, False, True, False, True, True)
    for t in xrange(4):
        all_data[t]+=data[t]
    data = prepare_data(room_fn, True, False, False, True, True, True)
    for t in xrange(4):
        all_data[t]+=data[t]
    data = prepare_data(room_fn, True, False, True, True, True, True)
    for t in xrange(4):
        all_data[t]+=data[t]

    out_fn=os.path.join(output_dir,fn)
    save_pkl(out_fn,all_data)
    print 'done {} cost {} s'.format(fn,time.time()-bg)


def prepare_s3dis_test_single_file(pkl_dir, output_dir, fn):
    room_fn=os.path.join(pkl_dir, fn)
    bg=time.time()
    data = prepare_data(room_fn, False, False, False, False, False, False, 128)
    output_fn=os.path.join(output_dir,fn)
    save_pkl(output_fn,data)
    print 'done {} cost {} s'.format(fn,time.time()-bg)


def prepare_dataset(pkl_dir,output_dir,num_cpus=4):
    executor=ProcessPoolExecutor(max_workers=num_cpus)
    train_list,test_list=read_train_test_stems()

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    futures=[]
    for fn in train_list:
        futures.append(executor.submit(prepare_s3dis_train_single_file,pkl_dir,output_dir,fn))

    for future in futures:
        future.result()

    for fn in test_list:
        futures.append(executor.submit(prepare_s3dis_test_single_file,pkl_dir,output_dir,fn))

    for future in futures:
        future.result()


def output_points(filename, pts, colors=None):
    has_color = pts.shape[1] >= 6
    with open(filename, 'w') as f:
        for i, pt in enumerate(pts):
            if colors is None:
                if has_color:
                    f.write('{} {} {} {} {} {}\n'.format(pt[0], pt[1], pt[2], int(pt[3]), int(pt[4]), int(pt[5])))
                else:
                    f.write('{} {} {}\n'.format(pt[0], pt[1], pt[2]))

            else:
                if colors.shape[0] == pts.shape[0]:
                    f.write('{} {} {} {} {} {}\n'.format(pt[0], pt[1], pt[2], int(colors[i, 0]), int(colors[i, 1]),
                                                         int(colors[i, 2])))
                else:
                    f.write('{} {} {} {} {} {}\n'.format(pt[0], pt[1], pt[2], int(colors[0]), int(colors[1]),
                                                         int(colors[2])))


if __name__=="__main__":
    # points,labels=read_pkl('/home/pal/data/test/pkl/0_Area_1_conferenceRoom_1.pkl')
    # print points.shape
    # print labels.shape
    # print np.min(points,axis=0),np.max(points,axis=0)
    # print np.min(labels,axis=0),np.max(labels,axis=0)
    # raw2pkl('/home/pal/data/Stanford3dDataset_v1.2_Aligned_Version','/home/pal/data/test')
    # prepare_dataset('/home/liuyuan/data/test/pkl','/home/liuyuan/data/test/block_test_5',4)

    train_list,test_list=read_train_test_stems()
    label_counts = np.zeros(13)
    block_counts = 0
    point_counts = []
    f=open('tmp.txt','w')
    for tfs in  train_list:
        xyzs,rgbs,lbls,block_mins=read_pkl('/home/liuyuan/data/test/block_test_5/{}'.format(tfs))
        block_counts+=len(xyzs)
        for i in xrange(len(xyzs)):
            # print xyzs[i].shape,rgbs[i].shape,lbls[i].shape
            # print np.min(xyzs[i],axis=0),np.max(xyzs[i],axis=0)
            # print np.min(rgbs[i],axis=0),np.max(rgbs[i],axis=0)
            hist,_=np.histogram(lbls[i],np.arange(14))
            label_counts+=hist
            point_counts.append(len(xyzs[i]))

        f.write('{} {}\n'.format(tfs,len(xyzs)))

    f.close()

    print block_counts
    print label_counts
    print np.histogram(point_counts)

    # xyzs, rgbs, lbls, block_mins = read_pkl('/home/liuyuan/data/test/block_test_5/{}'.format('1_Area_1_conferenceRoom_2.pkl'))
    # for i in xrange(len(xyzs)):
    #     output_points('test_result/{}.txt'.format(i),xyzs[i]+block_mins[i],rgbs[i]*127+128)
