import pickle
import numpy as np
import math

def save_pkl(filename,obj):
    with open(filename,'wb') as f:
        pickle.dump(obj,f,protocol=2)


def read_pkl(filename):
    with open(filename,'rb') as f:
        obj=pickle.load(f)
    return obj

def flip(points,axis=0):
    result_points=points[:]
    result_points[:,axis]=-result_points[:,axis]
    return result_points


def swap_xy(points):
    result_points = np.empty_like(points, dtype=np.float32)
    result_points[:,0]=points[:,1]
    result_points[:,1]=points[:,0]
    result_points[:,2:]=points[:,2:]

    return result_points


def rotate(xyz,rotation_angle):
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, sinval, 0],
                                [-sinval, cosval,  0],
                                [      0,      0, 1]],dtype=np.float32)
    xyz[:,:3]=np.dot(xyz[:,:3],rotation_matrix)
    return xyz


def _fetch_subset(all, idxs, subsets=None):
    if subsets is None:
        subsets=[[] for _ in range(len(all))]

    for item,subset in zip(all,subsets):
        for idx in idxs:
            subset.append(item[idx])

    return subsets


def _append(lists, items):
    for l,i in zip(lists,items):
        l.append(i)


def get_beg_list(maxx,block_size,stride,resample_ratio=0.03,back_sample=False):
    x_list=[]
    spacex=maxx-block_size
    if spacex<0: x_list.append(0)
    else:
        x_list+=list(np.arange(0,spacex,stride))
        if back_sample:
            if (spacex-int(spacex/stride)*stride)/block_size>resample_ratio:
                x_list+=list(np.arange(spacex,0,-stride))
        else:
            x_list.append(spacex)

    return x_list


def uniform_sample_block(xyzs, block_size=3.0, stride=1.5, min_pn=2048, normalized=True, use_gpu=True):
    assert stride<=block_size
    if not normalized:
        xyzs -= np.min(xyzs, axis=0, keepdims=True)

    max_xyz=np.max(xyzs, axis=0, keepdims=True)
    maxx,maxy=max_xyz[0,0],max_xyz[0,1]
    beg_list=[]
    x_list=get_beg_list(maxx,block_size,stride,False)
    y_list=get_beg_list(maxy,block_size,stride,False)
    for x in x_list:
        for y in y_list:
            beg_list.append((x,y))

    if not use_gpu:
        idxs=[]
        for beg in beg_list:
            x_cond= (xyzs[:, 0] >= beg[0]) & (xyzs[:, 0] < beg[0] + block_size)
            y_cond= (xyzs[:, 1] >= beg[1]) & (xyzs[:, 1] < beg[1] + block_size)
            cond=x_cond&y_cond
            if(np.sum(cond)<min_pn):
                continue
            idxs.append((np.nonzero(cond))[0])
    else:
        import libPointUtil
        beg_list=np.asarray(beg_list,dtype=np.float32)
        idxs=libPointUtil.gatherBlockPointsGPU(xyzs, beg_list, block_size)
        idxs=[idx for idx in idxs if len(idx)>=min_pn]

    return idxs


def voxel_downsample_idxs(xyzs,voxel_len,use_gpu=True):
    if use_gpu:
        import libPointUtil
        ds_idxs=libPointUtil.gridDownsampleGPU(xyzs,voxel_len,False)
    else:
        loc2pt = {}
        for pt_index, pt in enumerate(xyzs):
            x_index = int(math.ceil(pt[0] / voxel_len))
            y_index = int(math.ceil(pt[1] / voxel_len))
            z_index = int(math.ceil(pt[2] / voxel_len))
            loc = (x_index, y_index, z_index)
            if loc in loc2pt:
                loc2pt[loc].append(pt_index)
            else:
                loc2pt[loc] = [pt_index]

        ds_idxs = []
        for k, v in loc2pt.items():
            grid_index = int(np.random.randint(0, len(v), 1))
            ds_idxs.append(v[grid_index])

    return ds_idxs


def sample_block(points, labels, ds_stride, block_size, block_stride, min_pn,
                 rescale=False, swap=False, flip_x=False, flip_y=False,
                 rotation=False, rot_ang=0.0):

    xyzs=np.ascontiguousarray(points[:,:3])
    rgbs=np.ascontiguousarray(points[:,3:])

    ds_idxs=voxel_downsample_idxs(xyzs,ds_stride)
    xyzs=xyzs[ds_idxs,:]
    rgbs=rgbs[ds_idxs,:]
    lbls=labels[ds_idxs]

    # flip and swap
    if swap:
        xyzs=swap_xy(xyzs)
    if flip_x:
        xyzs=flip(xyzs,axis=0)

    if flip_y:
        xyzs=flip(xyzs,axis=1)

    # rotation
    if rotation:
        xyzs=rotate(xyzs,rot_ang)

    # rescale
    if rescale:
        rescale_val=np.random.uniform(0.9,1.1,[1,3])
        xyzs[:,:3]*=rescale_val

    min_xyzs=np.min(xyzs,axis=0,keepdims=True)
    xyzs-=min_xyzs
    idxs = uniform_sample_block(xyzs,block_size,block_stride,normalized=True,min_pn=min_pn)
    xyzs+=min_xyzs

    xyzs, rgbs, lbls=_fetch_subset([xyzs, rgbs, lbls], idxs)

    return xyzs, rgbs, lbls


def normalize_block(xyzs,rgbs,lbls,block_size,resample=False,
                    resample_low=0.8,resample_high=1.0,
                    max_sample=False,max_pt_num=10240,
                    jitter_color=False,jitter_val=2.5):
    bn=len(xyzs)
    block_mins=[]
    for bid in range(bn):
        if resample:
            pt_num=len(xyzs[bid])
            random_down_ratio=np.random.uniform(resample_low,resample_high)
            idxs=np.random.choice(pt_num,int(pt_num*random_down_ratio))
            xyzs[bid]=xyzs[bid][idxs,:]
            rgbs[bid]=rgbs[bid][idxs,:]
            lbls[bid]=lbls[bid][idxs]

        if max_sample and len(xyzs[bid])>max_pt_num:
            pt_num=len(xyzs[bid])
            ratio=max_pt_num/float(len(xyzs[bid]))
            idxs=np.random.choice(pt_num,int(pt_num*ratio))
            xyzs[bid]=xyzs[bid][idxs,:]
            rgbs[bid]=rgbs[bid][idxs,:]
            lbls[bid]=lbls[bid][idxs]

        # offset center to zero
        min_xyz=np.min(xyzs[bid],axis=0,keepdims=True)
        min_xyz[:,:2]+=block_size/2.0
        xyzs[bid]-=min_xyz
        block_mins.append(min_xyz)

        if jitter_color:
            rgbs[bid]+=np.random.uniform(-jitter_val,jitter_val,rgbs[bid].shape)
            rgbs[bid]-=128
            rgbs[bid]/=(128+jitter_val)
        else:
            rgbs[bid]-=128
            rgbs[bid]/=(128+jitter_val)

        mask=lbls[bid]>12
        if np.sum(mask)>0:
            lbls[bid][mask]=12
        lbls[bid]=lbls[bid].flatten()

    return xyzs,rgbs,lbls,block_mins
