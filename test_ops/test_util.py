import numpy as np

def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad


def compute_nbegs(nlens):
    clen=0
    nbegs=np.empty_like(nlens)
    for i,len_val in enumerate(nlens):
        nbegs[i]=clen
        clen+=len_val
    return nbegs


def compute_ncens(nlens):
    ncens=[]
    for i,len_val in enumerate(nlens):
        for _ in xrange(len_val):
            ncens.append(i)

    return np.asarray(ncens,np.int32)


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

