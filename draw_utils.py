import numpy as np

def get_s3dis_class_colors():
    colors=np.asarray(
            [[0,255,0],
            [0,0,255],
            [93,201,235],
            [255,255,0],
            [255,140,0],
            [0,0,128],
            [255,69,0],
            [255,127,80],
            [255,0,0],
            [255,250,240],
            [255,0,255],
            [255,255,255],
            [105,105,105],
            [205,92,92]],dtype=np.int)
    return colors

def output_points(filename,pts,colors=None):
    has_color=pts.shape[1]>=6
    with open(filename, 'w') as f:
        for i,pt in enumerate(pts):
            if colors is None:
                if has_color:
                    f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],int(pt[3]),int(pt[4]),int(pt[5])))
                else:
                    f.write('{} {} {}\n'.format(pt[0],pt[1],pt[2]))

            else:
                if colors.shape[0]==pts.shape[0]:
                    f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],int(colors[i,0]),int(colors[i,1]),int(colors[i,2])))
                else:
                    f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],int(colors[0]),int(colors[1]),int(colors[2])))