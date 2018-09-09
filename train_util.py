import tensorflow as tf
import numpy as np

def compute_iou(label,pred,num_classes=13):
    fp = np.zeros(num_classes, dtype=np.int)
    tp = np.zeros(num_classes, dtype=np.int)
    fn = np.zeros(num_classes, dtype=np.int)
    correct_mask=label==pred
    incorrect_mask=label!=pred
    for i in range(num_classes):
        label_mask=label==i
        pred_mask=pred==i

        tp[i]=np.sum(correct_mask&label_mask)
        fn[i]=np.sum(incorrect_mask&label_mask)
        fp[i]=np.sum(incorrect_mask&pred_mask)

    iou = tp / (fp + fn + tp + 1e-6).astype(np.float)
    miou=np.mean(iou)
    oiou=np.sum(tp) / float(np.sum(tp + fn + fp))
    acc = tp / (tp + fn + 1e-6)
    macc = np.mean(acc)
    oacc = np.sum(tp) / float(np.sum(tp+fn))

    return iou, miou, oiou, acc, macc, oacc



def acc_val(label,pred,fp,tp,fn,num_classes=13):
    correct_mask=(label==pred)
    incorrect_mask=(label!=pred)
    for i in range(num_classes):
        label_mask=label==i
        pred_mask=pred==i

        tp[i]+=int(np.sum(correct_mask&label_mask))
        fn[i]+=int(np.sum(incorrect_mask&label_mask))
        fp[i]+=int(np.sum(incorrect_mask&pred_mask))

    return fp,tp,fn


def val2iou(fp,tp,fn):
    iou = tp / (fp + fn + tp + 1e-6).astype(np.float)
    miou=np.mean(iou)
    oiou=np.sum(tp) / float(np.sum(tp + fn + fp))
    acc = tp / (tp + fn + 1e-6)
    macc = np.mean(acc)
    oacc = np.sum(tp) / float(np.sum(tp+fn))

    return iou, miou, oiou, acc, macc, oacc

def log_str(message,filename,print_message=True):
    with open(filename,'a') as f:
        f.write(message+'\n')
    if print_message:
        print message


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads