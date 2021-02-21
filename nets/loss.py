import numpy as np
import tensorflow as tf
from keras import backend as K


#---------------------------------------------------#
#   平滑标签
#---------------------------------------------------#
def _smooth_labels(y_true, label_smoothing):
    num_classes = tf.cast(K.shape(y_true)[-1], dtype=K.floatx())
    label_smoothing = K.constant(label_smoothing, dtype=K.floatx())
    return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes


def grasp_loss(args, num_classes, ignore_thresh=.5, label_smoothing=0.1, print_loss=False):
    print(args)
    grasp_outputs = args[0]
    y_true = args[1]

    input_shape = K.cast(K.shape(grasp_outputs)[1:3] * 32, K.dtype(y_true))

    loss = 0

    batch_size = K.shape(grasp_outputs)[0]
    batch_size_f = K.cast(batch_size, K.dtype(grasp_outputs))

    object_mask = y_true[..., 5:6]
    object_mask_bool = K.cast(object_mask, 'bool')
    # print("object_mask : ", object_mask)
    true_class_probs = y_true[..., 6:]

    if label_smoothing:
        true_class_probs = _smooth_labels(true_class_probs, label_smoothing)

    true_box = tf.boolean_mask(y_true[..., 0:4], object_mask_bool[..., 0])

    # 位置损失
    # x_loss = object_mask * tf.reduce_mean(tf.square(y_true[..., 0] - grasp_outputs[..., 0]))
    x_loss = object_mask * tf.reduce_mean(tf.square((y_true[..., 0] - grasp_outputs[..., 0]))^2)
    y_loss = object_mask * tf.reduce_mean(tf.square(y_true[..., 1] - grasp_outputs[..., 1]))

    # 长度损失
    d_loss = object_mask * tf.reduce_mean(tf.square(y_true[..., 2] - grasp_outputs[..., 2]))

    # 角度损失
    sin_loss = object_mask * tf.reduce_mean(tf.square(y_true[..., 3] - grasp_outputs[..., 3]))
    cos_loss = object_mask * tf.reduce_mean(tf.square(y_true[..., 4] - grasp_outputs[..., 4]))
    # print("losses : ", x_loss, y_loss, d_loss, sin_loss, cos_loss)

    # 置信度损失
    prob_loss = object_mask * K.binary_crossentropy(object_mask, grasp_outputs[..., 5:6], from_logits=True)
    # 类别损失
    class_loss = object_mask * K.binary_crossentropy(true_class_probs, grasp_outputs[..., 6:], from_logits=True)


    prob_loss = K.sum(prob_loss) / batch_size_f
    class_loss = K.sum(class_loss) / batch_size_f

    loss += prob_loss + class_loss + x_loss + y_loss + d_loss + sin_loss + cos_loss

    return loss