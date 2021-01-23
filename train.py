import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from nets.GraspDet import GraspDet_body
from nets.loss import grasp_loss
from utils.utils import get_random_data,prepare_ytrue, rand,WarmUpCosineDecayScheduler


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_maxLen_in_annotations(anno_path):
    '''
    求一个标注文件里的最长抓取距离用于归一化
    :param anno_path: 标注文件路径，如2007_train.txt
    :return: 最大抓取长度(int)
    '''
    with open(anno_path, "r") as f:
        lines = f.readlines()

    biggest_dis = 0

    for line in lines[:10]:              # 遍历每行数据
        line = line.strip().split()

        for box in line[1:]:             # 每行第二个开始为box
            metas = box.split(",")
            box_dis = int(metas[2])      # 获取标注框的抓取距离
            biggest_dis = max(box_dis, biggest_dis)

    return biggest_dis


def data_generator(annotation_lines, batch_size, input_shape, num_classes, maxDis):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            # 每次开头打乱数据
            if i==0:
                np.random.shuffle(annotation_lines)
            # 获取一副图片及其中的box
            image, box = get_random_data(annotation_lines[i], input_shape, max_dis=maxDis, max_boxes=5)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = prepare_ytrue(box_data, input_shape, num_classes)
        yield [image_data, y_true], np.zeros(batch_size)


if __name__ == "__main__":
    # 标签的位置
    annotation_path = '2021_train.txt'
    # 获取classes和anchor的位置
    classes_path = 'model_data/classes_new.txt'

    weights_path = 'model_data/yolov4.h5'

    # 获取class列表
    class_names = get_classes(classes_path)
    num_classes = len(class_names)

    # 训练后的模型保存的位置
    log_dir = 'logs/'
    # 输入的shape大小
    # 显存比较小可以使用416x416
    # 现存比较大可以使用608x608
    input_shape = (416, 416)
    Cosine_scheduler = True
    label_smoothing = 0

    # 清除session
    K.clear_session()

    # 输入的图像为
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    net_scale = 32

    # 计算最大的抓取距离
    max_grasp_dis = get_maxLen_in_annotations(annotation_path)

    # 创建yolo模型
    print('Create GraspDet model with {} classes.'.format(num_classes))
    model_body = GraspDet_body(image_input, num_classes)

    # 载入预训练权重
    # print('Load weights {}.'.format(weights_path))
    # model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)

    y_true = Input(shape=(h//net_scale, w//net_scale, num_classes+6))    # 13*13*(x、y、d、2sin、2cos、prob + C)

    # 输入为*model_body.input, *y_true
    # 输出为model_loss
    loss_input = [model_body.output, y_true]
    model_loss = Lambda(grasp_loss, output_shape=(1,), name='grasp_loss',
                        arguments={'num_classes': num_classes, 'ignore_thresh': 0.5, 'label_smoothing': label_smoothing})(loss_input)

    model = Model([model_body.input, y_true], model_loss)

    # 训练参数设置
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'epoch{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    # 0.1用于验证，0.9用于训练
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val
    print("Using {} imgs to train and {} imgs for validation".format(num_train, num_val))

    # 每个阶段Epoch数目
    Freeze_epoch = 60
    Total_epoch = 100

    # 冻结层数
    freeze_layers = 86


    # 调整非主干模型first
    if True:
        # 冻结网络前面的特征提取层前面层
        for i in range(freeze_layers): model_body.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_body.layers)))

        # batch_size大小，由于可训练层数少可以设置稍大
        batch_size = 8
        # 最大学习率
        learning_rate_base = 1e-3
        if Cosine_scheduler:
            # 预热期
            warmup_epoch = int((Freeze_epoch - 0) * 0.2)
            # 总共的步长
            total_steps = int((Freeze_epoch - 0) * num_train / batch_size)
            # 预热步长
            warmup_steps = int(warmup_epoch * num_train / batch_size)
            # 学习率
            reduce_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                                   total_steps=total_steps,
                                                   warmup_learning_rate=1e-4,
                                                   warmup_steps=warmup_steps,
                                                   hold_base_rate_steps=num_train,
                                                   min_learn_rate=1e-6
                                                   )
            model.compile(optimizer=Adam(), loss={'grasp_loss': lambda y_true, y_pred: y_pred})
        else:
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
            model.compile(optimizer=Adam(learning_rate_base), loss={'grasp_loss': lambda y_true, y_pred: y_pred})

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(
            data_generator(lines[:num_train], batch_size, input_shape,num_classes, max_grasp_dis),
            steps_per_epoch=max(1, num_train // batch_size),
            validation_data=data_generator(lines[num_train:], batch_size, input_shape, num_classes, max_grasp_dis),
            validation_steps=max(1, num_val // batch_size),
            epochs=Freeze_epoch,
            initial_epoch=0,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_stage1.h5')


    # 解冻后训练
    if True:
        # 解冻所有层进行微调
        for i in range(freeze_layers): model_body.layers[i].trainable = True

        # batch_size大小，每次喂入多少数据
        batch_size = 4

        # 最大学习率
        learning_rate_base = 1e-4
        if Cosine_scheduler:
            # 预热期
            warmup_epoch = int((Total_epoch-Freeze_epoch)*0.2)
            # 总共的步长
            total_steps = int((Total_epoch-Freeze_epoch) * num_train / batch_size)
            # 预热步长
            warmup_steps = int(warmup_epoch * num_train / batch_size)
            # 学习率
            reduce_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                                        total_steps=total_steps,
                                                        warmup_learning_rate=1e-5,
                                                        warmup_steps=warmup_steps,
                                                        hold_base_rate_steps=num_train//2,
                                                        min_learn_rate=1e-6
                                                        )
            model.compile(optimizer=Adam(), loss={'grasp_loss': lambda y_true, y_pred: y_pred})
        else:
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
            model.compile(optimizer=Adam(learning_rate_base), loss={'grasp_loss': lambda y_true, y_pred: y_pred})

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator(lines[:num_train], batch_size, input_shape, num_classes, max_grasp_dis),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator(lines[num_train:], batch_size, input_shape, num_classes, max_grasp_dis),
                validation_steps=max(1, num_val//batch_size),
                epochs=Total_epoch,
                initial_epoch=Freeze_epoch,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_final.h5')