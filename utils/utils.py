import numpy as np
import keras
import keras.backend as K
from functools import reduce
import math
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import cv2

np.set_printoptions(suppress=True)


def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


'''
为图片添加相框
'''
def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image


def prepare_ytrue(true_boxes, input_shape, num_classes):
    #print("true_boxes[..., -1].shape : ", true_boxes.shape, true_boxes[..., -1].shape)
    assert (true_boxes[..., -1] < num_classes).all(), 'class id must be less than num_classes'

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    # 读出xy轴，读出长宽
    # 中心点(m,n,2)
    grasp_center = true_boxes[..., 0:2].copy()
    boxes_diameter = true_boxes[..., 2].copy()
    # print(true_boxes[..., 0:1].shape, input_shape[::-1].shape)

    true_boxes[..., 0:1] = true_boxes[..., 0:1] / input_shape[-1]
    true_boxes[..., 1:2] = true_boxes[..., 1:2] / input_shape[0]


    # m张图
    num_imgs = true_boxes.shape[0]
    grid_shape = input_shape // 32          # 最终特征图大小 (416, 416)  ==>  (13, 13)

    y_true = np.zeros((num_imgs, grid_shape[0], grid_shape[1], 6+num_classes), dtype='float32')

    net_scale = np.array((32, 32), dtype='int32')

    for i in range(num_imgs):                    # 遍历每个图片
        for b in true_boxes[i]:           # 遍历每个box
            if b[2] == 0:                 # 跳过填充max_box的数据
                continue
            # print("b: ", b)
            m = np.floor(b[0] * grid_shape[1]).astype('int32')
            n = np.floor(b[1] * grid_shape[0]).astype('int32')

            y_true[i, n, m, 0] = (b[0] * input_shape[-1] % net_scale[-1]) / net_scale[-1]    # x：抓取中心相对网格点左上点
            y_true[i, n, m, 1] = (b[1] * input_shape[0] % net_scale[0]) / net_scale[0]
            # print("...", y_true[i, n, m, 0:2])
            y_true[i, n, m, 2] = b[2]                                     # 直径
            y_true[i, n, m, 3:5] = b[3:5]                                 # 2sin、2cos
            y_true[i, n, m, 5] = 1                                        # prob
            c = b[5].astype('int32')
            y_true[i, n, m, 6+c] = 1                                      # 类别one-hot

    # print("y_true : ", y_true.shape, y_true)
    return y_true

'''
获取一个图片的训练数据
'''
def get_random_data(annotation_line, input_shape, max_dis, max_boxes=10, jitter=.3, hue=.1, sat=1.5, val=1.5):
    '''获取旋转抓取标注的数据'''
    line = annotation_line.split()  # 将一行标注打散成list
    # print("annotation_line : ", annotation_line)

    # 处理图像
    image = Image.open(line[0])
    #plt.imshow(image)
    #plt.show()
    ow, oh = image.size
    ih, iw = input_shape
    image = image.resize(input_shape, Image.BICUBIC)
    # 色域扭曲
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
    x[..., 0] += hue * 360
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x[:, :, 0] > 360, 0] = 360
    x[:, :, 1:][x[:, :, 1:] > 1] = 1
    x[x < 0] = 0
    image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)  # numpy array, 0 to 1
    #plt.imshow(image_data)
    #plt.show()

    # 处理目标标注
    boxes = np.array([np.array(list(map(float, box.split(',')))) for box in line[1:]])
    #print(boxes)

    for box in boxes:
        if box[3] > np.pi/2:
            box[3] = box[3] - np.pi

    # 将弧度在标注中处理为sin2a、cos2a
    double_theta = 2 * boxes[:, 3]
    sin_2theta = np.sin(double_theta)
    cos_2theta = np.cos(double_theta)
    boxes = np.insert(boxes, 4, cos_2theta, axis=1)
    boxes = np.insert(boxes, 4, sin_2theta, axis=1)
    boxes = np.delete(boxes, 3, axis=1)
    # 输入标注格式为：x、y、d、2sin、2cos
    num_boxes = len(boxes)
    # print("boxes : ", num_boxes, boxes)

    '''
    会导致制作y_true时维度不同
    box_data = np.zeros((num_boxes,6))
    if num_boxes > 0:
        boxes[:, 0] = boxes[:, 0] * iw / ow
        boxes[:, 1] = boxes[:, 1] * ih / oh

        boxes[:, 2] = boxes[:, 2] / 160    # 以最大抓取距离160归一化

        box_data[:len(boxes)] = boxes

    #print("box_data : ", box_data)
    # show_processed_image(image, box_data)

    if num_boxes > max_boxes:
        box_data = box_data[:max_boxes, ...]'''

    box_data = np.zeros((max_boxes, 6))
    if num_boxes > 0:
        boxes[:, 0] = boxes[:, 0] * iw / ow
        boxes[:, 1] = boxes[:, 1] * ih / oh
        boxes[:, 2] = boxes[:, 2] / max_dis  # 以最大抓取距离归一化

        if num_boxes > max_boxes:
            boxes = boxes[:max_boxes]
        box_data[:len(boxes)] = boxes

    return image_data, box_data


def show_processed_image(image, boxes):
    draw = ImageDraw.Draw(image)
    # draw.line([(50, 50), (200, 200)], fill=(255, 0, 0))
    # image.show()
    for box in boxes:
        print("box : ", box)
        x = box[0]
        y = box[1]
        width = box[2]
        height = box[3]
        angle = box[4]

        anglePi = angle  #-angle * math.pi / 180.0
        cosA = math.cos(anglePi)
        sinA = math.sin(anglePi)

        x1 = x - 0.5 * width
        y1 = y - 0.5 * height

        x0 = x + 0.5 * width
        y0 = y1

        x2 = x1
        y2 = y + 0.5 * height

        x3 = x0
        y3 = y2

        x0n = (x0 - x) * cosA - (y0 - y) * sinA + x
        y0n = (x0 - x) * sinA + (y0 - y) * cosA + y

        x1n = (x1 - x) * cosA - (y1 - y) * sinA + x
        y1n = (x1 - x) * sinA + (y1 - y) * cosA + y

        x2n = (x2 - x) * cosA - (y2 - y) * sinA + x
        y2n = (x2 - x) * sinA + (y2 - y) * cosA + y

        x3n = (x3 - x) * cosA - (y3 - y) * sinA + x
        y3n = (x3 - x) * sinA + (y3 - y) * cosA + y

        draw.line([(x0n, y0n), (x1n, y1n)], fill=(0, 0, 255))
        draw.line([(x1n, y1n), (x2n, y2n)], fill=(255, 0, 0))
        draw.line([(x2n, y2n), (x3n, y3n)], fill=(0, 0, 255))
        draw.line([(x0n, y0n), (x3n, y3n)], fill=(255, 0, 0))
    plt.imshow(image)
    plt.show()
    pass


def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0,
                             min_learn_rate=0,
                             ):
    """
    参数：
            global_step: 上面定义的Tcur，记录当前执行的步数。
            learning_rate_base：预先设置的学习率，当warm_up阶段学习率增加到learning_rate_base，就开始学习率下降。
            total_steps: 是总的训练的步数，等于epoch*sample_count/batch_size,(sample_count是样本总数，epoch是总的循环次数)
            warmup_learning_rate: 这是warm up阶段线性增长的初始值
            warmup_steps: warm_up总的需要持续的步数
            hold_base_rate_steps: 这是可选的参数，即当warm up阶段结束后保持学习率不变，知道hold_base_rate_steps结束后才开始学习率下降
    """
    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                            'warmup_steps.')
    #这里实现了余弦退火的原理，设置学习率的最小值为0，所以简化了表达式
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(np.pi *
        (global_step - warmup_steps - hold_base_rate_steps) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    #如果hold_base_rate_steps大于0，表明在warm up结束后学习率在一定步数内保持不变
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                    learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                                'warmup_learning_rate.')
        #线性增长的实现
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        #只有当global_step 仍然处于warm up阶段才会使用线性增长的学习率warmup_rate，否则使用余弦退火的学习率learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                    learning_rate)

    learning_rate = max(learning_rate,min_learn_rate)
    return learning_rate


class WarmUpCosineDecayScheduler(keras.callbacks.Callback):
    """
    继承Callback，实现对学习率的调度
    """
    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 min_learn_rate=0,
                 # interval_epoch代表余弦退火之间的最低点
                 interval_epoch=[0.05, 0.15, 0.30, 0.50],
                 verbose=0):
        super(WarmUpCosineDecayScheduler, self).__init__()
        # 基础的学习率
        self.learning_rate_base = learning_rate_base
        # 热调整参数
        self.warmup_learning_rate = warmup_learning_rate
        # 参数显示
        self.verbose = verbose
        # learning_rates用于记录每次更新后的学习率，方便图形化观察
        self.min_learn_rate = min_learn_rate
        self.learning_rates = []

        self.interval_epoch = interval_epoch
        # 贯穿全局的步长
        self.global_step_for_interval = global_step_init
        # 用于上升的总步长
        self.warmup_steps_for_interval = warmup_steps
        # 保持最高峰的总步长
        self.hold_steps_for_interval = hold_base_rate_steps
        # 整个训练的总步长
        self.total_steps_for_interval = total_steps

        self.interval_index = 0
        # 计算出来两个最低点的间隔
        self.interval_reset = [self.interval_epoch[0]]
        for i in range(len(self.interval_epoch)-1):
            self.interval_reset.append(self.interval_epoch[i+1]-self.interval_epoch[i])
        self.interval_reset.append(1-self.interval_epoch[-1])

	#更新global_step，并记录当前学习率
    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        self.global_step_for_interval = self.global_step_for_interval + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

	#更新学习率
    def on_batch_begin(self, batch, logs=None):
        # 每到一次最低点就重新更新参数
        if self.global_step_for_interval in [0]+[int(i*self.total_steps_for_interval) for i in self.interval_epoch]:
            self.total_steps = self.total_steps_for_interval * self.interval_reset[self.interval_index]
            self.warmup_steps = self.warmup_steps_for_interval * self.interval_reset[self.interval_index]
            self.hold_base_rate_steps = self.hold_steps_for_interval * self.interval_reset[self.interval_index]
            self.global_step = 0
            self.interval_index += 1

        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps,
                                      min_learn_rate = self.min_learn_rate)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_step + 1, lr))