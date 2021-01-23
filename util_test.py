from utils.utils import get_random_data, prepare_ytrue
import numpy as np
import matplotlib.pyplot as plt

input_shape = (416, 416)
anno1 = "./VOCdevkit/GBG2021/JPEGImages/can(21).jpg 169,129,272,3.13,7 170,263,255,3.12,7"
anno2 = "./VOC/JPEGImages/grasp_garbage00003.jpg 253,344,153,3.021,1 455,466,152,0.02,1"
# get_random_data(anno, input_shape)

box_data = []
box_data.append(get_random_data(anno1, (416, 416), max_boxes=3)[1])
#box_data.append(get_random_data(anno2, (416, 416), max_boxes=3)[1])
box_data = np.array(box_data)
print("box_data.shape : ", box_data.shape)
yt = prepare_ytrue(box_data, input_shape, num_classes=12)

print("yt.shape ： ", yt.shape)
for i in range(6):    # yt.shape[-1]
    mater = yt[0, :, :, i]
    plt.matshow(mater, cmap=plt.cm.Blues)
    plt.show()

'''for i in range(yt.shape[-1]):
    mater = yt[1, :, :, i]
    plt.matshow(mater, cmap=plt.cm.Reds)
    plt.show()'''