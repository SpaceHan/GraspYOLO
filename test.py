from nets.GraspDet import GraspDet_body
from keras.layers import Input, Conv2D
from keras.models import Model
import keras.backend as K
from PIL import Image
import matplotlib.pyplot as plt
from utils.utils import letterbox_image
import numpy as np


test_img_path = "test_imgs/test1.jpg"
image = Image.open(test_img_path)

new_image_size = (416, 416)
boxed_image = letterbox_image(image, new_image_size)
image_data = np.array(boxed_image, dtype='float32')
image_data /= 255.
image_data = np.expand_dims(image_data, 0)


input = Input(shape=(416, 416, 3))

model = GraspDet_body(input, 12) #Model(input, dark)
# model.summary()

model.load_weights('model_data/last1.h5', by_name=True, skip_mismatch=True)

# print(model.output.shape)    # (?, 13, 13, 18)

sess = K.get_session()
feats = sess.run(model.output, feed_dict={input: image_data})


for i in range(6):
    mater = feats[0, :, :, i]
    plt.matshow(mater, cmap=plt.cm.Blues)
    plt.show()

'''f1_val = feats
# print(f1_val)
print(f1_val.shape)

layer = [35, 69, 200, 167]
map_l = f1_val[0, ..., layer]

ll_res = Conv2D(23, (3,3))(model.output)
print(ll_res.shape)


fig, axes = plt.subplots(2,2)
axes[0][0].imshow(map_l[0])
axes[0][1].imshow(map_l[1])
axes[1][0].imshow(map_l[2])
axes[1][1].imshow(map_l[3])
plt.show()'''