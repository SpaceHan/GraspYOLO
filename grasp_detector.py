from nets.GraspDet import GraspDet_body
from keras.layers import Input, Conv2D
from keras.models import Model
import os
import math
import keras.backend as K
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np


class GraspDetector():
    def __init__(self, input_shape, class_path, weights_path):
        self.session = K.get_session()
        self.weights = weights_path
        self.input_shape = input_shape
        self.classes = self._get_classes(class_path)
        self.num_classes = len(self.classes)
        self.detector = self._get_model()
        self.score = 0.8

    def _get_model(self):
        model_input = Input(shape=(self.input_shape[0], self.input_shape[1], 3))

        model = GraspDet_body(model_input, self.num_classes)  # Model(input, dark)
        # model.summary()

        model.load_weights(self.weights)  # , by_name=True, skip_mismatch=True

        return model

    def _get_classes(self, cls_path):
        classes = []
        with open(cls_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            classes.append(line.strip())

        return classes

    def _get_single_image_data(self, img_path):
        image = Image.open(img_path)

        image = image.resize(self.input_shape, Image.BICUBIC)
        image_data = np.array(image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)

        return image_data

    def _get_multi_imgs_data(self, dir, names):
        '''

        :param folder:
        :param shape:
        :return:
        '''
        imgs_data = []

        for iname in names:
            image = Image.open(os.path.join(dir, iname))

            image = image.resize(self.input_shape, Image.BICUBIC)
            image_data = np.array(image, dtype='float32')
            image_data /= 255.
            imgs_data.append(image_data)

        return np.array(imgs_data)

    def _get_feasible_grasp_points(self, prob_map, threshold):
        '''
        获取合适抓取点：prob高于thresh，且大于8邻域
        :param prob_map:
        :param threshold:
        :return:
        '''
        map_h, map_w = prob_map.shape

        ret = []
        for i in range(1, map_h - 1):
            for j in range(1, map_w - 1):
                if prob_map[i, j] > threshold and prob_map[i, j] > max(prob_map[i - 1, j], prob_map[i + 1, j],
                                                                       prob_map[i, j - 1], prob_map[i, j + 1]):
                    # print(prob_map[i, j])
                    ret.append([i, j])

        return ret

    def _half_angle(self, sin2theta, cos2theta):
        '''
        半角公式，根据输出的sin2theta、cos2theta求抓取角度
        :param sin2theta:
        :param cos2theta:
        :return:
        '''
        t = math.atan(sin2theta / (1 + cos2theta))
        return math.degrees(t)

    def _compute_real_grasp(self, feats, index, score_thresh):
        '''
        从网络输出中计算真正的抓取数据
        :param feats:
        :return:
        '''
        # x、y、d、sin、cos数值图
        x_map = feats[index, :, :, 0]
        y_map = feats[index, :, :, 1]
        d_map = feats[index, :, :, 2]
        sin_map = feats[index, :, :, 3]
        cos_map = feats[index, :, :, 4]
        # 抓取位置概率图
        prob_map = feats[index, :, :, 5]

        '''print("x_map : ", x_map)
        print("y_map : ", y_map)
        print("d_map : ", d_map)
        print("s_map : ", sin_map)
        print("c_map : ", cos_map)'''

        # 根据概率图求存在抓取的网格
        grids = self._get_feasible_grasp_points(prob_map, score_thresh)

        num_grasps = len(grids)

        real_grasp_data = []  # np.zeros(shape=(num_grasps, 4))    # 每个抓取表示为x、y、dis、theta、clss.

        for grid in grids:
            grasp_x = grid[0] * 32 + x_map[grid[0], grid[1]] * 32
            grasp_y = grid[1] * 32 + y_map[grid[0], grid[1]] * 32
            grasp_d = d_map[grid[0], grid[1]] * 240
            sin2t = sin_map[grid[0], grid[1]]
            cos2t = cos_map[grid[0], grid[1]]
            grasp_theta = self._half_angle(sin2t, cos2t)
            grasp_class = np.argmax(feats[0, grid[0], grid[1], 6:])
            # print(grid, "==", grasp_x, grasp_y, grasp_d, sin2t, cos2t, grasp_theta, grasp_class)
            real_grasp_data.append([grasp_x, grasp_y, grasp_d, grasp_theta, grasp_class])
        real_grasp_data = np.array(real_grasp_data)

        real_grasp_data = real_grasp_data[real_grasp_data[:, 2] > 20]  # 筛选：抓取长度大于32像素；
        return real_grasp_data

    def _draw_grasps_in_image(self, im_path, grasps):
        '''

        :param im_path:
        :param grasps:
        :return:
        '''
        img = Image.open(im_path)
        img_width, img_height = img.size
        # print(img_width, img_height)
        draw = ImageDraw.Draw(img)

        for grasp in grasps:
            y = (grasp[0] / 416) * img_height
            x = (grasp[1] / 416) * img_width
            # print(grasp[0], grasp[1], x, y)
            width = grasp[2]
            radius = width / 2
            anglePi = grasp[3]

            cosA = math.cos(anglePi)
            sinA = math.sin(anglePi)

            a = sinA * radius
            b = cosA * radius

            x0 = x - b
            y0 = y - a
            x1 = x + b
            y1 = y + a

            # print(anglePi, radius, sinA, cosA, a, b, x0, y0, x1, y1)

            draw.line([(x0, y0), (x1, y1)], fill=(0, 0, 255), width=5)
            draw.text((x - 15, y + 15), self.classes[int(grasp[4])], fill=(255, 0, 0))

        return img

    def predict_an_image(self, image_path):
        img_data = self._get_single_image_data(image_path)
        output = self.session.run(self.detector.output, feed_dict={self.detector.input: img_data})
        grasps = self._compute_real_grasp(output, 0, self.score)

        print("图片 {} 共有可用抓取 {} 个".format(image_path, len(grasps)))
        #print(grasps)

        drawed_img = self._draw_grasps_in_image(image_path, grasps)

        plt.imshow(drawed_img)
        plt.show()
        pass

    def predict_a_folder(self, src_folder, dst_folder, batch_size):
        img_names = os.listdir(src_folder)
        num_tests = len(img_names)

        for k in range(num_tests // batch_size):
            batch_img_names = img_names[batch_size * k: batch_size * (k + 1)]
            data_imgs = self._get_multi_imgs_data(src_folder, batch_img_names)  # (batch, 416, 416, 3)

            feats = self.session.run(self.detector.output, feed_dict={self.detector.input: data_imgs})  # (batch, 13, 13, 6+C)

            for i in range(batch_size):
                img_index_of_all = k * batch_size + i
                grasps = self._compute_real_grasp(feats, i, self.score)
                print("图片 {} 共有可用抓取 {} 个".format(img_index_of_all, len(grasps)))
                # print(grasps)

                drawed_img = self._draw_grasps_in_image(os.path.join(src_folder, img_names[img_index_of_all]), grasps)

                drawed_img.save(os.path.join(dst_folder, img_names[img_index_of_all]))
        pass




