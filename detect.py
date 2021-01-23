from grasp_detector import GraspDetector


def detect_image(model, image_path):
    model.predict_an_image(image_path)
    pass


def detect_folder(model, folder, dst_path, batch_size):
    model.predict_a_folder(folder, dst_path, batch_size)

if __name__ == "__main__":
    #
    img_path = "test_imgs/test2.jpg"    # 测试图片路径
    folder_path = "test_imgs/"          # 测试目录

    dst = "imgs_out/"                   # 输出文件夹
    bs = 4                              # batch_size

    # 获取模型
    detector = GraspDetector(input_shape=(416, 416),
                             class_path="model_data/classes_new.txt",
                             weights_path="model_data/trained_22_cos_noact_frese102.h5")

    # 测试单个图片
    # detect_image(detector, img_path)

    # 测试目录下所有图片
    detect_folder(detector, folder_path, dst, bs)