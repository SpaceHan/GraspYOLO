from grasp_detector import GraspDetector
import os


def detect_image(model, image_path):
    model.predict_an_image(image_path)
    pass


def detect_folder(model, folder, dst_path, batch_size):
    model.predict_a_folder(folder, dst_path, batch_size)

if __name__ == "__main__":
    #
    img_path = "test_imgs/beve.jpg"    # 测试图片路径

    folder_path = "test_imgs/"          # 测试目录
    bs = 2                              # batch_size

    # folder_path = ""
    # bs = 20  # batch_size

    dst = "imgs_out/"                   # 输出文件夹

    # 获取模型
    detector = GraspDetector(input_shape=(416, 416),
                             class_path="model_data/classes_new.txt",
                             #weights_path="model_data/trained_22_cos_noact_frese102.h5")
                             weights_path="model_data/trained_last_nocos_noact_86_4-4.h5")

    # 测试单个图片
    #detect_image(detector, img_path)

    # 测试目录下所有图片
    # detect_folder(detector, "VOCdevkit/GBG2021/JPEGImages/", dst, 20)
    # detect_folder(detector, "test_imgs/", dst, 2)

    # folder_path = "VOCdevkit/GBG2021/JPEGImages/"
    imgs = os.listdir(folder_path)
    #imgs = ["can.jpg", "test1.jpg"]

    detector.predict_several_for_paper(folder_path, imgs)
    for im in imgs:
        detector.pred_save(folder_path, im, dst)