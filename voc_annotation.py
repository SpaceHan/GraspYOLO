import xml.etree.ElementTree as ET
from os import getcwd

sets = [('2021', 'train'), ('2021', 'val'), ('2021', 'test')]

# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
classes = ["MagicCube", "CiggButt", "Chopsticks", "Bone", "Banana", "FishBone", "Hanger", "Can", "Shoe", "BeverageBottle", "DryBattery", "Ointment"]


def convert_annotation(year, image_id, list_file):
    in_file = open('VOCdevkit/GBG%s/Annotations/%s.xml' % (year, image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text

        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('robndbox')
        bw = round(float(xmlbox.find('w').text))
        bh = round(float(xmlbox.find('h').text))
        grasp_dis = max(bw, bh)
        rot_box = (
            round(float(xmlbox.find('cx').text)), round(float(xmlbox.find('cy').text)), grasp_dis,
            round(float(xmlbox.find('angle').text), 3))
        list_file.write(" " + ",".join([str(a) for a in rot_box]) + ',' + str(cls_id))

wd = getcwd()


for year, image_set in sets:
    image_ids = open('VOCdevkit/GBG%s/ImageSets/Main/%s.txt' % (year, image_set)).read().strip().split()
    print(image_ids)
    list_file = open('%s_%s.txt' % (year, image_set), 'w')
    for image_id in image_ids:
        image_id = image_id.strip()
        list_file.write('%s/VOCdevkit/GBG%s/JPEGImages/%s.jpg' % (wd, year, image_id))
        print(image_id)
        convert_annotation(year, image_id, list_file)
        list_file.write('\n')
    list_file.close()
