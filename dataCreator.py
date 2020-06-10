import os
import cv2
import random
import cvutils
import numpy as np
import xml.etree.cElementTree as ET

# Darknet https://codeload.github.com/AlexeyAB/darknet/zip/master

def generate_dataset(object_path, scene_path, save_path, data_distribution):
    """
    Generates a dataset out of the given images
    :param object_path: Where the object images go
    :param scene_path: Where the background images go
    :param save_path: Where the new images will be saved
    :param data_distribution: The percentage of train images (0 - 1)
    :return:
    """
    obj_path = os.path.abspath(object_path)
    objects = os.listdir(obj_path)
    scn_path = os.path.abspath(scene_path)
    scenes = os.listdir(scn_path)
    save_path = os.path.abspath(save_path)

    object_names = []
    count = 0

    img_names = []

    for obj in objects:
        obj_img = cv2.imread(obj_path + "\\" + obj)

        # do object editing here
        # Require objects to be square instead of uneven sizes

        object_name = "".join(filter(lambda x: not x.isdigit(), obj)).split(".")[0]

        if object_name not in object_names:
            object_names.append(object_name)

        flip = False
        angle_step = 20
        max_angle = 180
        current_angle = 0

        for i in range(int((max_angle / angle_step) * 2)):

            new_size = random.randrange(100, 300)

            resized_obj = cvutils.resize(obj_img, new_size)
            new_obj = cvutils.rotate(resized_obj, current_angle)

            if flip:
                new_obj = cvutils.flip(new_obj, True)

            if current_angle >= max_angle:
                current_angle = 0
                flip = True
            else:
                current_angle += angle_step

            for scene in scenes:
                scene_img = cv2.imread(scn_path + "\\" + scene)
                scene_x, scene_y = scene_img.shape[:2]

                x_off = random.randrange(0, scene_x - new_obj.shape[0])
                y_off = random.randrange(0, scene_y - new_obj.shape[1])

                # Get both normal and flipped image
                save_image = cvutils.combine(new_obj, scene_img, x_off, y_off)

                img_names.append(f"{object_name}{str(count)}.jpg")

                cv2.imwrite(f"{save_path}\\images\\{object_name}{str(count)}.jpg", save_image)

                # Create the bounding box information

                with open(f"{save_path}\\images\\{object_name}{str(count)}.txt", "w") as bounding:
                    bounding.write(
                        f"{object_names.index(object_name)} {(x_off + new_obj.shape[0] / 2) / scene_x} {(y_off + new_obj.shape[1] / 2) / scene_y} " +
                        f"{new_obj.shape[0] / scene_x} {new_obj.shape[1] / scene_y}")

                count += 1

    # Make the custom.names file
    with open(f"{save_path}\\custom.names", "w") as names:
        for obj in object_names:
            names.write(f"{obj}\n")

    # Distribute the train and test images
    with open(f"{save_path}\\train.txt", "w") as train:
        random.shuffle(img_names)
        for i in range(int(len(img_names) * data_distribution)):
            train.write(f"{save_path}\\images\\{img_names.pop()}\n")

    # Distribute the train and test images
    with open(f"{save_path}\\test.txt", "w") as train:
        for i in img_names:
            train.write(f"{save_path}\\images\\{i}\n")

    # YOLO datafile
    with open(f"{save_path}\\detector.data", "w") as data:
        data.write(f"classes={len(object_names)}\n")
        data.write(f"train={save_path}\\train.txt\n")
        data.write(f"valid={save_path}\\test.txt\n")
        data.write(f"names={save_path}\\custom.names\n")
        data.write("backup=backup\\\n")


if __name__ == "__main__":
    generate_dataset(r"data\\objects", r"data\\scenes", r"data\\custom_data", 0.75)

"""
root = ET.Element("annotation")
ET.SubElement(root, "folder").text = "images"
ET.SubElement(root, "filename").text = f"{object_name}{str(count)}.jpg"
ET.SubElement(root, "path").text = f"{save_path}\\images\\{object_name}{str(count)}.jpg"
src = ET.SubElement(root, "source")
ET.SubElement(src, "database").text = "Unknown"
size = ET.SubElement(root, "size")
ET.SubElement(size, "width").text = str(scene_x)
ET.SubElement(size, "height").text = str(scene_y)
ET.SubElement(size, "depth").text = "3"
ET.SubElement(root, "segmented").text = "0"
obj_node = ET.SubElement(root, "object")
ET.SubElement(obj_node, "name").text = f"{object_name}"
ET.SubElement(obj_node, "pose").text = "Unspecified"
ET.SubElement(obj_node, "truncated").text = "0"
ET.SubElement(obj_node, "difficult").text = "0"
bound = ET.SubElement(obj_node, "bndbox")
ET.SubElement(bound, "xmin").text = str(x_off)
ET.SubElement(bound, "ymin").text = str(y_off)
ET.SubElement(bound, "xmax").text = str(new_obj.shape[0] + x_off)
ET.SubElement(bound, "ymax").text = str(new_obj.shape[1] + y_off)

tree = ET.ElementTree(root)
tree.write(f"{save_path}\\images\\{object_name}{str(count)}.xml")
"""