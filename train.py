import os
import cv2
import random
import cvutils
import yoloGenerator
from pathlib import Path


# Darknet https://github.com/AlexeyAB/darknet.git

def data_gen(obj_dir, scene_dir, output_dir):
    """
    Generates a dataset out of the given images
    :param obj_dir: Object images folder
    :param scene_dir: Scene images folder
    :param output_dir: Where to store the images
    :return: Array of objects and image names
    """
    object_names = []
    file_count = 0

    img_names = []

    objects = os.listdir(obj_dir)
    scenes = os.listdir(scene_dir)

    for obj in objects:
        obj_img = cv2.imread(obj_dir + "/" + obj)

        # Remove numbers from object name
        object_name = "".join(filter(lambda x: not x.isdigit(), obj)).split(".")[0]

        if object_name not in object_names:
            object_names.append(object_name)

        # Initial settings for file reproduction
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
                scene_img = cv2.imread(scene_dir + "/" + scene)
                scene_x, scene_y = scene_img.shape[:2]

                x_off = random.randrange(0, scene_x - new_obj.shape[0])
                y_off = random.randrange(0, scene_y - new_obj.shape[1])

                # Get both normal and flipped image
                save_image = cvutils.combine(new_obj, scene_img, x_off, y_off)

                file_name = f"{object_name}{str(file_count)}"

                img_names.append(f"{file_name}.jpg")

                cv2.imwrite(f"{output_dir}/{file_name}.jpg", save_image)

                # Create the bounding box information

                with open(f"{output_dir}/{file_name}.txt", "w") as bounding:
                    bounding.write(
                        f"{object_names.index(object_name)} {(x_off + new_obj.shape[0] / 2) / scene_x} {(y_off + new_obj.shape[1] / 2) / scene_y} " +
                        f"{new_obj.shape[0] / scene_x} {new_obj.shape[1] / scene_y}")

                file_count += 1

    return [object_names, img_names]


def train(obj_dir, scene_dir, model_dir, darknet_path, save_dir, model_name="latest", data_distribution=0.75):
    """
    Generates a dataset out of the given images
    :param obj_dir: Path for the objject images $objectname#.jpg
    :param scene_dir: Path for the background images
    :param model_dir: Where the model will be stored
    :param darknet_path: Darknet folder location
    :param save_dir: Where the new images will be saved
    :param model_name: Name for the model
    :param data_distribution: The percentage of train images (0 - 1)
    :return:
    """
    obj_dir = os.path.abspath(obj_dir)

    scene_dir = os.path.abspath(scene_dir)

    og_model = os.path.abspath(model_dir)
    darknet = os.path.abspath(darknet_path)

    save_dir = os.path.abspath(save_dir)

    m_path = {
        "model": f"{save_dir}/{model_name}/model",
        "settings": f"{save_dir}/{model_name}/data",
        "images": f"{save_dir}/{model_name}/images",
        "weights": f"{save_dir}/{model_name}/model/weights"
    }
    # Create directories
    for value in m_path.values():
        Path(value).mkdir(parents=True, exist_ok=True)

    object_names, img_names = data_gen(obj_dir, scene_dir, m_path['images'])

    # Create the yolo.cfg
    with open(f"{m_path['model']}/{model_name}.cfg", "w") as cfg:
        cfg.write(yoloGenerator.yolov3_tiny(len(object_names)))

    # Make the custom.names file
    with open(f"{m_path['settings']}/custom.names", "w") as names:
        for obj in object_names:
            names.write(f"{obj}\n")

    # Distribute the train and test images
    with open(f"{m_path['settings']}/train.txt", "w") as train:
        random.shuffle(img_names)
        for i in range(int(len(img_names) * data_distribution)):
            train.write(f"{m_path['images']}/{img_names.pop()}\n")

    with open(f"{m_path['settings']}/test.txt", "w") as train:
        for i in img_names:
            train.write(f"{m_path['images']}/{i}\n")

    # YOLO datafile
    with open(f"{m_path['settings']}/detector.data", "w") as data:
        data.write(f"classes={len(object_names)}\n")
        data.write(f"train={m_path['settings']}/train.txt\n")
        data.write(f"valid={m_path['settings']}/test.txt\n")
        data.write(f"names={m_path['settings']}/custom.names\n")
        data.write(f"backup={m_path['model']}/weights/\n")

    # Get pre-trained weights using the stock info
    os.system(f"{darknet} partial {og_model}/yolov3-tiny.cfg {og_model}/yolov3-tiny.weights {m_path['model']}/{model_name}.conv.15 15")

    # Train the model
    os.system(f"{darknet} detector train {m_path['settings']}/detector.data {m_path['model']}/{model_name}.cfg {m_path['model']}/{model_name}.conv.15 > {m_path['model']}/train.log")


if __name__ == "__main__":
    # train(r"data/objects", r"data/scenes", r"data/yolov3-tiny", r"../darknet/darknet", r"data/models")

    import argparse
    import datetime

    parser = argparse.ArgumentParser(description='Image dataset generator and YOLO model trainer.')

    parser.add_argument('object_path', type=str, help="Where the object images are stored.")
    parser.add_argument('scene_path', type=str, help="Where the scene images are stored.")
    parser.add_argument('yolo_path', type=str, help="Where the yolo.cfg and .conv files are stored. "
                                                    "[YOLOV3-tiny is currently the only suported model]")
    parser.add_argument('darknet', type=str, help="Darknet script.")
    parser.add_argument('save_path', type=str, help="Where the model will be saved.")
    parser.add_argument('--m', type=str, help="The model name, if none selected a date of creation will be provided.")
    parser.add_argument('--d', type=float, help="Data distribution.")

    args = parser.parse_args()

    # If model name empty
    model_name = args.m
    if model_name is None:
        model_name = datetime.datetime.now().strftime("%Y-%m-%dT%H")

    data_distrib = args.d
    if data_distrib is None:
        data_distrib = 0.75

    train(args.object_path, args.scene_path, args.yolo_path, args.darknet, args.save_path, model_name, data_distrib)
