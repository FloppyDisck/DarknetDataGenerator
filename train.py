import os
import cv2
import random
import yoloGenerator
import cvutils
from pathlib import Path

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
        obj_img = cv2.imread(os.path.join(obj_dir, obj))

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
                scene_img = cv2.imread(os.path.join(scene_dir, scene))
                scene_y, scene_x = scene_img.shape[:2]

                x_off = random.randrange(0, scene_x - new_obj.shape[1])
                y_off = random.randrange(0, scene_y - new_obj.shape[0])

                # Get both normal and flipped image
                save_image = cvutils.combine(new_obj, scene_img, x_off, y_off)

                file_name = object_name + str(file_count)
                image_name = file_name + ".jpg"

                img_names.append(image_name)

                cv2.imwrite(os.path.join(output_dir, image_name), save_image)

                # Create the bounding box information

                with open(os.path.join(output_dir, file_name+".txt"), "w") as bounding:
                    bounding.write(
                        "{} {} {} ".format(object_names.index(object_name), (x_off + new_obj.shape[1] / 2) / scene_x, (y_off + new_obj.shape[0] / 2) / scene_y) +
                        "{} {}".format(new_obj.shape[1] / scene_x, new_obj.shape[0] / scene_y))

                file_count += 1

    return [object_names, img_names]


def train(obj_dir, scene_dir, model_dir, darknet_path, save_dir, model_name="latest", data_distribution=0.75,
          populate_scenes=True):
    """
    Generates a dataset out of the given images
    :param obj_dir: Path for the object images $objectname#.jpg
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
    model_path = os.path.join(save_dir, model_name)
    m_path = {
        "model": os.path.join(model_path, "model"),
        "settings": os.path.join(model_path, "data"),
        "images": os.path.join(model_path, "images"),
        "weights": os.path.join(model_path, "model/weights")
    }
    # Create directories
    for value in m_path.values():
        Path(value).mkdir(parents=True, exist_ok=True)

    # Generate training images
    object_names, img_names = data_gen(obj_dir, scene_dir, m_path['images'])

    # Create the yolo.cfg
    with open(os.path.join(m_path['model'], model_name+".cfg"), "w") as cfg:
        cfg.write(yoloGenerator.yolov3_tiny(len(object_names)))

    # Make the custom.names file
    with open(os.path.join(m_path['settings'], "custom.names"), "w") as names:
        for obj in object_names:
            names.write(obj+"\n")

    # Distribute the train and test images
    with open(os.path.join(m_path['settings'], "train.txt"), "w") as train:
        random.shuffle(img_names)
        for i in range(int(len(img_names) * data_distribution)):
            train.write(os.path.join(m_path['images'], img_names.pop()) + "\n")

    with open(os.path.join(m_path['settings'], "test.txt"), "w") as test:
        # Go through the remaining images
        for i in img_names:
            test.write(os.path.join(m_path['images'], i) + "\n")

    # YOLO datafile
    with open(os.path.join(m_path['settings'], "detector.data"), "w") as data:
        data.write("classes={}\n".format(len(object_names)))
        data.write("train={}\n".format(os.path.join(m_path['settings'], "train.txt")))
        data.write("valid={}\n".format(os.path.join(m_path['settings'], "test.txt")))
        data.write("names={}\n".format(os.path.join(m_path['settings'], "custom.names")))
        data.write("backup={}\n".format(os.path.join(m_path['model'], "weights/")))

    model_name_path = os.path.join(m_path['model'], model_name)

    # Get pre-trained weights using the stock info
    os.system(
        "{0} partial {1}/yolov3-tiny.cfg {1}/yolov3-tiny.weights {2}.conv.15 15".format(darknet, og_model, model_name_path))

    # Train the model
    os.system(
        "{0} detector train {1}/detector.data {2}.cfg {2}.conv.15 -gpus 0 -map > {3}/train.log".format(darknet, m_path['settings'], model_name_path, m_path['model']))


if __name__ == "__main__":
    import argparse
    import datetime

    parser = argparse.ArgumentParser(description='Image dataset generator and YOLO model trainer.')

    parser.add_argument('object_path', type=str,
                        help="Where the object images are stored.")
    parser.add_argument('scene_path', type=str,
                        help="Where the scene images are stored.")
    parser.add_argument('yolo_path', type=str,
                        help="Where the yolo.cfg and .conv files are stored. "
                             "[YOLOV3-tiny is currently the only suported model]")
    parser.add_argument('darknet', type=str,
                        help="Darknet script.")
    parser.add_argument('save_path', type=str,
                        help="Where the model will be saved.")
    parser.add_argument('-m', '--model_name', type=str, default="yolo-model",
                        help="The model name, if none selected a date of creation will be provided.")
    parser.add_argument('-d', '--distribution', type=float, default=0.75,
                        help="Data distribution.")
    # parser.add_argument('-c', '--continue', action="store_true", help="Continue training from latest weights.")

    args = parser.parse_args()

    # If model name empty
    model_name = args.model_name
    if model_name is None:
        model_name = datetime.datetime.now().strftime("%Y-%m-%dT%H")

    data_distrib = args.distribution
    if data_distrib is None:
        data_distrib = 0.75

    train(args.object_path, args.scene_path, args.yolo_path, args.darknet, args.save_path, model_name, data_distrib)