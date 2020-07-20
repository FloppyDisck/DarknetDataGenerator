import os
import cv2
import random
import cvutils


def scene_gen(scene_dir, random_colors=20):
    """
    Generate solid color images in 1080p res
    :param scene_dir: Where the images will be saved.
    :param random_colors: The qty of images with random colors to be generated.
    """
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
              (255, 0, 255), (0, 255, 255), (255, 255, 255), (0, 0, 0)]

    for r in range(random_colors):
        colors.append((random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)))

    for i in range(len(colors)):
        image = cvutils.create_blank(1920, 1080, colors[i])
        name = "generatedImage{}.jpg".format(i)
        cv2.imwrite(os.path.join(scene_dir, name), image)


if __name__ == "__main__":
    # train(r"../data/objects", r"../data/scenes", r"../data/yolov3-tiny", r"../darknet/darknet", r"../data/models")

    import argparse

    parser = argparse.ArgumentParser(description='Scene image generator.')

    parser.add_argument('scene_path', type=str,
                        help="Where the scene images are stored.")

    parser.add_argument('qty', type=int,
                        help="Amount of scenes to be generated.")

    args = parser.parse_args()

    scene_gen(os.path.abspath(args.scene_path), random_colors=args.qty)
