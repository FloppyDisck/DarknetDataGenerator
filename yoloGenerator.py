

def yolov3_tiny(classes):
    """
    Create a yolov3-tiny configuration file
    :param classes: The number of classes in your config
    :return: A text configuration
    """
    return (
            net(classes) +
            convolutional(1, 16, 3, "leaky") +
            maxpool(2, 2) +
            convolutional(1, 32, 3, "leaky") +
            maxpool(2, 2) +
            convolutional(1, 64, 3, "leaky") +
            maxpool(2, 2) +
            convolutional(1, 128, 3, "leaky") +
            maxpool(2, 2) +
            convolutional(1, 256, 3, "leaky") +
            maxpool(2, 2) +
            convolutional(1, 512, 3, "leaky") +
            maxpool(2, 1) +
            convolutional(1, 1024, 3, "leaky") +
            convolutional(1, 256, 1, "leaky") +
            convolutional(1, 512, 3, "leaky") +
            convolutional(0, (classes + 5) * 3, 1, "linear") +
            yolo_layer("3,4,5", classes) +
            route("-4") +
            convolutional(1, 128, 1, "leaky") +
            upsample(2) +
            route("-1, 8") +
            convolutional(1, 256, 3, "leaky") +
            convolutional(0, (classes + 5) * 3, 1, "linear") +
            yolo_layer("0,1,2", classes)
    )


def net(class_qty):
    # max_batches cannot be lower than 6000
    max_batches = class_qty * 2000 if class_qty >= 3 else 6000

    return (
        "[net]\n"
        "batch=64\n"
        "subdivisions=16\n"
        "width=416\n"
        "height=416\n"
        "channels=3\n"
        "momentum=0.9\n"
        "decay=0.0005\n"
        "angle=0\n"
        "saturation = 1.5\n"
        "exposure = 1.5\n"
        "hue=.1\n"
        "learning_rate=0.001\n"
        "burn_in=400\n"
        f"max_batches = {max_batches}\n"
        "policy=steps\n"
        f"steps={int(max_batches * 0.8)},{int(max_batches * 0.9)}\n"
        "scales=.1,.1\n"
    )


def convolutional(batch_normalize, filters, size, activation):
    batch_str = "" if batch_normalize == 0 else f"batch_normalize={batch_normalize}"
    filters_str = "" if filters == 0 else f"filters={filters}"
    size_str = "" if size == 0 else f"size={size}"

    return (
        "[convolutional]\n"
        f"{batch_str}\n"
        f"{filters_str}\n"
        f"{size_str}\n"
        "stride=1\n"
        "pad=1\n"
        f"activation={activation}\n"
    )


def maxpool(size, stride):
    return (
        "[maxpool]\n"
        f"size={size}\n"
        f"stride={stride}\n"
    )


def yolo_layer(mask, classes):
    return (
        "[yolo]\n"
        f"mask = {mask}\n"
        "anchors = 10,14, 23,27, 37,58, 81,82, 135,169, 344,319\n"
        f"classes={classes}\n"
        "num=6\n"
        "jitter=.3\n"
        "ignore_thresh = .7\n"
        "truth_thresh = 1\n"
        "random=1\n"
    )


def route(layers):
    return (
        "[route]\n"
        f"layers = {layers}\n"
    )


def upsample(stride):
    return (
        "[upsample]\n"
        f"stride={stride}\n"
    )

    # Make custom yolo file
    # Batch=64, subdivisions=16, max_batches = classes*2000 but no less than 6000
    # steps to 80% and 90% of max_batches,
    # line classes=80 to your number of objects in each of 3 [yolo]-layers
    # change [filters=255] to filters=(classes + 5)x3 in the 3 [convolutional] before each [yolo] layer,
    # keep in mind that it only has to be the last [convolutional] before each of the [yolo] layers
    # when using[Gaussian_yolo] layers, change[filters = 57] filters = (classes + 9)x3 in the [convolutional] before each[Gaussian_yolo] layer


if __name__ == "__main__":
    with open("data/custom_data/cfg/yolov3-tiny-generated.cfg", 'w') as cfg:
        cfg.write(yolov3_tiny(1))
