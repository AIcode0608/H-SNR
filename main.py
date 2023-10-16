from __future__ import print_function
import os.path as osp

import click
import cv2
import matplotlib.cm as cm
import numpy as np
import torch
from torchvision import models, transforms

from grad_cam import (
    BackPropagation,
    Deconvnet,
    GradCAM,
    GuidedBackPropagation,
    occlusion_sensitivity,
)


# if a model includes LSTM, such as in image captioning,
# torch.backends.cudnn.enabled = False


def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def load_images(image_paths):
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path)
        images.append(image)
        raw_images.append(raw_image)
    return images, raw_images


def get_classtable():
    classes = []
    with open("samples/synset_words.txt") as lines:
        for line in lines:
            line = line.strip().split(" ", 1)[1]
            line = line.split(", ", 1)[0].replace(" ", "_")
            classes.append(line)
    return classes


def preprocess(image_path):
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (224,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image


def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, np.uint8(gradient))


def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))


def save_H_SNR(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.bwr_r(gcam)[..., :3]
    cmap = np.uint8(cmap)
    hsv = cv2.cvtColor(cmap * 255, cv2.COLOR_BGR2HSV)  # BGR to HSV
    low_hsv = np.array([0, 43, 46])  # Fill in the three min values based on the HSV table below
    high_hsv = np.array([10, 255, 255])  # Fill in the three max values
    mask = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)

    cv2.imwrite(filename, mask)
    img = cv2.imread(filename)
    canny = cv2.Canny(img, 0, 255)
    indices = np.where(canny != [0])
    coordinates = zip(indices[1], indices[0])
    right_x = 0
    right = (0, 0)
    left_x = 1000
    left = (0, 0)
    down_y = 0
    top = (0, 0)
    k = 0
    for i in coordinates:
        # print(i)
        if k == 0:
            top = i
        if right_x < i[0]:
            right_x = i[0]
            right = i
        if left_x > i[0]:
            left_x = i[0]
            left = i
        down = i
        k = k + 1
    l, t, r, d = left[0], top[1], right[0], down[1]
    if l != 0:
        l = l - 5
        if l < 0:
            l = 0
    if t != 0:
        t = t - 5
        if t < 0:
            t = 0
    if r != 224:
        r = r + 5
        if r > 224:
            r = 224
    if d != 224:
        d = d + 5
        if d > 224:
            d = 224
    imgCrop = raw_image[t:d, l:r]

    cv2.imwrite(filename, imgCrop)


def save_sensitivity(filename, maps):
    maps = maps.cpu().numpy()
    scale = max(maps[maps > 0].max(), -maps[maps <= 0].min())
    maps = maps / scale * 0.5
    maps += 0.5
    maps = cm.bwr_r(maps)[..., :3]
    maps = np.uint8(maps * 255.0)
    maps = cv2.resize(maps, (224, 224), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(filename, maps)


# torchvision models
model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


@click.group()
@click.pass_context
def main(ctx):
    print("Mode:", ctx.invoked_subcommand)

@main.command()
@click.option("-i", "--image-paths", type=str, multiple=True, required=True)
@click.option("-a", "--arch", type=click.Choice(model_names), required=True)
@click.option("-t", "--target-layer", type=str, required=True)
@click.option("-k", "--topk", type=int, default=1)
@click.option("-o", "--output-dir", type=str, default="./results")
@click.option("--cuda/--cpu", default=True)
def demo1(image_paths, target_layer, arch, topk, output_dir, cuda):
    """
    Visualize model responses given multiple images
    """

    device = get_device(cuda)

    # Synset words
    classes = get_classtable()

    # Model from torchvision
    model = models.__dict__[arch](pretrained=True)
    model.to(device)
    model.eval()

    # Images
    images, raw_images = load_images(image_paths)
    images = torch.stack(images).to(device)

    print("Vanilla Backpropagation:")

    bp = BackPropagation(model=model)
    probs, ids = bp.forward(images)  # sorted

    bp.remove_hook()

    print("Deconvolution:")

    deconv = Deconvnet(model=model)
    _ = deconv.forward(images)

    deconv.remove_hook()

    gcam = GradCAM(model=model)
    _ = gcam.forward(images)

    gbp = GuidedBackPropagation(model=model)
    _ = gbp.forward(images)

    for i in range(topk):
        # Guided Backpropagation
        gbp.backward(ids=ids[:, [i]])
        gradients = gbp.generate()

        # Grad-CAM
        gcam.backward(ids=ids[:, [i]])
        regions = gcam.generate(target_layer=target_layer)

        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            # Grad-CAM
            # save_gradcam(
            #     filename=osp.join(
            #         output_dir,
            #         "{}-{}-gradcam-{}-{}.png".format(
            #             j, arch, target_layer, classes[ids[j, i]]
            #         ),
            #     ),
            #     gcam=regions[j, 0],
            #     raw_image=raw_images[j],
            # )
            # H_SNR
            save_H_SNR(
                filename=osp.join(
                    output_dir,
                    "{}-{}-gradcam-{}-{}.png".format(
                        j, arch, target_layer, classes[ids[j, i]]
                    ),
                ),
                gcam=regions[j, 0],
                raw_image=raw_images[j],
            )



