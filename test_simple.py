from __future__ import absolute_import, division, print_function

import argparse
import glob
import os

import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import PIL.Image as pil
import torch
from torchvision import transforms

import networks
from layers import disp_to_depth


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple testing function for MonoPCC models.")
    parser.add_argument(
        "--image_path",
        type=str,
        help="path to a test image or folder of images",
        required=True,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="path to the model weights",
        required=True,
    )
    parser.add_argument(
        "--ext",
        type=str,
        help="image extension to search for in a folder",
        default="png",
    )
    parser.add_argument(
        "--no_cuda",
        help="if set, disables CUDA",
        action="store_true",
    )
    return parser.parse_args()


def test_simple(args):
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_path = os.path.expanduser(args.model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    print("-> Loading model from {}".format(model_path))

    encoder_dict = torch.load(encoder_path, map_location=device)
    feed_height = encoder_dict["height"]
    feed_width = encoder_dict["width"]

    encoder = networks.mpvit_small()
    model_dict = encoder.state_dict()
    encoder.load_state_dict({
        key: value for key, value in encoder_dict.items()
        if key in model_dict
    })
    encoder.to(device)
    encoder.eval()

    depth_decoder = networks.DepthDecoder()
    depth_decoder.load_state_dict(
        torch.load(depth_decoder_path, map_location=device))
    depth_decoder.to(device)
    depth_decoder.eval()

    image_path = os.path.expanduser(args.image_path)
    if os.path.isfile(image_path):
        paths = [image_path]
        output_directory = os.path.dirname(image_path)
    elif os.path.isdir(image_path):
        paths = sorted(glob.glob(os.path.join(
            image_path, "*.{}".format(args.ext))))
        output_directory = image_path
    else:
        raise FileNotFoundError(
            "Cannot find image path {}".format(image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    with torch.no_grad():
        for index, path in enumerate(paths):
            if path.endswith("_disp.jpeg"):
                continue

            input_image = pil.open(path).convert("RGB")
            original_width, original_height = input_image.size
            input_image = input_image.resize(
                (feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            output = depth_decoder(encoder(input_image.to(device)))
            disparity = output[("disp", 0)]
            disparity_resized = torch.nn.functional.interpolate(
                disparity,
                (original_height, original_width),
                mode="bilinear",
                align_corners=False,
            )

            output_name = os.path.splitext(os.path.basename(path))[0]
            output_npy = os.path.join(
                output_directory, "{}_disp.npy".format(output_name))
            scaled_disparity, _ = disp_to_depth(disparity, 0.1, 150)
            np.save(output_npy, scaled_disparity.cpu().numpy())

            disparity_image = disparity_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disparity_image, 95)
            normalizer = mpl.colors.Normalize(
                vmin=disparity_image.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap="magma")
            color_disparity = (
                mapper.to_rgba(disparity_image)[:, :, :3] * 255
            ).astype(np.uint8)

            output_image = os.path.join(
                output_directory, "{}_disp.jpeg".format(output_name))
            pil.fromarray(color_disparity).save(output_image)

            print("   Processed {:d} of {:d} images".format(
                index + 1, len(paths)))
            print("   - {}".format(output_image))
            print("   - {}".format(output_npy))

    print("-> Done!")


if __name__ == "__main__":
    test_simple(parse_args())
