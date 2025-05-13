import sys
import torch
import argparse

import openvino as ov

from onnx.modnet_onnx import MODNet

MODNET_MODEL_PATH = {
    "webcam": "./ov_models/modnet_webcam_portrait_matting_{}_{}.xml",
    "photographic": "./ov_models/modnet_photographic_portrait_matting_{}_{}.xml"
}

MODNET_WEIGHT_PATH = {
    "webcam": "./pretrained/modnet_webcam_portrait_matting.ckpt",
    "photographic": "./pretrained/modnet_photographic_portrait_matting.ckpt"
}

def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', 
                      help='Show this help message and exit.')
    args.add_argument('-ih', '--input_height', type = int, default = 512,
                      help='Optional. Height of input.')
    args.add_argument('-iw', '--input_width', type = int, default = 512,
                      help='Optional. Width of input.')
    args.add_argument('-mt', '--model_type', type=str, default="webcam",
                      help='path of the OpenVINO model')

    return parser.parse_args()


def create_modnet(weights):
    model = MODNet(backbone_pretrained=False)

    checkpoint = torch.load(weights, map_location='cpu')
    ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    model.load_state_dict(ckpt)

    return model

def main():
    args = parse_args()

    model_type = args.model_type
    HEIGHT = args.input_height
    WIDTH = args.input_width

    ov_model_path = MODNET_MODEL_PATH[model_type].format(WIDTH, HEIGHT)
    
    modnet = create_modnet(MODNET_WEIGHT_PATH[model_type])

    example_input = torch.randn(1, 3, HEIGHT, WIDTH)
    ov_model = ov.convert_model(modnet, example_input=example_input, input=[1, 3, HEIGHT, WIDTH])

    ov.save_model(ov_model, ov_model_path)

if __name__ == "__main__":
    sys.exit(main())