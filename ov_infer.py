import os
import cv2
import sys
import logging
import argparse

import numpy as np
import openvino as ov

from pymediainfo import MediaInfo

logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout) 
log = logging.getLogger()

MODNET_MODEL_PATH = {
    "webcam": "./ov_models/modnet_webcam_portrait_matting_{}_{}.xml",
    "photographic": "./ov_models/modnet_photographic_portrait_matting_{}_{}.xml"
}

def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', 
                      help='Show this help message and exit.')
    args.add_argument('-i', '--input_name', type=str, required=True, 
                      help='path of the input image/video (a file)')
    args.add_argument('-o', '--output_dir', type=str, default="output_imgs/",
                      help='paht for saving the predicted alpha matte (a file)')
    args.add_argument('-ih', '--input_height', type = int, default = 512,
                      help='Optional. Height of input.')
    args.add_argument('-iw', '--input_width', type = int, default = 512,
                      help='Optional. Width of input.')
    args.add_argument('-mt', '--model_type', type=str, default="webcam",
                      help='path of the OpenVINO model')

    return parser.parse_args()

def get_scale_factor(im_h, im_w, ref_size=512):
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w

    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32

    x_scale_factor = im_rw / im_w
    y_scale_factor = im_rh / im_h

    return x_scale_factor, y_scale_factor

def img_preprocessing(img):
    if len(img.shape) == 2:
        img = img[:, :, None]
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.shape[2] == 4:
        img = img[:, :, 0:3]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return (img - 127.5) / 127.5

def infer(compiled_model, img, img_w, img_h, input_width, input_height):
    result = compiled_model(np.expand_dims(
        np.transpose(cv2.resize(img_preprocessing(img), (input_width, input_height)), (2, 0, 1)), 0))[0]
    matte = (np.squeeze(result) * 255).astype('uint8')
    matte = cv2.resize(matte, dsize=(img_w, img_h), interpolation = cv2.INTER_AREA)

    output_image = np.where(np.stack((matte,) * 3, axis=-1) > 200, img, cv2.cvtColor(matte, cv2.COLOR_GRAY2RGB))

    return output_image

def main():
    args = parse_args()

    input_name = args.input_name
    model_type = args.model_type
    HEIGHT = args.input_height
    WIDTH = args.input_width
    
    ov_model_path = MODNET_MODEL_PATH[model_type].format(WIDTH, HEIGHT)

    output_path = (
        args.output_dir + input_name.split('/')[-1].split('.')[0] + 
        "_{}_{}_{}.{}".format(model_type, WIDTH, HEIGHT, input_name.split('/')[-1].split('.')[-1]))

    core = ov.Core()
    compiled_model = core.compile_model(ov_model_path, "GPU")

    m_info = MediaInfo.parse(input_name)
    for track in m_info.tracks:
        if track.track_type == "Video":
            cap = cv2.VideoCapture(input_name)
            v_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            v_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 30.0, (v_width,  v_height))
            while cap.isOpened():
                ret, v_frame = cap.read()
                if not ret:
                    break
                matte = infer(compiled_model, v_frame, v_width, v_height, WIDTH, HEIGHT)
                out.write(matte)

            cap.release()
            out.release()

        elif track.track_type == "Image":
            img = cv2.imread(input_name)
            img_h, img_w, _ = img.shape
            matte = infer(compiled_model, img, img_w, img_h, WIDTH, HEIGHT)

            cv2.imwrite(output_path, matte)

    log.info ("Result is saved to '%s'.", output_path)

if __name__ == "__main__":
    sys.exit(main())