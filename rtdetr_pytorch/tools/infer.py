"""by nguyenpdg
"""

import os 
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import glob

import cv2
import numpy as np
import pandas as pd
import tqdm
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor

from src.core import YAMLConfig


def infer(model, images_list, input_size=(640, 640), threshold=-1):
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)
    model.eval()
    tensor_size = torch.tensor([list(input_size)]).to(device)
    results = []

    for img_path in tqdm.tqdm(images_list):
        img = cv2.imread(img_path)
        img =  cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
        ih, iw, _ = img.shape

        det_scale = [
            float(iw) / input_size[0],
            float(ih) / input_size[1],
        ]
        img = cv2.resize(img, input_size)
        tensor_input = ToTensor()(img)[None].to(device)

        t_labels, t_boxes, t_scores = model(tensor_input, tensor_size)

        labels = t_labels.detach().cpu().numpy()[0]
        boxes = t_boxes.detach().cpu().numpy()[0]
        scores = t_scores.detach().cpu().numpy()[0]

        idx = scores > threshold

        restore_boxes = boxes[idx].copy()
        restore_boxes[:, 0::2] *= det_scale[0]
        restore_boxes[:, 1::2] *= det_scale[1]

        result = {
            'labels': labels[idx],
            'boxes': boxes[idx],
            'restore_boxes': restore_boxes,
            'scores': scores[idx],
        }
        

        results.append(result)

    return results


def convert2vai(images_path_list, results, dst_path):
    data = []
    for i in range(len(images_path_list)):
        img_path = images_path_list[i]
        result = results[i]

        image_id = os.path.splitext(os.path.split(img_path)[-1])[0]
        
        if len(result['labels']) > 0:
            for j in range(len(result['labels'])):
                class_id = int(result['labels'][j])
                if class_id > 10:
                    class_id += 1
                confidence_score = float(result['scores'][j])
                x_min, y_min, x_max, y_max = list(
                    map(int, list(result['restore_boxes'][j]))
                )
                record_ = {
                    'image_id': image_id,
                    'class_id': class_id,
                    'confidence_score': confidence_score,
                    'x_min': x_min,
                    'y_min': y_min,
                    'x_max': x_max,
                    'y_max': y_max,
                }
                data.append(record_)
        else:
            record_ = {
                'image_id': image_id,
                'class_id': 15,
                'confidence_score': 0.0,
                'x_min': 0,
                'y_min': 0,
                'x_max': 0,
                'y_max': 0,
            }
            data.append(record_)
    df = pd.DataFrame.from_records(data)

    output_dir = os.path.split(dst_path)[0]
    if os.path.isdir(output_dir) is False:
        os.makedirs(output_dir)
    df.to_csv(dst_path)
    return df


def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('only support resume to load model.state_dict by now.')

    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)
    
    model = Model()
    model.eval()

    images_list = glob.glob(os.path.join(args.images_dir, '*.jpg'))
    images_list.sort()

    results = infer(model, images_list, input_size=(640, 640), threshold=-1.0)
    df = convert2vai(images_list, results, args.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--images_dir', '-i', type=str, default=None)
    parser.add_argument('--output_path', '-o', type=str, default='output/output.csv')

    args = parser.parse_args()

    main(args)
