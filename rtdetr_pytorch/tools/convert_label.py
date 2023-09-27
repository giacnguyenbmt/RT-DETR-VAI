import os
import json

import tqdm
import cv2
import pandas as pd

def vai2coco(src_img_dir, src_gt_path, index_col=None):
    df = pd.read_csv(src_gt_path, index_col=index_col)
    
    categories_dict = {}
    for i in range(len(df)):
        record_ = df.iloc[i]
        c_id = record_['class_id']
        c_name = record_['class_name']
        if categories_dict.get(c_id, None) is None:
            categories_dict[c_id] = c_name
    categories_list = sorted(list(categories_dict.items()), key=lambda x: x[0])
    categories = [
        {
            "id": int(idx), "name": value[1], "supercategory": value[1],
        } for idx, value in enumerate(categories_list)
    ]
    categories_dict = {
        value[1]: idx for idx, value in enumerate(categories_list) 
    }

    annotations = []
    images_dict = {}
    images = []
    for i in tqdm.trange(len(df)):
        record_ = df.iloc[i]
        c_name = record_['class_name']

        img_path = os.path.join(src_img_dir, record_['image_id'] + '.jpg')
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
        bbox = [
            int(record_['x_min']),
            int(record_['y_min']),
            int(record_['x_max'] - record_['x_min']),
            int(record_['y_max'] - record_['y_min']),
        ]

        if images_dict.get(record_['image_id'], None) is None:
            images_dict[record_['image_id']] = len(images_dict)
        
        images.append(
            {
                "id": int(images_dict[record_['image_id']]),
                "file_name": record_['image_id'] + '.jpg',
                "height": int(img_h),
                "width": int(img_w),
            }
        )
            
        annotations.append(
            {
                "id": int(i),
                "image_id": images_dict[record_['image_id']],
                "category_id": int(categories_dict[c_name]),
                "bbox": bbox,
                "area": int(bbox[-1] * bbox[-2]),
                "segmentation": [],
                "iscrowd": 0
            }
        )

    coco = {
        'info': {
            'date_created': '2023-09-23T00:00:00+00:00'
        },
        'images': images,
        'annotations': annotations,
        'licenses': [],
        'categories': categories
    }
    dst_path = os.path.splitext(src_gt_path)[0] + '.json'
    json_object = json.dumps(coco, indent=4)
    with open(dst_path, "w") as outfile:
        outfile.write(json_object)
    return coco

if __name__ == '__main__':
    src_img_dir = 'det/warmup/train/images'
    src_gt_path = 'det/warmup/train/groundtruth.csv'
    vai2coco(src_img_dir, src_gt_path, index_col=0)

    src_img_dir = 'det/public/train/images'
    src_gt_path = 'det/public/train/groundtruth.csv'
    vai2coco(src_img_dir, src_gt_path, index_col=None)
