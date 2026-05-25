from PIL import Image, ImageDraw
from tqdm import tqdm
import json
import re
import os
import argparse
import numpy as np
import math
from scipy.optimize import linear_sum_assignment

def draw_image(path, bboxes, gt_bboxes, output_path):
    image = Image.open(path)
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        draw.rectangle(bbox, outline="red", fill=None)
    for gt_bbox in gt_bboxes:
        draw.rectangle(gt_bbox, outline="green", fill=None)
    image.save(output_path)
    return 

def clean_prediction_symbols(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("-", " ")

    s = re.sub(r"(?<!\w):(?=\w)", " ", s)
    s = re.sub(r"(?<=\w):(?!\w)", " ", s)

    s = re.sub(r"(?<!\w)[^\w\s:]+(?=\w)", "", s)
    s = re.sub(r"(?<=\w)[^\w\s:]+(?!\w)", "", s)

    s = re.sub(r"\s+", " ", s).strip()

    return s

# This is the resize function of Qwen2.5-VL
def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

def extract_bbox(text):
    # r'\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]' for [[28, 586, 362, 793]]
    # r'<box>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)</box>' for <box>826 704 993 984</box>
    # r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]' for [28, 586, 362, 793]
    # r'\[(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+)\]' for [0.3, 0.2, 0.4, 0.28]
    matches = re.findall(r'\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)', text)
    if matches == []:
        matches = re.findall(r'\[(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+)\]', text)
    if matches == []:
        matches = re.findall(r'\((\d+\.\d+),(\d+\.\d+)\),\((\d+\.\d+),(\d+\.\d+)\)', text)
    if matches == []:
        matches = re.findall(r'\((\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+)\)', text)
    if matches == []:
        matches = re.findall(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', text)
    if matches == []:
        matches = re.findall(r'<box>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)</box>', text)
    boxes = [[float(val) for val in match] for match in matches] 
    return boxes

def calculate_iou(box1, box2):
    if box2 == []:
        return 0.0
    if isinstance(box2[0], list):
        box2 = box2[0]
    try:
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
    except:
        print(f'[Notice]Error when calculating IOU:{box1},{box2}')
        return 0.0
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area
    return iou

def compute_miou(gt_boxes, pred_boxes):
    """
    Convert to absolute coordinates when needed.
    :param gt_boxes: gt bounding boxes [[x1,y1,x2,y2], ...]
    :param pred_boxes: pred bounding boxes [[x1,y1,x2,y2], ...]
    :return: mIoU
    """

    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 1.0
    elif len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return 0.0

    # IoU matrix construction
    iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, gt_box in enumerate(gt_boxes):
        for j, pred_box in enumerate(pred_boxes):
            iou_matrix[i, j] = calculate_iou(gt_box, pred_box)
    
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_ious = iou_matrix[row_ind, col_ind]

    matched_gt_count = len(matched_ious)
    unmatched_gt_count = len(gt_boxes) - matched_gt_count
    total_ious = np.sum(matched_ious)

    return (total_ious + unmatched_gt_count * 0.0) / len(gt_boxes)

def parse_args():
    parser = argparse.ArgumentParser(description="Run unified single/multi-process inference with vLLM for Qwen-VL")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input dataset JSON file")
    parser.add_argument("--pred_type", type=str, required=True, help="Type of prediction task")
    parser.add_argument("--miou_threshold", type=float, default=0.5, help='miou threshold')
    return parser.parse_args()

def main():
    args = parse_args()
    print(args.data_path)
    predict_texts = json.load(open(args.data_path, 'r'))
    pred_type = args.pred_type
    assert pred_type in ['grounding', 'vqa'], 'not implemented yet.'

    correct, total_miou, total_count = 0, 0, 0
    
    for predict_text in tqdm(predict_texts):        
        # gt
        ground_truth = predict_text.get('ground_truth')
        # predict
        try: 
            len(predict_text['content'][0].split('answer')) > 1
        except:
            print(predict_text['content'][0])
        pred = predict_text['content'][0]

        if pred_type == 'grounding':
            answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
            
            try:
                answer_match = answer_pattern.search(pred).group(1).strip()
            except:
                continue
            
            json_pattern = re.compile(r'```json\s*([\s\S]*?)\s*```')
            try:
                json_match = json_pattern.search(answer_match).group(1).strip()
            except:
                continue

            predict = extract_bbox(json_match)
            miou = compute_miou(ground_truth, predict)
        
            total_miou+=miou
            if miou > args.miou_threshold:
                correct += 1

        else: # vqa
            gt = ground_truth.strip().lower()
            pred = pred.strip().lower()
            pred = clean_prediction_symbols(pred)
            gt = clean_prediction_symbols(gt)
            gt_pattern = rf"\b{re.escape(gt)}\b"
            pred_pattern = rf"\b{re.escape(pred)}\b"

            if re.search(gt_pattern, pred, flags=re.IGNORECASE) is not None:
                correct += 1
            elif re.search(pred_pattern, gt, flags=re.IGNORECASE) is not None:
                correct += 1
        total_count += 1

    
    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(total_count, 100. * correct/total_count))
    if pred_type == 'grounding':
        print('Avg miou: {:.3f}\n'.format(total_miou/total_count))

if __name__ == "__main__":
    main()