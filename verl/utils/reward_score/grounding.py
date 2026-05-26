import re
import json
import torch
import numpy as np
from torchvision.ops.boxes import box_area
from scipy.optimize import linear_sum_assignment
import wandb
import os

bbox_lower_threshold = 0.3
format_pattern = r"<think>[\S\n\t\v ]*?</think>\s*<answer>[\S\n\t\v ]*?</answer>"
bbox_patterns = [
    re.compile(r'\((\d*?),.*?(\d*?)\),\((\d*?),(\d*?)\)'),
    re.compile(r'\[(\d*?), (\d*?), (\d*?), (\d*?)\]'),
    re.compile(r'\((\d*?), (\d*?), (\d*?), (\d*?)\)'),
    re.compile(r'\((\d*?), (\d*?)\)\n?.*?\((\d*?), (\d*?)\)'),
]

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def extract_think_content(solution_str):
    think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
    think_match = think_pattern.search(solution_str)
    return think_match is not None

def extract_answer_content(solution_str):
    answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
    answer_match = answer_pattern.search(solution_str)
    if answer_match:
        return answer_match.group(1).strip()
    return None

def extract_json_from_answer(answer_content):
    if answer_content is None:
        return None

    # Find JSON text enclosed in ```json and ```
    json_pattern = re.compile(r'```json\s*([\s\S]*?)\s*```')
    json_match = json_pattern.search(answer_content)
    
    if json_match:
        try:
            json_text = json_match.group(1).strip()
            # add for json type string
            json_text = json_text.replace("'bbox_2d'", '"bbox_2d"')
            json_text = json_text.replace("'[", "[").replace("]'", "]")
            json_data = json.loads(json_text)
            # Check the data type to ensure it is a list
            if not isinstance(json_data, list):
                return None
            
            for item in json_data:
                if not isinstance(item, dict):
                    return None
            return json_data
        except Exception as e:
            print(f"JSON parsing error: {e}")
            print(f"JSON text to be parsed: {json_text}")
    return None

def extract_pred_json_data(solution_str):
    # Try to extract from <answer> first
    answer_content = extract_answer_content(solution_str)
    
    # If <answer> exists, extract it from the tag's content; otherwise, extract it from the entire string
    search_content = answer_content if answer_content is not None else solution_str
    
    return extract_json_from_answer(search_content)

def extract_gt_json_data(solution_str):
    ''' 
    For convenient modification
    '''

    answer_content = extract_answer_content(solution_str)
    answer_content = json.loads(answer_content)
    if not isinstance(answer_content, list):
        return None
    # Check if the elements in the list are dictionaries
    for item in answer_content:
        if not isinstance(item, list):
            return None
    json_data = [{"bbox_2d": answer_item, "label": "person"} for answer_item in answer_content]
    return json_data

def format_reward(solution_str: str) -> float:
    """
    Format reward (maximum 0.75):
    - <think>: 0.25 points (exactly one pair, and content is not empty)
    - <answer>: 0.25 points (exactly one pair, and content is not empty)
    """

    reward = 0.0
    format_details = {}

    # Check <think> - ensure there is exactly one pair and that the content is not empty
    think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
    think_matches = think_pattern.findall(solution_str)
    if len(think_matches) == 1:
        think_content = think_matches[0].strip()
        if think_content:
            reward += 0.25
            format_details['think_format'] = 1
        else:
            format_details['think_format'] = 0
    else:
        format_details['think_format'] = 0
    
    # Check <answer> - ensure there is exactly one pair and that the content is not empty
    answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
    answer_matches = answer_pattern.findall(solution_str)
    answer_content = None
    if len(answer_matches) == 1:
        answer_content = answer_matches[0].strip()
        if answer_content:
            reward += 0.25
            format_details['answer_format'] = 1
        else:
            format_details['answer_format'] = 0
    else:
        format_details['answer_format'] = 0

    # Inspect  ```json``` code block - inspect only within the answer
    has_json_block = False
    if answer_content is not None:
        json_pattern = re.compile(r'```json\s*([\s\S]*?)\s*```')
        json_matches = json_pattern.findall(answer_content)
        if len(json_matches) == 1:
            has_json_block = True
    
    if has_json_block:
        reward += 0.25
        format_details['json_block_format'] = 1
    else:
        format_details['json_block_format'] = 0
    
    if wandb.run is not None:
        wandb.log({
            'sample_format/think_format': format_details['think_format'],
            'sample_format/answer_format': format_details['answer_format'], 
            'sample_format/json_block_format': format_details['json_block_format'],
            'sample_format/total_format_score': reward
        }, commit=False)
    
    return reward

def calculate_bbox_iou(pred_bbox, gt_bbox):
    """
    Calculate the IoU value between the two bounding boxes
    """
    
    if not isinstance(pred_bbox, list) or not isinstance(gt_bbox, list):
        return 0.0
        
    if len(pred_bbox) != 4 or len(gt_bbox) != 4:
        return 0.0
    
    try:
        gt_bbox_tensor = torch.tensor([gt_bbox], dtype=torch.float32)
        pred_bbox_tensor = torch.tensor([pred_bbox], dtype=torch.float32)
        
        iou, _ = box_iou(gt_bbox_tensor, pred_bbox_tensor)
        return iou.item()
    except Exception as e:
        return 0.0

def calculate_iou_reward(iou_value):
    if iou_value <=bbox_lower_threshold:
        return 0.0
    else:
        return iou_value

def calculate_quantity_penalty(n_pred, n_gt, penalty_factor=0.5):
    """
    Calculate the number of penalties to prevent reward hacking
    """
    if n_pred <= n_gt:
        return 0.0
    
    excess_ratio = (n_pred - n_gt) / n_gt
    penalty = min(penalty_factor * excess_ratio, 1.0)
    
    return penalty

def single_target_iou_reward(pred_json, gt_json):
    """
     bbox IoU reward
    """
    if len(pred_json) != 1 or len(gt_json) != 1:
        return 0.0
    
    pred_item = pred_json[0]
    gt_item = gt_json[0]
    
    if "bbox_2d" not in pred_item or "bbox_2d" not in gt_item:
        return 0.0
    
    iou_value = calculate_bbox_iou(pred_item["bbox_2d"], gt_item["bbox_2d"])
    iou_reward = calculate_iou_reward(iou_value)
    
    if wandb.run is not None:
        wandb.log({
            'sample_single_target/iou_value': iou_value,
            'sample_single_target/iou_reward': iou_reward
        }, commit=False)
    
    return iou_reward

def create_iou_matrix(pred_list, gt_list):
    n_pred = len(pred_list)
    n_gt = len(gt_list)
    
    iou_matrix = np.zeros((n_pred, n_gt))
    
    for i, pred in enumerate(pred_list):
        for j, gt in enumerate(gt_list):
            try:
                iou = calculate_bbox_iou(pred["bbox_2d"], gt["bbox_2d"])
            except:
                iou = 0.0
            iou_matrix[i, j] = iou

    return iou_matrix

def calculate_normalized_total_score(pred_list, gt_list, alpha=0.3):
    """
    Calculate the normalized total IoU reward score
    """
    
    # Create a global cost matrix
    iou_matrix = create_iou_matrix(pred_list, gt_list)
    
    # Find the optimal match using the Hungarian algorithm
    try:
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    except:
        row_ind, col_ind = [], []
    
    total_rewards = []
    matched_gt_indices = set()
    
    if len(row_ind)!=0:
        for i, j in zip(row_ind, col_ind):
            iou_value = iou_matrix[i, j]
            iou_reward = calculate_iou_reward(iou_value)
            
            total_rewards.append(iou_reward)
            matched_gt_indices.add(j)
    
    unmatched_gt_count = len(gt_list) - len(matched_gt_indices)
    for _ in range(unmatched_gt_count):
        total_rewards.append(0.0)
        
    raw_total_score = sum(total_rewards)

    if len(gt_list) == 1:
        normalized_score = raw_total_score
        difficulty_multiplier = 1.0
    else:
        base_normalized_score = (raw_total_score / len(gt_list))
        difficulty_multiplier = 1.0 + alpha * np.log(len(gt_list))
        
        normalized_score = base_normalized_score * difficulty_multiplier
    
    return normalized_score, raw_total_score, difficulty_multiplier

def multi_target_iou_reward(pred_json, gt_json, alpha=0.3, penalty_factor=0.5):
    valid_pred = []
    valid_gt = []
    
    for pred_item in pred_json:
        if (isinstance(pred_item, dict) and "bbox_2d" in pred_item and 
            isinstance(pred_item["bbox_2d"], list) and len(pred_item["bbox_2d"]) == 4):
            valid_pred.append(pred_item)
    
    for gt_item in gt_json:
        if (isinstance(gt_item, dict) and "bbox_2d" in gt_item and
            isinstance(gt_item["bbox_2d"], list) and len(gt_item["bbox_2d"]) == 4):
            valid_gt.append(gt_item)
    
    if len(valid_pred) == 0 or len(valid_gt) == 0:
        return 0.0
    
    n_pred = len(valid_pred)
    n_gt = len(valid_gt)
    
    normalized_score, raw_total_score, difficulty_multiplier = calculate_normalized_total_score(valid_pred, valid_gt, alpha)

    quantity_penalty = calculate_quantity_penalty(n_pred, n_gt, penalty_factor)
    
    final_score = max(0.0, normalized_score - quantity_penalty)
    
    if wandb.run is not None:
        iou_matrix = create_iou_matrix(valid_pred, valid_gt)
        total_iou = np.sum(iou_matrix[iou_matrix > 0])
        total_matches = np.sum(iou_matrix > 0)
        avg_iou = total_iou / total_matches if total_matches > 0 else 0.0
        
        wandb.log({
            'sample_multi_target/strategy': 'normalized_total_score',
            'sample_multi_target/n_pred': n_pred,
            'sample_multi_target/n_gt': n_gt,
            'sample_multi_target/raw_total_score': raw_total_score,
            'sample_multi_target/difficulty_multiplier': difficulty_multiplier,
            'sample_multi_target/normalized_score': normalized_score,
            'sample_multi_target/quantity_penalty': quantity_penalty,
            'sample_multi_target/final_score': final_score,
            'sample_multi_target/avg_iou': avg_iou,
            'sample_multi_target/high_iou_count': high_iou_count,
            'sample_multi_target/high_iou_ratio': high_iou_count / total_matches if total_matches > 0 else 0.0
        }, commit=False)
    
    return final_score

def iou_reward(solution_str: str, ground_truth: str, extra_info) -> float:
    pred_json = extract_pred_json_data(solution_str)
    gt_json = extract_gt_json_data(ground_truth)

    if not pred_json:
        return 0.0
    if not gt_json:
        return 0.0
    
    if len(pred_json) == 1 and len(gt_json) == 1:
        reward = single_target_iou_reward(pred_json, gt_json)
        if wandb.run is not None:
            wandb.log({
                'sample_target_type': 'single'
            }, commit=False)
        return reward
    else:
        reward = multi_target_iou_reward(pred_json, gt_json, alpha=0.3, penalty_factor=0.5)
        if wandb.run is not None:
            wandb.log({
                'sample_target_type': 'multi'
            }, commit=False)
        return reward

def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info) -> float:
    # format_reward
    format_score = format_reward(solution_str)

    # iou reward
    iou_score = iou_reward(solution_str, ground_truth, extra_info)

    final_score = format_score + iou_score
    
    if wandb.run is not None:
        pred_json = extract_pred_json_data(solution_str)
        gt_json = extract_gt_json_data(ground_truth)
        
        wandb.log({
            'sample_reward/format_score': format_score,
            'sample_reward/iou_score': iou_score,
            'sample_reward/total_score': final_score,
            'sample_data/pred_json_length': len(pred_json) if pred_json else 0,
            'sample_data/gt_json_length': len(gt_json) if gt_json else 0
        }, commit=False)
    
    return final_score
