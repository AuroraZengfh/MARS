import re
import json
import wandb


format_pattern = r"<think>[\S\n\t\v ]*?</think>\s*<answer>[\S\n\t\v ]*?</answer>"


def extract_answer_content(solution_str: str):
    answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    answer_match = answer_pattern.search(solution_str)
    if answer_match:
        return answer_match.group(1).strip()
    return None

def format_reward(solution_str: str) -> float:
    """
    Format reward (maximum 0.75):
    - <think>: 0.25 points (exactly one pair, and content is not empty)
    - <answer>: 0.25 points (exactly one pair, and content is not empty)
    """

    reward = 0.0
    format_details = {}

    # Check <think> - ensure there is exactly one pair and that the content is not empty
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    think_matches = think_pattern.findall(solution_str)
    if len(think_matches) == 1:
        think_content = think_matches[0].strip()
        if think_content:
            reward += 0.25
            format_details["think_format"] = 1
        else:
            format_details["think_format"] = 0
    else:
        format_details["think_format"] = 0

    # Check <answer> - ensure there is exactly one pair and that the content is not empty
    answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    answer_matches = answer_pattern.findall(solution_str)
    answer_content = None
    if len(answer_matches) == 1:
        answer_content = answer_matches[0].strip()
        if answer_content:
            reward += 0.25
            format_details["answer_format"] = 1
        else:
            format_details["answer_format"] = 0
    else:
        format_details["answer_format"] = 0

    if wandb.run is not None:
        wandb.log(
            {
                "sample_format/think_format": format_details["think_format"],
                "sample_format/answer_format": format_details["answer_format"],
                "sample_format/total_format_score": reward,
            },
            commit=False,
        )

    return reward

def acc_reward(solution_str: str, ground_truth: str, extra_info) -> float:
    """
    Accuracy reward for yes/no questions:
    - Reward is 1.0 if predicted decision matches ground-truth decision, else 0.0.
    - Extract yes/no from <answer>...</answer> if present.
    """
    pred = extract_answer_content(solution_str).strip().lower()
    gt = extract_answer_content(ground_truth).strip().lower()

    if pred is None or gt is None:
        return 0.0
    
    score = 1.0 if (pred is not None and gt is not None and  pred in gt or gt in pred) else 0.0
    
    if wandb.run is not None:
        wandb.log(
            {
                "sample_reward/acc_score": score,
                "sample_reward/pred_yesno": pred if pred is not None else "None",
                "sample_reward/gt_yesno": gt if gt is not None else "None",
            },
            commit=False,
        )
    return score


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info) -> float:
    # format_reward
    format_score = format_reward(solution_str)

    # acc reward
    acc_score = acc_reward(solution_str, ground_truth, extra_info)

    final_score = format_score + acc_score

    if wandb.run is not None:
        wandb.log(
            {
                "sample_reward/format_score": format_score,
                "sample_reward/acc_score": acc_score,
                "sample_reward/total_score": final_score,
            }, commit=False)

    return final_score
