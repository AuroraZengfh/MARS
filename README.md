# Does Seeing More Mean Knowing More? Mono-Anchored Advantage Normalization for Multi-Source Visual Reasoning

This repo is the official implementation of the paper: **Does Seeing More Mean Knowing More? Mono-Anchored Advantage Normalization for Multi-Source Visual Reasoning**

> Does Seeing More Mean Knowing More? Mono-Anchored Advantage Normalization for Multi-Source Visual Reasoning
>
> Fanhu Zeng, Zhicong Luo, Zefan Wang, You Li, Chi Chen, Maosong Sun

[![arXiv](https://img.shields.io/badge/Arxiv-2502.17159-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2605.25437)
[![🤗 Dataset (HuggingFace)](https://img.shields.io/badge/Dataset-HuggingFace-FFD21E.svg?logo=huggingface&logoColor=yellow)](https://huggingface.co/datasets/AuroraZengfh/MARS) 

**Key words: Visual Reasoning, Multimodal Large Language Models, Reinforcement Learning, Multi-Source Perception.**

**TL;DR: An effective way for resolving conflict in multi-source visual reasoning.**

## :newspaper: News

- **[2025.04.11]** We release [Training](#Training) and [Evaluation](#Evaluation) script for MARS. Feel free to try it now! :fireworks:
- **[2025.02.24]** [MARS](https://arxiv.org/abs/2605.25437) is available on Arxiv. :candy:

## :star2: Motivation
<div align="center">
  <img src=figures/illustration.png width="840px">
</div>

Visual reasoning has exhibited strong understanding capabilities under multi-image inputs. Despite the progress of visual reasoning, current methods largely optimize for aligned representations, and the complementary strengths of different sources are often assumed and overutilized, but potential interference or conflicts are seldom explicitly explored. 
In particular, existing RLVR frameworks optimize multi-source rewards directly, without explicitly assessing whether integrating additional sources yields positive information gain or instead introduces interference relative to strong mono-source reasoning,
especially when their attributes and semantics have significant differences, such as medical imaging, autonomous driving, remote sensing, and so on. In these scenarios, **naively integrating multiple sources can even lead to performance inferior to strong mono-source reasoning**, when a specific source contains the dominant and reliable signal. This contradicts the cognition of humans that integrating more information always brings more knowledge, and naturally raises an open question:

> Does seeing more mean knowing more in multi-source visual reasoning?

We therefore aim to enable adaptive regulation of different sources during RLVR training and improve the performance of multi-source reasoning.


## :open_book: Abstract
Visual reasoning through reinforcement learning with verifiable rewards (RLVR) has achieved remarkable progress. However, when dealing with multi-source inputs, existing approaches tend to treat them as a mere accumulation of information, lacking explicit mechanisms to distinguish whether integrating additional sources yields information gain or introduces interference. Therefore, they struggle to effectively model dynamic interaction when integrating multiple sources, particularly when they differ significantly in physical properties and semantics, \eg, infrared and depth, leading to inferior performance to mono-source reasoning when a certain source holds the dominant signal. To address this issue, we propose MARS, a novel mono-anchored multi-source reasoning framework that models each visual modality as an independent information source. Specifically, by treating mono-source rewards as dynamic anchors, our method explicitly incorporates the information gain introduced by multi-source fusion into advantage normalization and adaptively emphasizes mutual promotion between sources while suppressing potential noise or conflicts during RLVR. From theoretical analysis, our method effectively quantifies information gain introduced by multi-source integration in gradient estimation, enabling consistent modality regulation. Empirical results also show impressive 3.2% and 4.9% performance gains on GRPO and DAPO across diverse datasets, confirming the effectiveness of our method.


## :rocket: Quick Start

## Install

Verl gives a detailed instruction for [installation](https://verl.readthedocs.io/en/latest/start/install.html) and we provide a quick start for environment:

1. Clone this repository

```bash
git clone https://github.com/AuroraZengfh/MARS.git
cd MARS
```

2. Install Package
```bash
conda create -n mars python=3.11 -y
conda activate mars
pip3 install -e ".[vllm]"
pip3 install -e ".[sglang]"
```
### Data & Model Prepraration

-- Create `data` folder and download all dataset needed for reasoning.

We provide data for reinforcement fine-tuning:

| Dataset | Download Path | RL data  |
|  :----:  |  :----:  |  :----:  |
| Infrared | [images](https://github.com/bupt-ai-cz/LLVIP) | [Data](https://huggingface.co/datasets/AuroraZengfh/MARS) |
| Depth | [images](https://github.com/BAAI-DCAI/SpatialBot) | [Data](https://huggingface.co/datasets/AuroraZengfh/MARS) | 
| Multi-view |[images](https://huggingface.co/datasets/lmms-lab/M4-Instruct-Data) | [Data](https://huggingface.co/datasets/AuroraZengfh/MARS) |
| Text-rich |[images](https://huggingface.co/datasets/lmms-lab/M4-Instruct-Data) | [Data](https://huggingface.co/datasets/AuroraZengfh/MARS)|

Download the images and put them in `data/images`. Download the RL data and put them appropriately for training below.

### Training

Run the scripts for training. You can specify the task and parameters you are training.

Take grounding (RGB&IR) for Qwen-2.5-VL-3B as an exmpale:

```bash
# GRPO
sh examples/grpo_trainer/run_qwen2_5_vl-3b.sh vllm llvip utils/reward_score/grounding.py 2

# DAPO
sh recipe/dapo/run_dapo_qwen2.5_3b.sh vllm llvip utils/reward_score/grounding.py 2
```

we provide different rewards for mllms: grounding.py for groudning task, and vqa.py for vqa task. You can also train the CoT model on your own multi-source data based on standard training procedure of [LLaMAFactory](https://github.com/hiyouga/LLaMAFactory).


### Inference

Merge the fine-tuned model:
```bash
sh scripts/merge.sh /path/to/fine/tuned/model
```

obtain the result file by specifying the dataset path:
```bash
sh evaluation/qwen_inference.sh /path/to/your/model /path/to/your/dataset data/results/result_qwen3b_grounding_multi_ours.json
```
the dataset we use are provided in `data/questions`.

### Evaluation

get the final results by giving the result file and the type of task:
```bash
sh evaluation/eval_qwen.sh results/result_qwen3b_grounding_multi_ours.json grounding 0.5 
```
Note: groundig for Qwen2.5-VL is absolute coordinates, but Qwen3-VL is relative coordinates, so remember to convert them when needed.

### TODOLIST
- [ ] Data release
- [ ] Pre-trained model release


## :blue_book: Citation
If you find this work useful, consider giving this repository a star :star: and citing :bookmark_tabs: our paper as follows:

```bibtex
@article{zeng2026does,
  title={Does Seeing More Mean Knowing More? Mono-Anchored Advantage Normalization for Multi-Source Visual Reasoning},
  author={Zeng, Fanhu and Luo, Zhicong and Wang, Zefan and Li, You and Chen, Chi and Sun, Maosong},
  journal={arXiv preprint arXiv:2605.25437},
  year={2026}
}
```


## Acknowlegdement

The repository is built upon [LLaMAFactory](https://github.com/hiyouga/LLaMAFactory) and [Verl](https://github.com/verl-project/verl), thanks for these valuable projects!
