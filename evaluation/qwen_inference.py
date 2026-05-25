from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
from tqdm import tqdm

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run unified single/multi-process inference with vLLM for Qwen-VL")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output JSON file")
    
    return parser.parse_args()

def main():
    args = parse_args()

    # # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto", device_map="auto"
    )

    # The default range for the number of visual tokens per image in the model is 4-16384.
    # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
    processor = AutoProcessor.from_pretrained(args.model_path)

    with open(args.question_path, 'r') as f:
        messages = json.load(f)
    
    output_data = []
    for message in tqdm(messages, desc='Inference...'):
        # Preparation for inference
        text = processor.apply_chat_template(
            [message], tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info([message])
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        new_item = {
            "content": output_text,
            "ground_truth": message.get('ground_truth')
        }
        output_data.append(new_item)
    
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()