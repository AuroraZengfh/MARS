from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/home/zengfanhu/pretrained/Qwen2.5-VL-7B", torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("/home/zengfanhu/pretrained/Qwen2.5-VL-7B")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)



# Messages containing multiple images and a text query
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "/home/zengfanhu/datasets/RoadScene-master/crop_LR_visible/FLIR_00006.jpg"},
            {"type": "image", "image": "/home/zengfanhu/datasets/RoadScene-master/cropinfrared/FLIR_00006.jpg"},
            {"type": "text", "text": "Input are visible and infrared images of the same scene. Answer the following question: What is the number on the rear hood of the car in front on the right?"},
        ],
    }
]
# 8056989820
# ['The number on the rear hood of the car in front on the right is 805-698-9828.']

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "image": "/home/zengfanhu/datasets/RoadScene-master/crop_LR_visible/FLIR_00006.jpg"},
#             {"type": "text", "text": "Answer the following question acoording to RGB input: What is the number on the rear hood of the car in front on the right?."},
#         ],
#     }
# ]
# ['The image appears to be overexposed, making it difficult to read the numbers on the rear hood of the car. However, based on the visible portion of the license plate, the numbers appear to be "515-808-9126".']

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "/home/zengfanhu/datasets/RoadScene-master/cropinfrared/FLIR_00006.jpg"},
            {"type": "text", "text": "Answer the following question acoording to infrared input: What is the number on the rear hood of the car in front on the right?."},
        ],
    }
]
# ['The number on the rear hood of the car in the image is 805-698-9828.']

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "image": "/home/zengfanhu/datasets/RoadScene-master/crop_LR_visible/FLIR_00006.jpg"},
#             {"type": "image", "image": "/home/zengfanhu/datasets/RoadScene-master/cropinfrared/FLIR_00006.jpg"},
#             {"type": "text", "text": "Input are visible and infrared images of the same scene. Answer the following question and Choose the right option: What is the number on the rear hood of the car in front on the right? A: 8056120193 B: 8294066201 C: 4502681278 D: 8056989820"},
#         ],
#     }
# ]
# ['D: 8056989820']

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "image": "/home/zengfanhu/datasets/RoadScene-master/crop_LR_visible/FLIR_00006.jpg"},
#             {"type": "text", "text": "Answer the following question acoording to RGB input and choose the right option: What is the number on the rear hood of the car in front on the right? A: 8056120193 B: 8294066201 C: 4502681278 D: 8056989820"},
#         ],
#     }
# ]
# ['The number on the rear hood of the car in front on the right is D: 8056989820.']


# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "image": "/home/zengfanhu/datasets/RoadScene-master/cropinfrared/FLIR_00006.jpg"},
#             {"type": "text", "text": "Answer the following question acoording to infrared input and choose the right option: What is the number on the rear hood of the car in front on the right? A: 8056120193 B: 8294066201 C: 4502681278 D: 8056989820"},
#         ],
#     }
# ]
# ['The number on the rear hood of the car in front on the right is 805-698-9820. Therefore, the correct answer is:\n\nD: 8056989820']



#  A: 8056120193 B: 8294066201 C: 4502681278 D: 8056989820

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
