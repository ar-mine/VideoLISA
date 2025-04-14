import torch
import cv2
import numpy as np
import random
from utils import SHORT_QUESTION_LIST, ANSWER_LIST
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
from model.VideoLISA import VideoLISA
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from utils import ModelArguments, ScriptArguments, predict

def process_func(example):
    """
    将数据集进行预处理
    """
    MAX_LENGTH = 8192
    input_ids, attention_mask, labels = [], [], []
    conversation = example["conversations"]
    input_content = conversation[0]["value"]
    # output_content = conversation[1]["value"]
    output_content = random.choice(ANSWER_LIST)
    file_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0]  # 获取图像路径
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{file_path}",
                    "resized_height": 280,
                    "resized_width": 280,
                },
                {"type": "text", "text": f"{random.choice(SHORT_QUESTION_LIST)}"},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )  # 获取文本
    image_inputs, video_inputs = process_vision_info(messages)  # 获取数据数据（预处理过）
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.tolist() for key, value in inputs.items()} #tensor -> list,为了方便拼接
    instruction = inputs

    response = tokenizer(f"{output_content}", add_special_tokens=False)

    input_ids = (
            instruction["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    )

    attention_mask = instruction["attention_mask"][0] + response["attention_mask"] + [1]
    labels = (
            [-100] * len(instruction["input_ids"][0])
            + response["input_ids"]
            + [tokenizer.pad_token_id]
    )
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)
    inputs['pixel_values'] = torch.tensor(inputs['pixel_values'])
    inputs['image_grid_thw'] = torch.tensor(inputs['image_grid_thw']).squeeze(0)  #由（1,h,w)变换为（h,w）
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
            "pixel_values": inputs['pixel_values'], "image_grid_thw": inputs['image_grid_thw']}

model_path = "Qwen/Qwen2.5-VL-3B-Instruct"

model = VideoLISA.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
model.init_sam_module(model_path="/media/automan/ExSpace/Projects/VideoLISA/checkpoints/sam_vit_h_4b8939.pth")
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
tokenizer.add_tokens("<seg>", special_tokens=False)
model.seg_token_idx = tokenizer.convert_tokens_to_ids("<seg>")
model.resize_token_embeddings(len(tokenizer))
processor = AutoProcessor.from_pretrained(model_path)
processor.tokenizer = tokenizer

# 获取测试模型
model_params = torch.load("/media/automan/ExSpace/Projects/VideoLISA/output/VideoLISA/video-lisa.pt")
model.load_state_dict(model_params)

origin_image_path = "/media/automan/6E94666294662CB1/A_Content/Datasets/ADEChallengeData2016/images/training/ADE_train_00000001.jpg"
messages = [{
    "role": "user",
    "content": [
        {
            "type": "image",
            "image": origin_image_path
        },
        {
            "type": "text",
            "text": "Can you segment the floor in this image?"
        }
    ]}]

response, image = predict(messages, model, processor)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imwrite("output/image.png", image.astype(np.uint8))
messages.append({"role": "assistant", "content": f"{response}"})
print(messages[-1])