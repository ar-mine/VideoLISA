import json
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
import torch
import os
import numpy as np
from qwen_vl_utils import fetch_video
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from datasets import load_dataset

SAVE_PATH = "/media/automan/6E94666294662CB1/A_Content/Youtube/videos"

def process_func(sample):
    sample_id = sample['id']
    video_path = os.path.join(SAVE_PATH, sample_id)

    """ Fetch clip videos file """
    clip_video_path = os.path.join(video_path, '{}_vtime.mp4'.format(sample_id))
    video_input, video_sample_fps = fetch_video({"video": clip_video_path,
                                                 "max_frames": 12},
                                                return_video_sample_fps=True)
    # print(f"Frames shape: {video_input.shape}, FPS: {video_sample_fps}")

    """ Convert timestamp (token) to frame number (ntoken)"""
    sample['meta']['ntoken'] = {}
    time_interval = (video_input.shape[0]-1) / sample['meta']['duration']
    output_content = sample['conversations'][1]['value']
    for k in sample['meta']['token'].keys():
        if not sample['meta']['token'][k] is None:
            sample['meta']['ntoken'][k] = int(sample['meta']['token'][k]*time_interval)
            output_content = output_content.replace(k, str(sample['meta']['ntoken'][k]))

    """ Fetch text prompt input and output"""
    prompt = sample['conversations'][0]['value'][8:]
    max_new_tokens = 2048
    total_pixels = 20480 * 28 * 28
    min_pixels = 16 * 28 * 28
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"video": video_path, "total_pixels": total_pixels, "min_pixels": min_pixels},
        ]
         },
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )  # 获取文本
    image_inputs = []
    video_inputs = [video_input]
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.tolist() for key, value in inputs.items() if isinstance(value, torch.Tensor)} #tensor -> list,为了方便拼接
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
    pixel_values_videos = inputs['pixel_values_videos']
    video_grid_thw = inputs['video_grid_thw']

    # if len(input_ids) > MAX_LENGTH:  # 做一个截断
    #     input_ids = input_ids[:MAX_LENGTH]
    #     attention_mask = attention_mask[:MAX_LENGTH]
    #     labels = labels[:MAX_LENGTH]

    #input_ids = torch.tensor(input_ids)
    #attention_mask = torch.tensor(attention_mask)
    #labels = torch.tensor(labels)
    #pixel_values_videos = torch.tensor(pixel_values_videos)
    #video_grid_thw = torch.tensor(video_grid_thw).squeeze(0)  #由（1,h,w)变换为（h,w）
    video_grid_thw = np.squeeze(video_grid_thw, axis=0)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
            "pixel_values_videos": pixel_values_videos, "video_grid_thw": video_grid_thw}

model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_path)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

# 使用示例：
# 假设 labels 是一个包含所有样本字典的列表
train_ds = load_dataset("json", data_files="index.json", split="train", streaming=True)
train_dataset = train_ds.map(process_func)

# 配置LoRA
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=64,  # Lora 秩
    lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.05,  # Dropout 比例
    bias="none",
)

# 获取LoRA模型
peft_model = get_peft_model(model, config)

# 配置训练参数
args = TrainingArguments(
    output_dir="./output/Qwen2_5-VL-3B",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=10,
    logging_steps=50,
    max_steps=500,
    num_train_epochs=5,  # 设置为你期望的 epoch 数
    save_steps=1000,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)

# 配置Trainer
trainer = Trainer(
    model=peft_model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
# 开启模型训练
trainer.train()
