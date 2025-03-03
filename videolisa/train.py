import os
import json
from dataclasses import dataclass, field
import torch
import numpy as np

from qwen_vl_utils import fetch_video
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from trl import (
    TrlParser,
)
from peft import LoraConfig, TaskType, get_peft_model
from datasets import IterableDataset

"""
accelerate launch --config_file configs/accelerate/zero3.yaml --num_processes=1 \
videolisa/train.py \
--model_name_or_path=Qwen/Qwen2.5-1.5B-Instruct \
--attn_implementation=flash_attention_2 \
--target_modules="q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj" \
--lora_r=64 \
--lora_alpha=16 \
--lora_dropout=0.05 \
--max_frames=12 \
--per_device_train_batch_size=1 \
--data_root=/media/automan/6E94666294662CB1/A_Content/Youtube/videos \
--dataset_path=/media/automan/ExSpace/Projects/VideoLISA/dataset/labels/train-10000.json \
--num_train_epochs=1
"""

# On server dataset is organized without folders
LOCAL = True

def process_func(sample, processor, tokenizer, max_frames, data_root):
    sample_id = sample['id']
    if LOCAL:
        video_path = os.path.join(data_root, sample_id)
    else:
        video_path = data_root
    """ Fetch clip videos file """
    clip_video_path = os.path.join(video_path, '{}_vtime.mp4'.format(sample_id))
    video_input, video_sample_fps = fetch_video({"video": clip_video_path,
                                                 "max_frames": max_frames},
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

def data_generator(samples, processor, tokenizer, max_frames, data_root):
    for sample in samples:
        yield process_func(sample, processor, tokenizer, max_frames, data_root)


@dataclass
class ModelArguments:
    model_name_or_path: str = field()
    attn_implementation: str = field()

    target_modules: list[str] = field()
    lora_r: int = field()
    lora_alpha: int = field()
    lora_dropout: float = field()

@dataclass
class ScriptArguments:
    max_frames: int = field(
        metadata={'help': 'Max number of frames to process'},
    )
    data_root: str = field()
    dataset_path: str = field()


def main(training_args, model_args, script_args):
    # 配置Backbone
    model_path = model_args.model_name_or_path
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=model_args.attn_implementation,
        # device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path)
    model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

    # 配置LoRA
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=model_args.target_modules,
        inference_mode=False,  # 训练模式
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
    )
    peft_model = get_peft_model(model, config)

    # 配置数据集
    # train_ds = load_dataset("json", data_files=script_args.dataset_path, streaming=True)
    # train_dataset = train_ds.map(process_func, fn_kwargs={
    #     "processor": processor,
    #     "tokenizer": tokenizer,
    #     "max_frames": script_args.max_frames,
    #     "data_root": script_args.data_root,
    # })

    train_dataset = IterableDataset.from_generator(data_generator,
                                                   gen_kwargs={"samples": json.load(open(script_args.dataset_path)),
                                                               "processor": processor,
                                                               "tokenizer": tokenizer,
                                                               "max_frames": script_args.max_frames,
                                                               "data_root": script_args.data_root})
    # 配置Trainer
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    # 开启模型训练
    trainer.train()


if __name__ == "__main__":
    parser = TrlParser((TrainingArguments, ModelArguments, ScriptArguments))
    training_args, model_args, script_args = parser.parse_args_and_config()
    main(training_args, model_args, script_args)