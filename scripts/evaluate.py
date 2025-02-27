import re
import json

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from qwen_vl_utils import process_vision_info


def multiply_numbers_in_string(text, coeff):
    # 回调函数：将匹配到的数字串转换为整数，乘以系数，再转换回字符串
    def repl(match):
        num_str = match.group(0)
        return str(int(num_str) * coeff)

    # 使用正则表达式替换所有连续数字串
    return re.sub(r'\d+', repl, text)


def predict(processor, messages, model):
    # 准备推理
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
    fps_inputs = video_kwargs['fps']
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        fps=fps_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # 生成输出
    output_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    output_text = multiply_numbers_in_string(output_text[0], 1/fps_inputs[0])
    return output_text


if __name__ == "__main__":
    model_path = "Qwen/Qwen2.5-VL-3B-Instruct"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, use_fast=False)

    model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
    # ===测试模式===
    # 配置测试参数
    val_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=True,  # 测试模式
        r=64,  # Lora 秩
        lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.05,  # Dropout 比例
        bias="none",
    )

    # 获取测试模型
    val_peft_model = PeftModel.from_pretrained(model, model_id="../output/Qwen2_5-VL-3B/checkpoint-500", config=val_config)

    # 读取测试数据
    with open("../dataset/labels/test-2000.json", "r") as f:
        test_dataset = json.load(f)

    MAX_TEST_SAMPLES = 1
    SAVE_PATH = "/media/automan/ExSpace/Projects/VideoLISA/dataset/videos"
    prompt = "Could you provide a summary of the incidents that occurred at various timestamps in the video?"

    for data in test_dataset[:MAX_TEST_SAMPLES]:
        video_id = data["id"]
        # video_path = f"{SAVE_PATH}/{video_id}/{video_id}_vtime.mp4"
        video_path = "./1.mp4"
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                # {"type": "text", "text": "During which time period in the video does the event 'a person is pointing to a mug' happens?"},
                {"type": "text", "text": prompt},
                {"video": video_path, "total_pixels": 20480 * 28 * 28, "min_pixels": 16 * 28 * 28},
            ]
             },
        ]
        {'role': 'assistant', 'content': 'From 0.0 to 1.5499357142857142, a coffee maker and a cup on a table. '
        'From 1.5499357142857142 to 3.6165166666666666, a person is pointing to a coffee maker and a mug. '
        'From 3.6165166666666666 to 5.6830976190476195, a coffee maker and a cup sitting on top of a table.'}
        response = predict(processor, messages, val_peft_model)
        messages.append({"role": "assistant", "content": f"{response}"})
        print(messages[-1])