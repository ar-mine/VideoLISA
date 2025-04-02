import re
import json
import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from qwen_vl_utils import process_vision_info
from transformers import TrainingArguments
from trl import TrlParser
from utils import ModelArguments, ScriptArguments
from tqdm import tqdm


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


def iou(outputs, gt):
    matches = re.findall(r'-?\d+\.\d+|-?\.\d+|-?\d+', outputs)
    matches = [float(num) for num in matches]
    if not matches:
        return 0.0
    from_number, to_number = matches[0], matches[-1]
    s, e = gt
    intersection = max(0, min(to_number, e) - max(from_number, s))
    union = max(to_number, e) - min(from_number, s)
    iou = intersection / union
    return round(iou, 2)


def main(training_args, model_args, script_args):
    model_path = model_args.model_name_or_path
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=model_args.attn_implementation,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

    # ===测试模式===
    # 配置测试参数
    val_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=model_args.target_modules,
        inference_mode=True,  # 测试模式
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
    )
    # 获取测试模型
    model_id = f"{os.getcwd()}/{training_args.output_dir}/checkpoint-{training_args.max_steps}"
    val_peft_model = PeftModel.from_pretrained(model, model_id=model_id, config=val_config)

    # val_peft_model = model

    # 读取测试数据
    val_samples = json.load(open(script_args.val_dataset_path))
    data_type = script_args.val_dataset_type
    if data_type not in ["internvid", "activitynet"]:
        raise NotImplementedError(f"Data type {data_type} is not supported.")
    count, scores = 0, 0.0
    if data_type == "internvid":
        data_root = script_args.data_root
        for data in val_samples[:10]:
            video_id = data["id"]
            video_path = f"{data_root}/{video_id}/{video_id}_vtime.mp4"
            print(f"Processing video: {video_path}")
            for i in range(len(data["conversations"])//2):
                question = data["conversations"][2*i]["value"][8:]
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": [
                        {"type": "text", "text": question},
                        {"video": video_path, "total_pixels": 20480 * 28 * 28, "min_pixels": 16 * 28 * 28, "max_frames": script_args.max_frames},
                    ]},
                ]
                response = predict(processor, messages, val_peft_model)
                messages.append({"role": "assistant", "content": f"{response}"})

                s = data["meta"]["token"][f"<s{i}>"]
                e = data["meta"]["token"][f"<e{i}>"]
                print("Ground Truth: " + f"{s} to {e}")
                r = messages[-1]["content"]
                print("Result:" + r)
                iou_score = iou(response, (s, e))
                print("IoU: " + f"{iou_score}")
                count += 1
                scores += iou_score
    elif data_type == "activitynet":
        data_root = script_args.data_root
        video_index = json.load(open(f"{data_root}/index.json", "r"))
        for video_id in tqdm(val_samples.keys()):
            if video_id not in video_index.keys():
                print(f"Skipping video: {video_id}")
                continue
            video_path = os.path.join(data_root, video_index[video_id])
            print(f"Processing video: {video_path}")
            data = val_samples[video_id]
            for i in range(len(data["sentences"])):
                question = data["sentences"][i]
                question = question.strip().lower()
                if question.endswith("."):
                    question = question[:-1]
                question = f"During which frames can we see {question}?"
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": [
                        {"type": "text", "text": question},
                        {"video": video_path, "total_pixels": 20480 * 28 * 28, "min_pixels": 16 * 28 * 28, "max_frames": script_args.max_frames},
                    ]},
                ]
                response = predict(processor, messages, val_peft_model)
                messages.append({"role": "assistant", "content": f"{response}"})

                s = data["timestamps"][i][0]
                e = data["timestamps"][i][1]
                # print("Ground Truth: " + f"{s} to {e}")
                r = messages[-1]["content"]
                # print("Result:" + r)
                iou_score = iou(response, (s, e))
                # print("IoU: " + f"{iou_score}")
                count += 1
                scores += iou_score
    mIoU = scores / count
    print(f"mIoU: {mIoU}")

if __name__ == "__main__":
    parser = TrlParser((TrainingArguments, ModelArguments, ScriptArguments))
    training_args, model_args, script_args = parser.parse_args_and_config()
    main(training_args, model_args, script_args)

