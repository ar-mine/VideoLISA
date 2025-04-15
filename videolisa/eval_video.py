import json

import torch
import random
from utils import SSV2_QUESTION_LIST, predict
from transformers import AutoProcessor, AutoTokenizer
from model.VideoLISA import VideoLISA


# Step 1: Load original model
model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
model = VideoLISA.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_path)

# Step 2: Load weights from ckpt
# from peft import LoraConfig, TaskType, get_peft_model, PeftModel
# config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
#     inference_mode=False,  # 训练模式
#     r=64,
#     lora_alpha=16,
#     lora_dropout=0.05,
#     bias="none",
# )
# model = PeftModel.from_pretrained(model, model_id="/media/automan/ExSpace/Projects/VideoLISA/output/Video/checkpoint-2639", config=config)
model_params = torch.load("/media/automan/ExSpace/Projects/VideoLISA/output/Video/video.pt")
model.load_state_dict(model_params)

# Step 3: Load evaluated video
num_eval_examples = 10
video_infos = json.load(open("/media/automan/6E94666294662CB1/A_Content/SSv2/labels/train.json"))
success, total = 0, 0
for video_info in video_infos[:num_eval_examples]:
    video_path = "/media/automan/6E94666294662CB1/A_Content/SSv2/20bn-something-something-v2/" + video_info["id"] + ".webm"
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_path,
                "fps": 2,
                "min_frames": 2,
                "max_frames": 12,
            },
            {
                "type": "text",
                "text": random.choice(SSV2_QUESTION_LIST)
            }
        ]}]

    response, _ = predict(messages, model, processor)
    messages.append({"role": "assistant", "content": f"{response}"})
    print(f"Id: " + video_info["id"] + str(messages[-1]))

    print(f"Gt: " + video_info["id"] + ":" + video_info["label"])

    gt = video_info["template"].lower()
    result = json.loads(messages[-1]["content"])
    objects = result["objects"]
    action = result["action"]
    for obj in objects:
        action = action.replace(obj, "[something]")
    if action == gt:
        success += 1
    else:
        a = 1
    total += 1
print(f"Success Rate: {success}/{total}")