import json

import random
import torch
import os
from utils import predict
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
model_params = torch.load("/media/automan/ExSpace/Projects/VideoLISA/output/frame-16-ep-4-bs-64-lora-128-s1/video.pt")
model.load_state_dict(model_params)

# Step 3: Load evaluated video
num_eval_examples = 10
video_infos = json.load(open("/media/automan/6E94666294662CB1/A_Content/Datasets/ActivityNet2/labels/val_1.json"))
video_paths = json.load(open("/media/automan/6E94666294662CB1/A_Content/Datasets/ActivityNet2/index.json"))
success, total = 0, 0
for k, v in video_infos.items():
    video_path = os.path.join("/media/automan/6E94666294662CB1/A_Content/Datasets/ActivityNet2/", video_paths[k])
    idx = random.randint(0, len(v["timestamps"]) - 1)
    sent = v["sentences"][idx]
    label = v["timestamps"][idx]
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_path,
                "fps": 2,
                "min_frames": 2,
                "max_frames": 16,
            },
            {
                "type": "text",
                "text": f"During which frames can we see '{sent}' happening in the video?"
            }
        ]}]

    response, _ = predict(messages, model, processor)
    messages.append({"role": "assistant", "content": f"{response}"})
    print(f"Id: " + k + str(messages[-1]))

    print(f"Gt: " + k + ":" + str(label))


    total += 1
    if total > num_eval_examples:
        break
print(f"Success Rate: {success}/{total}")