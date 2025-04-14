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
model_params = torch.load("/media/automan/ExSpace/Projects/VideoLISA/output/Video/video-lisa.pt")
model.load_state_dict(model_params)

# Step 3: Load evaluated video
num_eval_examples = 10
video_paths = json.load(open("/media/automan/6E94666294662CB1/A_Content/SSv2/labels/validation.json"))
for video_path in video_paths[:num_eval_examples]:
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "video",
                "image": video_path
            },
            {
                "type": "text",
                "text": random.choice(SSV2_QUESTION_LIST)
            }
        ]}]

    response, _ = predict(messages, model, processor)
    messages.append({"role": "assistant", "content": f"{response}"})
    print(messages[-1])