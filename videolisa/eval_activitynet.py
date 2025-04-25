import json

import random
import torch
import os


from utils import predict, calculate_iou
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
model_params = torch.load("/media/automan/ExSpace/Projects/VideoLISA/output/frame-24-ep-4-bs-64-lora-128-s1/video.pt")
model.load_state_dict(model_params)

# Step 3: Load evaluated video
num_eval_examples = 1000
video_infos = json.load(open("/media/automan/6E94666294662CB1/A_Content/Datasets/ActivityNet-Captions/labels/val_1.json"))
video_paths = json.load(open("/media/automan/6E94666294662CB1/A_Content/Datasets/ActivityNet-Captions/index.json"))

messages = [{
    "role": "user",
    "content": [
        {
            "type": "video",
            "video": "",
            "fps": 2,
            "min_frames": 2,
            "max_frames": 16,
        },
        {
            "type": "text",
            "text": ""
        }
    ]}]

def temporal_grounding():
    total_iou, total_iou_l3, total_iou_l5, total_iou_l7, total = 0, 0, 0, 0, 0
    for k, v in video_infos.items():
        video_path = os.path.join("/media/automan/6E94666294662CB1/A_Content/Datasets/ActivityNet-Captions/", video_paths[k])
        idx = random.randint(0, len(v["timestamps"]) - 1)
        sent = v["sentences"][idx]
        label = v["timestamps"][idx]

        messages[0]["content"][0]["video"] = video_path
        messages[0]["content"][1]["text"] = f"During which frames can we see '{sent}' happening in the video?"
        response, _, fps = predict(messages, model, processor)
        video_sample_fps = fps["fps"][0]
        # response_dict = {"role": "assistant", "content": f"{response}"}
        s_et = json.loads(response)
        s_et = [s_et[0]/video_sample_fps, s_et[1]/video_sample_fps]
        if s_et[0] >= s_et[1] or label[0] >= label[1]:
            continue
        iou = calculate_iou(s_et[0], s_et[1], label[0], label[1])
        print(f"Id: " + k + str(messages[-1]) + str(s_et))
        print(f"Gt: " + k + ":" + str(label))
        print(f"IoU: " + k + ":" + str(iou))
        total_iou += iou
        if iou > 0.3:
            total_iou_l3 += 1
        if iou > 0.5:
            total_iou_l5 += 1
        if iou > 0.7:
            total_iou_l7 += 1
        total += 1
        if total > num_eval_examples:
            break
    total -= 1
    print(f"mIoU: {total_iou/total}")
    print(f"R@0.3: {total_iou_l3/total}")
    print(f"R@0.5: {total_iou_l5/total}")
    print(f"R@0.7: {total_iou_l7/total}")

def dense_caption():
    output = {}
    total = 0
    for k, v in video_infos.items():
        video_path = os.path.join("/media/automan/6E94666294662CB1/A_Content/Datasets/ActivityNet-Captions/", video_paths[k])
        label = v

        messages[0]["content"][0]["video"] = video_path
        messages[0]["content"][1]["text"] = "Describe the video with its related frame index in JSON format and it should be a list including 'description' and 'time' as keys."

        response, _, fps = predict(messages, model, processor)
        video_sample_fps = fps["fps"][0]
        # messages.append({"role": "assistant", "content": f"{response}"})
        try:
            results = json.loads(response)
        except:
            continue
        for r in results:
            r['time'] = [r['time'][0]/video_sample_fps, r['time'][1]/video_sample_fps]
        print(f"Id: " + k + str(messages[-1]) + str(results))
        print(f"Gt: " + k + ":" + str(label))
        output[k] = results
        total += 1
        if total > num_eval_examples:
            break
    output_dict = {"results": output}
    json.dump(output_dict, open("results-16.json", "w"))

if __name__ == "__main__":
    dense_caption()