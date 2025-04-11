import os
import random

import numpy as np
import torch
from PIL import Image

from qwen_vl_utils import process_vision_info

from .meta import *


def init_ade20k(base_image_dir, split="train"):
    split_map = {
        "train": "training",
        "val": "validation",
        "test": "validation",
    }
    split = split_map[split]
    ade20k_classes = np.array(ADE20K_CLASSES)
    image_ids = sorted(
        os.listdir(os.path.join(base_image_dir, "images", split))
    )
    ade20k_image_ids = []
    for x in image_ids:
        if x.endswith(".jpg"):
            ade20k_image_ids.append(x[:-4])
    ade20k_images = []
    for image_id in ade20k_image_ids:  # self.descriptions:
        ade20k_images.append(
            os.path.join(
                base_image_dir,
                "images",
                split,
                "{}.jpg".format(image_id),
            )
        )
    ade20k_labels = [
        x.replace(".jpg", ".png").replace("images", "annotations")
        for x in ade20k_images
    ]
    print("ade20k: ", len(ade20k_images))
    return ade20k_classes, ade20k_images, ade20k_labels


class SemSegDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        processor,
        tokenizer,
        precision: str = "fp32",
        sem_seg_data="ade20k",
    ):

        self.processor = processor
        self.tokenizer = tokenizer
        self.precision = precision

        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        self.data2list = {}
        self.data2classes = {}

        # Initialize dataset index
        self.sem_seg_datas = sem_seg_data.split("||")
        for ds in self.sem_seg_datas:
            classes, images, labels = eval("init_{}".format(ds))(base_image_dir[ds])
            self.data2list[ds] = (images, labels)
            self.data2classes[ds] = classes

        self.length = 0

    def __len__(self):
        if self.length == 0:
            for ds in self.sem_seg_datas:
                self.length += len(self.data2list[ds][0])
            ret = self.length
        else:
            ret = self.length
        return ret


    def __getitem__(self, idx):
        # Select from random dataset
        ds = random.randint(0, len(self.sem_seg_datas) - 1)
        ds = self.sem_seg_datas[ds]

        if ds in ["ade20k"]:
            # Get image and label array with idx
            images, masks = self.data2list[ds]
            idx = random.randint(0, len(images) - 1)
            image_path = images[idx]
            mask_path = masks[idx]
            mask = Image.open(mask_path)
            mask = np.array(mask)
            if ds in ["ade20k"]:
                mask -= 1
            image = Image.open(image_path)
            image = np.array(image)

            # Ground Truth
            output_content = random.choice(ANSWER_LIST)
            response = self.tokenizer(f"{output_content}", add_special_tokens=False)
            unique_label = np.unique(mask).tolist()
            if 255 in unique_label:
                unique_label.remove(255)
            if len(unique_label) == 0:
                return self.__getitem__(0)
            classes = [self.data2classes[ds][class_id] for class_id in unique_label]
            sampled_class = np.random.choice(classes, size=1, replace=False).tolist()[0]
            class_id = self.data2classes[ds].tolist().index(sampled_class)
            mask = (mask == class_id).astype(float)

            # Preprocess image for qwen
            message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"{image_path}",
                            "resized_height": 280,
                            "resized_width": 280,
                        },
                        {"type": "text",
                         "text": {random.choice(SHORT_QUESTION_LIST).format(class_name=sampled_class.lower())}},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )  # 获取文本
            image_input, video_input = process_vision_info(message)  # 获取数据数据（预处理过）
            inputs = self.processor(
                text=[text],
                images=image_input,
                videos=video_input,
                padding=True,
                return_tensors="pt",
            )

            inputs = {key: value.tolist() for key, value in inputs.items()} #tensor -> list,为了方便拼接
            instruction = inputs

            input_ids = (
                    instruction["input_ids"][0] + response["input_ids"] + [self.tokenizer.pad_token_id]
            )

            attention_mask = instruction["attention_mask"][0] + response["attention_mask"] + [1]
            labels = (
                    [-100] * len(instruction["input_ids"][0])
                    + response["input_ids"]
                    + [self.tokenizer.pad_token_id]
            )
            # if len(input_ids) > MAX_LENGTH:  # 做一个截断
            #     input_ids = input_ids[:MAX_LENGTH]
            #     attention_mask = attention_mask[:MAX_LENGTH]
            #     labels = labels[:MAX_LENGTH]

            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)
            labels = torch.tensor(labels)
            mask = torch.tensor(mask)
            inputs['pixel_values'] = torch.tensor(inputs['pixel_values'])
            inputs['image_grid_thw'] = torch.tensor(inputs['image_grid_thw']).squeeze(0)  #由（1,h,w)变换为（h,w
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
                    "pixel_values": inputs['pixel_values'], "image_grid_thw": inputs['image_grid_thw'],
                    "gt_masks": mask, "original_images": image}

        else:
            raise NotImplementedError