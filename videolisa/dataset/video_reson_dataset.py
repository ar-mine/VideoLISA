import os
import random
import json
import torch

from qwen_vl_utils import process_vision_info, fetch_video


def init_ssv2(base_data_dir, split="train"):
    """
    初始化 Something-Something 数据集，加载元数据并构建视频路径。

    参数：
        base_data_dir (str): 包含 '20bn-something-something-v2' 和 'labels' 文件夹的基础目录。
        split (str): 数据集划分（'train'、'val' 或 'test'）。

    返回：
        label_map (dict): 标签ID到标签名称的映射。
        video_paths (list): 视频文件路径列表。
        labels (list): 对应的标签ID列表。
    """
    split_map = {"train": "train.json", "val": "validation.json", "test": "test.json"}
    split_file = split_map[split]

    # 加载标签映射
    with open(os.path.join(base_data_dir, "labels", "labels.json"), "r") as f:
        ssv2_classes = json.load(f)

    # 加载特定划分的元数据
    with open(os.path.join(base_data_dir, "labels", split_file), "r") as f:
        data = json.load(f)

    # 构建视频路径和标签
    ssv2_videos = [
        os.path.join(base_data_dir, "20bn-something-something-v2", f"{item['id']}.webm")
        for item in data
    ]
    ssv2_labels = [{"action": item["label"], "objects": item["placeholders"]} for item in data]

    print(f"SSv2 ({split}): {len(ssv2_videos)} videos")
    return ssv2_classes, ssv2_videos, ssv2_labels


class VideoDataset(torch.utils.data.Dataset):
    """
    Something-Something 数据集的 PyTorch Dataset 类，与 Qwen2.5-VL 兼容。
    """

    def __init__(
            self,
            base_data_dir,
            processor,
            tokenizer,
            precision: str = "fp32",
            split: str = "train",
            max_frames: int = 12,
    ):
        """
        初始化数据集。

        参数：
            base_data_dir (str): 数据集目录路径。
            processor: Qwen2.5-VL 处理器，用于处理输入。
            tokenizer: Qwen2.5-VL 分词器。
            precision (str): 数据精度（'fp32' 或其他）。
            split (str): 数据集划分（'train'、'val' 或 'test'）。
            num_frames (int): 从每个视频中采样的帧数。
        """
        self.base_data_dir = base_data_dir
        self.processor = processor
        self.tokenizer = tokenizer
        self.precision = precision
        self.split = split
        self.max_frames = max_frames

        # 初始化数据集
        self.label_map, self.video_paths, self.labels = init_ssv2(
            base_data_dir["ssv2"], split
        )
        self.length = len(self.video_paths)

        # 问题和回答列表占位符（可根据需要自定义）
        self.short_question_list = [
            "Output the action shown in the video with its interacting objects in JSON format and it should contains 'action' and 'objects' as keys.",
            "Generate the action depicted in the video along with its interacting objects in JSON format, including 'action' and 'objects' as keys.",
            "Produce the action shown in the video and its related objects in JSON format, with 'action' and 'objects' as keys."
            "Output the action from the video and the objects involved in JSON format, containing 'action' and 'objects' as keys."
            "Create a JSON representation of the action in the video and its interacting objects, using 'action' and 'objects' as keys."
        ]

    def __len__(self):
        """
        返回数据集中视频的总数。

        返回：
            int: 数据集长度。
        """
        return self.length

    def __getitem__(self, idx):
        """
        获取并处理数据集中的单个样本。

        参数：
            idx (int): 样本索引。

        返回：
            dict: 包含 input_ids、attention_mask、labels 等模型输入的字典。
        """
        # 获取视频路径和标签
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        #label_text = self.label_map[str(label)]

        # 为 Qwen2.5-VL 构建消息
        output_content = json.dumps(label)
        response = self.tokenizer(f"{output_content}", add_special_tokens=False)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "fps": 2,
                        "min_frames": 2,
                        "max_frames": self.max_frames,
                    },
                    {
                        "type": "text",
                        "text": random.choice(self.short_question_list),
                    },
                ],
            }
        ]

        # 处理消息
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_input, video_input = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_input,
            videos=video_input,
            padding=True,
            return_tensors="pt",
        )

        # 将输入转换为列表以便拼接
        inputs = {key: value.tolist() for key, value in inputs.items() if isinstance(value, torch.Tensor)} #tensor -> list,为了方便拼接
        instruction = inputs

        # 构建 input_ids、attention_mask 和 labels
        input_ids = (
                instruction["input_ids"][0]
                + response["input_ids"]
                + [self.tokenizer.pad_token_id]
        )
        attention_mask = (
                instruction["attention_mask"][0] + response["attention_mask"] + [1]
        )
        labels = (
                [-100] * len(instruction["input_ids"][0])
                + response["input_ids"]
                + [self.tokenizer.pad_token_id]
        )

        # 转换为张量
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        labels = torch.tensor(labels)

        # 准备额外输入
        output_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "gt_masks": None,
            "original_images": None
        }

        # 如果处理器提供，添加 pixel_values 和 image_grid_thw
        if "pixel_values" in inputs:
            output_dict["pixel_values"] = torch.tensor(inputs["pixel_values"])
        if "image_grid_thw" in inputs:
            output_dict["image_grid_thw"] = torch.tensor(inputs["image_grid_thw"]).squeeze(0)
        if "pixel_values_videos" in inputs:
            output_dict["pixel_values_videos"] = [torch.tensor(inputs["pixel_values_videos"])]
        if "video_grid_thw" in inputs:
            output_dict["video_grid_thw"] = [torch.tensor(inputs["video_grid_thw"]).squeeze(0)]
        return output_dict


# 示例用法
if __name__ == "__main__":
    from transformers import AutoProcessor, AutoTokenizer

    # 加载处理器和分词器（替换为实际 Qwen2.5-VL 路径）
    processor = AutoProcessor.from_pretrained("path/to/qwen2.5-vl")
    tokenizer = AutoTokenizer.from_pretrained("path/to/qwen2.5-vl")

    # 初始化数据集
    dataset = VideoDataset(
        base_data_dir="/path/to/something-something",
        processor=processor,
        tokenizer=tokenizer,
        split="train",
        num_frames=16,
    )

    # 获取一个样本
    sample = dataset[0]
    print({k: v.shape if isinstance(v, torch.Tensor) else v for k, v in sample.items()})