import json
import os
import random
from tqdm import tqdm
from decord import VideoReader
from moviepy.editor import VideoFileClip, concatenate_videoclips


def select_random_subsets(input_list):
    # 检查输入列表是否至少有2个元素
    if len(input_list) < 2:
        return []

    result = []
    remaining_elements = input_list.copy()

    # 估算最大迭代次数（最坏情况下每次选2个元素）
    max_iterations = len(input_list) // 2 + 1

    # 使用 tqdm 包装迭代
    for _ in tqdm(range(max_iterations), desc="Generating subsets"):
        # 检查剩余元素是否足够，且是否需要继续生成
        if len(remaining_elements) < 2 or (len(result) >= 3 and len(remaining_elements) < 2):
            break
        # 随机选择2到4个元素（不超过剩余元素数量）
        num_elements = random.randint(2, min(4, len(remaining_elements)))
        # 随机选择元素
        subset = random.sample(remaining_elements, num_elements)
        result.append(subset)
        # 从剩余元素中移除已选择的元素
        for item in subset:
            remaining_elements.remove(item)

    return result


def merge_videos(video_paths, output_path):
    total_frames = 0
    for video_path in video_paths:
        vr = VideoReader(video_path)
        frame_count = len(vr)
        fps = vr.get_avg_fps()
        duration = frame_count / fps
        total_frames += frame_count
        # print(f"{os.path.basename(video_path)}: {frame_count} 帧, {duration:.2f} 秒, {fps:.2f} fps")

    clips = [VideoFileClip(path) for path in video_paths]
    final_clip = concatenate_videoclips(clips, method="compose")

    # 使用简单的 tqdm 进度条，兼容 moviepy 1.0.3
    final_clip.write_videofile(
        output_path,
        verbose=False,    # 禁用 FFmpeg 日志
        logger=None,     # 禁用 moviepy 日志
        fps=clips[0].fps,
        audio_codec="aac"  # 确保音频兼容性
    )

    for clip in clips:
        clip.close()
    final_clip.close()


def generate_json(label_base_dir, file_base_dir, save_path):
    labels = []
    split_list = ["train.json", "validation.json"]
    for split in split_list:
        with open(os.path.join(label_base_dir, split), "r") as f:
            labels += json.load(f)
    print("Load labels completed!")
    # labels = random.choices(labels, k=1000)
    sampled_labels = select_random_subsets(labels)
    print("Sampling completed!")
    new_labels = {}
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        os.mkdir(os.path.join(save_path, "videos"))
    for idx, s_label in tqdm(enumerate(sampled_labels), total=len(sampled_labels)):
        duration = 0.0
        timestamps, sentences, ids, video_paths = [], [], [], []
        for l in s_label:
            ids.append(l["id"])
            video_path = os.path.join(file_base_dir, "{}.webm".format(l["id"]))
            vr = VideoReader(video_path)
            # 获取总帧数和平均帧率
            frame_count = len(vr)
            avg_fps = vr.get_avg_fps()
            # 计算时长（秒）
            timestamp = [duration, duration+frame_count/avg_fps]
            duration = timestamp[1]
            timestamps.append(timestamp)
            sentences.append(l["label"])
            video_paths.append(video_path)
        new_labels[str(idx).zfill(5)] = {
            "duration": duration,
            "timestamps": timestamps,
            "sentences": sentences,
            "ids": ids,
        }
        merge_videos(video_paths, os.path.join(save_path, "videos", "{}.webm".format(str(idx).zfill(5))))
    with open(os.path.join(save_path, "label.json"), "w") as f:
        json.dump(new_labels, f)
    print("All completed!")

if __name__ == "__main__":
    label_base_dir = "/media/automan/6E94666294662CB1/A_Content/SSv2/labels"
    file_base_dir = "/media/automan/6E94666294662CB1/A_Content/SSv2/20bn-something-something-v2"
    save_dir = "/media/automan/6E94666294662CB1/A_Content/SSv2/captions"
    generate_json(label_base_dir, file_base_dir, save_dir)
