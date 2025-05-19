import json
import os
from tqdm import tqdm
from decord import VideoReader
from moviepy.editor import VideoFileClip


def crop_video(input_path, output_path, start_time, end_time):
    # 使用 moviepy 加载和裁剪视频
    clip = VideoFileClip(input_path)
    # 裁剪视频（start_time 到 end_time，单位为秒）
    cropped_clip = clip.subclip(start_time, end_time)

    # 写入裁剪后的视频
    cropped_clip.write_videofile(
        output_path,
        verbose=False,    # 禁用 FFmpeg 日志
        logger=None,     # 禁用 moviepy 日志
        fps=clip.fps,    # 保持原始帧率
        codec="libx264", # 确保视频编码兼容
        audio=False
    )

    # 关闭剪辑以释放资源
    clip.close()
    cropped_clip.close()


def generate_json(label_base_dir, file_base_dir, save_path):
    with open(os.path.join(label_base_dir, "fho_main.json"), "r") as f:
        labels = json.load(f)
    print("Load labels completed!")
    new_labels = {}
    count = 0
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        os.mkdir(os.path.join(save_path, "videos"))
    for video_idx, video in tqdm(enumerate(labels['videos']), total=len(labels['videos'])):
        video_path = os.path.join(file_base_dir, "{}.mp4".format(video["video_uid"]))
        if not os.path.exists(video_path):
            continue
        for interval_idx, interval in enumerate(video['annotated_intervals']):
            start_sec = interval['start_sec']
            duration = 0.0
            timestamps, sentences, ids = [], [], []
            if len(interval['narrated_actions']) == 0:
                continue
            for action_idx, action in enumerate(interval['narrated_actions']):
                if action['start_sec']-start_sec > duration:
                    duration = action['end_sec']-start_sec
                    timestamps.append([action['start_sec']-start_sec, action['end_sec']-start_sec])
                    sentences.append(action['narration_text'].replace('#C C', 'Human'))
                    ids.append([video_idx, interval_idx, action_idx])
                if duration >= 60.0:
                    break

            new_labels[str(count).zfill(5)] = {
                "duration": duration,
                "timestamps": timestamps,
                "sentences": sentences,
                "ids": ids,
            }
            crop_video(video_path,
                       os.path.join(save_path, "videos", "{}.mp4".format(str(count).zfill(5))),
                       start_time=start_sec, end_time=start_sec+duration)
            count += 1
    with open(os.path.join(save_path, "label.json"), "w") as f:
        json.dump(new_labels, f)
    print("All completed!")


if __name__ == "__main__":
    label_base_dir = "/media/automan/6E94666294662CB1/A_Content/Datasets/ego4d_data/v2/annotations"
    file_base_dir = "/media/automan/6E94666294662CB1/A_Content/Datasets/ego4d_data/v2/full_scale"
    save_dir = "/media/automan/6E94666294662CB1/A_Content/Datasets/ego4d_data/captions"
    generate_json(label_base_dir, file_base_dir, save_dir)
