import os
import json
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from moviepy.editor import VideoFileClip
import yt_dlp

# 定义错误码
SUCCESS = 0
DOWNLOAD_ERROR_TRANSIENT = 1      # 可通过子进程重试的下载错误
DOWNLOAD_ERROR_AUTH = 2           # 认证问题的下载错误
DOWNLOAD_ERROR_EXIST = 3          # 资源不存在的下载错误
DOWNLOAD_ERROR_UNKNOWN = -1

TRIM_SUCCESS = 0
TRIM_ERROR_INVALID_TIME = 4       # 裁剪时间不合理
TRIM_ERROR_EXCEPTION = 5          # 裁剪过程出现其他异常

# 下载视频的函数，根据不同情况返回不同的错误码
def type_download(refer, source_type, save_path):
    print('refer:' + refer + ' ' + source_type + ' start download')
    if source_type == 'video+audio':
        ydl_opts['format'] = 'bestvideo+bestaudio'
    else:
        ydl_opts['format'] = 'best' + source_type
    ydl_opts['outtmpl'] = os.path.join(save_path, '{}.%(ext)s'.format(refer))

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print('downloading...')
            ydl.download([PREFIX + refer])
    except yt_dlp.utils.DownloadError as e:
        err_str = str(e)
        # 如果错误信息不包含下面关键词，则尝试用子进程重试下载
        if not ('unavailable' in err_str or 'Private' in err_str or 'terminated' in err_str or 'age' in err_str or 'removed' in err_str):
            if source_type == 'video+audio':
                ret = subprocess.call(
                    'yt-dlp --format bestvideo+bestaudio ' + PREFIX + refer + ' -o ' + save_path + ' -R 50',
                    shell=True)
            else:
                ret = subprocess.call(
                    'yt-dlp --format best' + source_type + ' ' + PREFIX + refer + ' -o ' + save_path + ' -R 50',
                    shell=True)
            if ret != 0:
                print(refer + ' ' + source_type + " subprocess error: " + err_str)
                time.sleep(1)
                return DOWNLOAD_ERROR_TRANSIENT
            else:
                time.sleep(1)
                print(refer + ' ' + source_type + ' done (via subprocess)')
                return SUCCESS
        else:
            print(refer + ' ' + source_type + " permanent error: " + err_str)
            time.sleep(1)
            if 'Private' in err_str or 'terminated' in err_str or 'age' in err_str:
                return DOWNLOAD_ERROR_AUTH
            if 'unavailable' in err_str or 'removed' in err_str:
                return DOWNLOAD_ERROR_TRANSIENT
    else:
        time.sleep(1)
        print(refer + ' ' + source_type + ' done')
        return SUCCESS

# 裁剪视频的函数，根据情况返回不同的错误码
def trim_video(input_path, output_path, start_time, end_time):
    """
    裁剪视频并保存
    :param input_path: 输入视频文件路径
    :param output_path: 输出裁剪后的视频文件路径
    :param start_time: 裁剪开始时间（秒）
    :param end_time: 裁剪结束时间（秒）
    """
    try:
        # 读取视频文件
        video = VideoFileClip(input_path)

        # 确保裁剪时间合理
        if start_time < 0 or end_time > video.duration or start_time >= end_time:
            print(input_path)
            print("错误：裁剪时间超出范围或无效！")
            print("start_time:", start_time, "end_time:", end_time, "duration:", video.duration)
            return TRIM_ERROR_INVALID_TIME

        # 裁剪视频
        trimmed_video = video.subclip(start_time, end_time)

        # 保存裁剪后的视频（可通过logger=None关闭进度条）
        trimmed_video.write_videofile(output_path, codec="libx264", fps=30, audio_codec="aac", logger=None)

        print(f"视频已成功裁剪并保存到 {output_path}")
        return TRIM_SUCCESS
    except Exception as e:
        print(input_path)
        print(f"发生错误: {e}")
        return TRIM_ERROR_EXCEPTION

# 处理单个label，下载视频并裁剪，返回裁剪后的视频路径、label及错误码（0为成功，其它表示不同错误类型）
def process_label(label, skip_list):
    if label['id'] in skip_list:
        return None, label, None  # 跳过的label直接返回
    video_path = os.path.join(SAVE_PATH, label['id'])
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    # 如果文件夹为空，则尝试下载视频
    if not os.listdir(video_path):
        download_result = type_download(label['id'], 'video+audio', video_path)
        if download_result != SUCCESS:
            # 下载失败时返回对应错误码
            return "", label, download_result
    # 检查下载后的视频文件（支持多种格式）
    for ext in ("mp4", "mkv", "webm"):
        original_video_path = os.path.join(video_path, '{}.{}'.format(label['id'], ext))
        if os.path.exists(original_video_path):
            break
    else:
        # 如果没有找到视频文件，视为永久性下载错误
        return "", label, DOWNLOAD_ERROR_UNKNOWN

    clip_video_path = os.path.join(video_path, '{}_vtime.mp4'.format(label['id']))
    # 如果剪辑后的视频不存在，则进行裁剪
    if not os.path.exists(clip_video_path):
        start_sec = label['meta']['split'][0]
        end_sec = label['meta']['split'][1]
        trim_result = trim_video(original_video_path, clip_video_path, start_sec, end_sec)
        if trim_result != TRIM_SUCCESS:  # 裁剪失败，返回对应错误码
            return "", label, trim_result
    return clip_video_path, label, SUCCESS

# 并行处理下载与裁剪任务，同时收集不同错误类型的失败列表
def download_dataset_parallel(data_list, max_data_number=0, skip_list=[], num_workers=2, start=0):
    if max_data_number > 0:
        max_labels = data_list[start:max_data_number]
    else:
        max_labels = data_list[start:]

    clip_video_paths = []
    modify_labels = []
    fail_list = [[] for _ in range(5)]  # 列表中存放 (label_id, error_code) 的元组

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_label, label, skip_list): label for label in max_labels}
        for future in as_completed(futures):
            clip_path, label, error_code = future.result()
            # 如果当前 label 被跳过，则 clip_path 为 None
            if clip_path is None:
                continue
            clip_video_paths.append(clip_path)
            modify_labels.append(label)
            if error_code != SUCCESS and error_code > 0:
                fail_list[error_code-1].append(label['id'])
    print('DOWNLOAD_ERROR_TRANSIENT:', fail_list[0])
    print('DOWNLOAD_ERROR_AUTH:', fail_list[1])
    print('DOWNLOAD_ERROR_EXIST:', fail_list[2])
    print('TRIM_ERROR_INVALID_TIME:', fail_list[3])
    print('TRIM_ERROR_EXCEPTION:', fail_list[4])
    return modify_labels, clip_video_paths

# 配置及主程序入口
if __name__ == "__main__":
    dataset = json.load(open("./labels/stage2.json"))
    print("标签总数：", len(dataset))

    SAVE_PATH = "/media/automan/ExSpace/Projects/VideoLISA/dataset/videos"
    PREFIX = 'https://www.youtube.com/watch?v='
    ydl_opts = {
        'quiet': True,  # 启动安静模式，将日志输出降到最低
        # 其它可选参数根据需求添加
    }

    skip_list = json.load(open("./skips/skip-list-20000.json"))["data"]
    START = 10000
    MAX_DATA_NUM = 20000
    labels, _ = download_dataset_parallel(dataset,
                                          start=START,
                                          max_data_number=MAX_DATA_NUM,
                                          skip_list=skip_list, num_workers=2)
    data_len = len(labels)
    train_len = int(data_len * 0.9)
    json.dump(labels[:train_len], open(f"labels/train-{MAX_DATA_NUM}.json", "w"), indent=2)
    json.dump(labels[train_len:], open(f"labels/test-{MAX_DATA_NUM}.json", "w"), indent=2)
