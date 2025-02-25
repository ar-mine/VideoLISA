import json
import time
import os
import yt_dlp
import subprocess
from moviepy.video.io.VideoFileClip import VideoFileClip
from concurrent.futures import ThreadPoolExecutor, as_completed

def type_download(refer, source_type, save_path):
    print('refer:' + refer + ' ' + source_type + ' start download')
    if source_type == 'video+audio':
        ydl_opts['format'] = 'bestvideo+bestaudio'
    else:
        ydl_opts['format'] = 'best' + source_type
    ydl_opts['outtmpl'] = save_path + os.path.sep + '{}.%(ext)s'.format(refer)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print('downloading...')
            ydl.download([PREFIX + refer])
    except yt_dlp.utils.DownloadError as e:
        # 三种youtube视频源丢失的报错
        if not 'unavailable' in str(e) and not 'Private' in str(e) and not 'terminated' in str(e) and not 'age' in str(e) and not 'removed' in str(e):
            if source_type == 'video+audio':
                ret = subprocess.call(
                    'yt-dlp --format bestvideo+bestaudio ' + PREFIX + refer + ' -o ' + save_path + ' -R 50')
            else:
                ret = subprocess.call('yt-dlp --format best' + source_type + PREFIX + refer + ' -o ' + save_path + ' -R 50')
            if ret:
                print(refer + ' ' + source_type + str(e))
                time.sleep(1)
        else:
            print(refer + ' ' + source_type + str(e))
            time.sleep(1)
        return 1
    else:
        time.sleep(1)
        print(refer + ' ' + source_type + ' done')
        return 0

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
            print(start_time, end_time, video.duration)
            return 1

        # 裁剪视频
        trimmed_video = video.subclip(start_time, end_time)

        # 保存裁剪后的视频
        trimmed_video.write_videofile(output_path, codec="libx264", audio_codec="aac")

        print(f"视频已成功裁剪并保存到 {output_path}")
        return 0
    except Exception as e:
        print(input_path)
        print(f"发生错误: {e}")
        return 1

def process_label(label, skip_list):
    # 如果 label 在跳过列表中，则直接返回
    if label['id'] in skip_list:
        return None, None, None
    video_path = os.path.join(SAVE_PATH, label['id'])
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    # 如果文件夹为空，则尝试下载视频
    if not os.listdir(video_path):
        if type_download(label['id'], 'video+audio', video_path):
            # 下载失败
            return "", label, True
    # 检查下载后的视频文件（支持多种格式）
    for ext in ("mp4", "mkv", "webm"):
        original_video_path = os.path.join(video_path, '{}.{}'.format(label['id'], ext))
        if os.path.exists(original_video_path):
            break
    clip_video_path = os.path.join(video_path, '{}_vtime.mp4'.format(label['id']))
    # 如果剪辑视频不存在，则根据给定时间段进行裁剪
    if not os.path.exists(clip_video_path):
        start_sec = label['meta']['split'][0]
        end_sec = label['meta']['split'][1]
        ret = trim_video(original_video_path, clip_video_path, start_sec, end_sec)
        if ret: # 裁剪失败
            return "", label, True
    return clip_video_path, label, False

def download_dataset_parallel(data_list, max_data_number=0, skip_list=[], num_workers=2, start=0):
    # 根据 max_data_number 选择要处理的标签
    if max_data_number > 0:
        max_labels = data_list[start:max_data_number]
    else:
        max_labels = data_list[start:]

    clip_video_paths = []
    modify_labels = []
    fail_list = []

    # 使用 ThreadPoolExecutor 进行多线程处理
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_label, label, skip_list): label for label in max_labels}
        for future in as_completed(futures):
            result = future.result()
            # 如果当前 label 被跳过，则 result 为 (None, None, None)
            if result[0] is None:
                continue
            clip_path, label, failed = result
            clip_video_paths.append(clip_path)
            modify_labels.append(label)
            if failed:
                fail_list.append(label['id'])
    print("下载失败的 id 列表：", fail_list)
    return modify_labels, clip_video_paths

if __name__ == "__main__":
    dataset = json.load(open("./labels/stage2.json"))
    print(len(dataset))

    SAVE_PATH = "/media/automan/ExSpace/Projects/VideoLISA/dataset/videos"
    PREFIX = 'https://www.youtube.com/watch?v='
    ydl_opts = {
        'quiet': True,  # 启动安静模式。如果与——verbose一起使用，则将日志打印到stderr
        # 'username': USER_NAME,
        # 'password': PASSWORD,
        # 'logger': MyLogger(),
        # 'retries': 50
        # 'postprocessors': [{
        #     'key': 'FFmpegExtractAudio',
        #     'preferredcodec': suffix,
        #     'preferredquality': '192',
        # }],
    }

    skip_list = [
        # Unavailable
        "Q595wIHZ1aA", "e4tFnhEmwRM", "anr4W-UEvOM", "C0WhoTm9qrE", "WFtV4rV_ARg", "pYcVzxzpZgM", "d0Kn7Rt1D0U",
        "MntLqxO53Ls", "oDs05I0NyR8", "BGKk5HmKZ5I", "oV2kjFLrXrw", "ZZBxws6iEFE", "Ssz7Zf8VQEs", "iNRhe7lHlr4",
        "qHfWmdvEYB0", "R2Zri26D3Nw", "i4uX2KlyGs8", "8rJJHvKNy3A", "p2tEpmT6fOA", "lg6CcvPc8o4", "0jMp51s9CqY",
        "D2nYM3Q3WU4", "okmbII6BPNY", "UwIt_Ny63gI", "WVNCfFJcQGs", "5zSSMjYMYFM", "-EBKPJH5pGI", "IPjAr1aKgEE",
        "RCPgYpsxClI", "v9GJojs46u0", "UIjS4TY197I", "KZ0YiFQjN9w", "GT_SEXX8jdw", "MbjzpyXSDFk", "eUZbGJBCstg",
        "x2c4RhlxMKo", "PakzukYqlKY", "0dpmQTU29go", "DKivVNCjOE4", "99xWswuND0o",
        # Private
        "WKB2mKTQo24", "nNGUKSD3j_s", "YcAqgVarvsM", "1RfQmpM4nkc", "CUoitqAxsOo", "d4VQV0Hr8w8", "zc9UWW6cX8M",
        "1RfQmpM4nkc", "LPjYqoSkUi0", "4ReLtzwQrY8", "1buEFjm0X2I", "002CTRIvZOI", "yD_aRo8Prdw", "xrHMonjyNs0",
        "OEES9_UMOo4", "mG8Q9HrgYMs", "HiHQfrJYxu4", "7Jcz763W1M4", "dFXdNJYK3ds", "S-EtZ6iG5LU", "nVTgLroGK_U",
        "4vZQ8KWA0QE", "BS69gdubk-o", "flVUzFKBcGs", "MqJt2-_na3g", "EtDB5NrtmWg", "l59IhXoJDpY", "Vu48LYFflOU",
        "iYNDBZBjFtM", "gnPNGEn_DRU", "jSv0FIVspRA", "vEV9P4Q3x_Q",
        # Broken
        "-9FE-Gs40tc", "-sUhkJcjxYQ", "OWf_S_LhV34",
        # 裁剪时间超出范围或无效
        'TGGI_jsmzN0', 'wob3xMIqMHU', "nK_A8_HTa-4", "hJcm64kAVOo", "UWH8RWYln0I", "0kClu-uz7aA", "JfWu_gqhp0I",
        "KzJbe547lZk", "W4AppJDCmOk", "JZEb3jKhIu8", "TjPJ5A2rLn8", "oCkNImUpu4w", "El8FU2McyMY", "CrtrvKHWrh4",
        "ahqH_jdyNF8", "8NFWFaAEQ9I", "puPSMmttimg", "g316YN9RX6U", "jD_I6KsCZus", "KfN1xztUHhY", "HORGyUit_mY",
        "DaPVmpjuJKw", "XghkJ6-bN7M",
        # 发生错误: 'video_fps'
        "RfCPnOwMYaI",
    ]
    labels, _ = download_dataset_parallel(dataset,
                                          max_data_number=1200,
                                          skip_list=skip_list, num_workers=2)
    json.dump(labels, open("labels/index.json", "w"))
