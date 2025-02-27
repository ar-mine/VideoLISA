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
        trimmed_video.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)

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

    skip_list = [
        # DOWNLOAD_ERROR_TRANSIENT
        'ZZBxws6iEFE', 'Ssz7Zf8VQEs', 'zc9UWW6cX8M', 'p2tEpmT6fOA', 'iNRhe7lHlr4', 'R2Zri26D3Nw', 'i4uX2KlyGs8',
        '8rJJHvKNy3A', 'lg6CcvPc8o4', 'D2nYM3Q3WU4', 'okmbII6BPNY', 'WVNCfFJcQGs', '5zSSMjYMYFM', 'IPjAr1aKgEE',
        'nVTgLroGK_U', 'UIjS4TY197I', 'RCPgYpsxClI', 'v9GJojs46u0', 'GT_SEXX8jdw', 'MbjzpyXSDFk', 'x2c4RhlxMKo',
        'PakzukYqlKY', '0dpmQTU29go', 'DKivVNCjOE4', '99xWswuND0o', 'Q595wIHZ1aA', 'e4tFnhEmwRM', 'anr4W-UEvOM',
        'WFtV4rV_ARg', 'pYcVzxzpZgM', 'd0Kn7Rt1D0U', 'MntLqxO53Ls', 'oDs05I0NyR8', 'BGKk5HmKZ5I', 'nl6IrjZzSt0',
        'dks0koqKOaA', 'Lhp96fthmIc', 'u93kgT-uAIA', 'QAmQ7MpTsTw', 'gEO-sTgwdl8', '4fLkr-4gTEQ', 'D_zs9VxCfd4',
        'ySJ0RCzXLOQ', 'MYN3QPEkFOs', 'MC_YBvWP9vc', 'et7xveZ5nqs', '720B0332K74', 'ycA_hdiF79c', '8hY-9YgU4-Y',
        'qFLQ-r-qW7c', 'FATGIiF3LWY', 'wefa7sGS_kg', 'LGHzfNJWBlA', 'OISjWi_Jmvo', 'jDMYuXiTs04', 'PUImat87F74',
        'WHxP6LQnd3M', 'rQjljdIi9Pg', 'UDr1IVAbPlo',
        # DOWNLOAD_ERROR_AUTH
        'oV2kjFLrXrw', 'CUoitqAxsOo', 'd4VQV0Hr8w8', '1RfQmpM4nkc', 'LPjYqoSkUi0', 'qHfWmdvEYB0', '4ReLtzwQrY8',
        '1buEFjm0X2I', '0jMp51s9CqY', 'UwIt_Ny63gI', '002CTRIvZOI', 'yD_aRo8Prdw', 'xrHMonjyNs0', 'OEES9_UMOo4',
        '7Jcz763W1M4', 'HiHQfrJYxu4', 'dFXdNJYK3ds', 'S-EtZ6iG5LU', 'KZ0YiFQjN9w', '4vZQ8KWA0QE', 'BS69gdubk-o',
        'flVUzFKBcGs', 'MqJt2-_na3g', 'EtDB5NrtmWg', 'Vu48LYFflOU', 'l59IhXoJDpY', 'iYNDBZBjFtM', 'gnPNGEn_DRU',
        'jSv0FIVspRA', 'vEV9P4Q3x_Q', 'WKB2mKTQo24', 'YcAqgVarvsM', 'U2oQcqvkCCk', '_JUgKIy_0Ys', 'Xz74m4cAosg',
        'qaibFRRbEO4', 'hT-VFY4exGM', 'tIvqIcko4R4', 'JlPcZGjXwHQ', 'oweXrPnj-Ic', 'fOIR0BU6QiU', '_lfLYKk3bvI',
        'EXMEuOWNzEk', 'aABM8x9B4D4', 'Tl1tA1-nlHI', 'xMTsW_Q0IwE', 'V9xAOc9pBvE', 'h9vNMuje1uY', '2KUe96v06os',
        'KlZ7P4DUL5w', 'WHwNCqy_LHY', 'ZP8V_j7C2Ws', 'Bp3nzbaOj3A', 'aZz8Hh_EsEI', 'n3KPxcihUQ8', 'xPU_4RnjiDo',
        'EA5EZr6h1lc', 'zRz5bKPbgKg',
        # DOWNLOAD_ERROR_EXIST
        # TRIM_ERROR_INVALID_TIME
        'JfWu_gqhp0I', 'KzJbe547lZk', '-EBKPJH5pGI', 'W4AppJDCmOk', 'JZEb3jKhIu8', 'g316YN9RX6U', 'TjPJ5A2rLn8',
        'oCkNImUpu4w', 'El8FU2McyMY', 'CrtrvKHWrh4', '8NFWFaAEQ9I', 'ahqH_jdyNF8', 'puPSMmttimg', 'XghkJ6-bN7M',
        'jD_I6KsCZus', 'KfN1xztUHhY', 'DaPVmpjuJKw', 'HORGyUit_mY', 'TGGI_jsmzN0', 'wob3xMIqMHU', 'nK_A8_HTa-4',
        'hJcm64kAVOo', 'UWH8RWYln0I', '0kClu-uz7aA', 'NRZ7bZi3Q-o', '832YsbRqxeU', 'YDfZWud2-84',
        # TRIM_ERROR_EXCEPTION
        '-9FE-Gs40tc', 'RfCPnOwMYaI', '-sUhkJcjxYQ', 'tLKVaizPItA', '-qKhLrUziCQ', 'L6FuZtM9LeI',
    ]
    MAX_DATA_NUM = 3000
    labels, _ = download_dataset_parallel(dataset,
                                          max_data_number=MAX_DATA_NUM,
                                          skip_list=skip_list, num_workers=2)
    data_len = len(labels)
    train_len = int(data_len * 0.9)
    json.dump(labels[:train_len], open(f"labels/train-{MAX_DATA_NUM}.json", "w"), indent=2)
    json.dump(labels[train_len:], open(f"labels/test-{MAX_DATA_NUM}.json", "w"), indent=2)
