#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess

def is_video_corrupted(video_path):
    """
    使用 ffmpeg 检查视频文件是否存在错误信息
    参数:
      video_path: 视频文件路径
    返回:
      True: 视频可能损坏
      False: 视频正常
    """
    # 运行 ffmpeg 命令，使用 -v error 仅输出错误信息，-f null 表示不保存输出结果
    cmd = ['ffmpeg', '-v', 'error', '-i', video_path, '-f', 'null', '-']
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # 如果 stderr 有输出，说明视频可能存在问题
    return bool(result.stderr.strip())

def main():
    folder = input("请输入需要检查的视频文件夹路径：").strip()
    if not os.path.isdir(folder):
        print("输入的路径不是有效的文件夹路径")
        return

    # 定义常见的视频文件扩展名
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv']
    corrupted_files = []

    # 遍历文件夹中的所有文件（包括子文件夹）
    for root, dirs, files in os.walk(folder):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_path = os.path.join(root, file)
                print(f"正在检查: {video_path}")
                if is_video_corrupted(video_path):
                    corrupted_files.append(video_path)

    print("\n检查完毕！")
    if corrupted_files:
        print("以下视频文件可能损坏：")
        for path in corrupted_files:
            print(path)
    else:
        print("未发现损坏的视频文件。")

if __name__ == '__main__':
    main()
