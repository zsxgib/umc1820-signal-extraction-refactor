"""npz ↔ wav 格式互转脚本

支持：
1. 单个文件转换：npz → wav 或 wav → npz
2. 批量转换：指定目录下的所有 npz/wav 文件

用法：
    # 转换单个文件
    python convert_format.py input.npz                    # npz → wav
    python convert_format.py input.wav                     # wav → npz
    python convert_format.py input.npz -o output.wav      # 指定输出路径

    # 批量转换目录
    python convert_format.py --dir step1_extracted/mic1 --to npz   # wav → npz
    python convert_format.py --dir step3_accumulated/mic1 --to wav # npz → wav
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from scipy.io import wavfile


def npz_to_wav(npz_path: Path, output_path: Path = None) -> bool:
    """将 npz 文件转换为 wav 格式"""
    if output_path is None:
        output_path = npz_path.with_suffix('.wav')

    try:
        loaded = np.load(npz_path)
        data = loaded['data']

        # 如果是 float64 类型，先转换为 int32
        if data.dtype == np.float64:
            int32_max = 2147483647
            data = np.clip(data, -int32_max, int32_max).astype(np.int32)

        wavfile.write(output_path, 192000, data)
        print(f"  转换成功: {npz_path.name} → {output_path.name}")
        return True
    except Exception as e:
        print(f"  转换失败: {npz_path.name}, 错误: {e}")
        return False


def wav_to_npz(wav_path: Path, output_path: Path = None) -> bool:
    """将 wav 文件转换为 npz 格式"""
    if output_path is None:
        output_path = wav_path.with_suffix('.npz')

    try:
        sr, data = wavfile.read(wav_path)

        # 转换为 float64 以便压缩
        if data.dtype != np.float64:
            data = data.astype(np.float64)

        np.savez_compressed(output_path, data=data)
        print(f"  转换成功: {wav_path.name} → {output_path.name}")
        return True
    except Exception as e:
        print(f"  转换失败: {wav_path.name}, 错误: {e}")
        return False


def convert_file(input_path: Path, output_path: Path = None, target_format: str = None) -> bool:
    """转换单个文件"""
    input_path = Path(input_path)

    if not input_path.exists():
        print(f"  文件不存在: {input_path}")
        return False

    suffix = input_path.suffix.lower()

    if suffix == '.npz':
        return npz_to_wav(input_path, output_path)
    elif suffix == '.wav':
        return wav_to_npz(input_path, output_path)
    else:
        print(f"  不支持的格式: {suffix}")
        return False


def convert_dir(dir_path: Path, target_format: str) -> int:
    """批量转换目录下的所有文件"""
    dir_path = Path(dir_path)

    if not dir_path.exists():
        print(f"  目录不存在: {dir_path}")
        return 0

    if target_format == 'npz':
        files = list(dir_path.glob('*.wav'))
        convert_func = wav_to_npz
    elif target_format == 'wav':
        files = list(dir_path.glob('*.npz'))
        convert_func = npz_to_wav
    else:
        print(f"  不支持的目标格式: {target_format}")
        return 0

    if not files:
        print(f"  目录下没有需要转换的 {target_format} 文件")
        return 0

    print(f"  找到 {len(files)} 个文件需要转换")

    success_count = 0
    for f in files:
        if convert_func(f):
            success_count += 1

    return success_count


def main():
    parser = argparse.ArgumentParser(description='npz ↔ wav 格式互转')
    parser.add_argument('input', nargs='?', help='输入文件或目录')
    parser.add_argument('-o', '--output', help='输出路径（单文件转换时使用）')
    parser.add_argument('-d', '--dir', dest='dir_path', help='批量转换的目录')
    parser.add_argument('-t', '--to', dest='target_format', choices=['npz', 'wav'],
                        help='目标格式（批量转换时使用）')

    args = parser.parse_args()

    # 批量转换模式
    if args.dir_path:
        if not args.target_format:
            print("错误: 批量转换需要指定 --to 参数")
            sys.exit(1)
        count = convert_dir(args.dir_path, args.target_format)
        print(f"批量转换完成: {count} 个文件")
        return

    # 单文件转换模式
    if not args.input:
        parser.print_help()
        sys.exit(1)

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None

    if input_path.is_dir():
        print("错误: 请使用 --dir 参数进行批量转换")
        sys.exit(1)

    if convert_file(input_path, output_path):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
