"""Step 3: 相干累积

对per-chirp matched文件进行相干累积，输出per-chirp accumulated文件
格式: {波型}_{Chirp编号:02d}_accumulated_{日期}.wav
"""

import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
from scipy.io import wavfile
from scipy import signal

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.chirp_params import WAVE_PARAMS, WAVE_TYPES, SAMPLE_RATE
from pipeline.config import PipelineConfig
from pipeline.logging import setup_logging, logger


class CoherentAccumulator:
    """相干累积处理器"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.timestamp = datetime.now().strftime('%Y%m%d')
        self.sr = SAMPLE_RATE

    def find_delay_by_crosscorr(self, ref_signal: np.ndarray, target_signal: np.ndarray) -> int:
        """
        使用互相关找目标信号相对于参考信号的时延

        Args:
            ref_signal: 参考信号 (1D array)
            target_signal: 目标信号 (1D array)，长度须与ref_signal相同

        Returns:
            int: 时延值（正值表示target滞后，负值表示target超前）
                 用 offset = -delay 进行线性移位可对齐
        """
        corr = signal.correlate(target_signal, ref_signal, mode='full')
        n = len(ref_signal)
        center = n - 1
        peak_idx = np.argmax(np.abs(corr))
        delay = peak_idx - center
        return delay

    def linear_shift(self, seg: np.ndarray, offset: int) -> np.ndarray:
        """
        线性移位（替代np.roll循环移位）

        Args:
            seg: 输入信号
            offset: 偏移量，正表示向右移，负表示向左移

        Returns:
            移位后的信号，长度与输入相同，边缘补零
        """
        if offset > 0:
            shifted = np.pad(seg, (offset, 0), mode='constant')[:len(seg)]
        elif offset < 0:
            shifted = np.pad(seg, (0, -offset), mode='constant')[-len(seg):]
        else:
            shifted = seg
        return shifted

    def accumulate_single(self, files: list) -> bool:
        """
        对一组per-chirp matched文件进行相干累积

        Args:
            files: 同一波型同一chirp编号的所有matched文件路径列表

        Returns:
            是否成功
        """
        if not files:
            return False

        # 从文件名解析波型和chirp编号
        # 格式: {raw_name}_mic5_{wave}_{chirp:02d}_matched_{date}.wav
        filename = files[0].name
        parts = filename.replace('.wav', '').split('_')
        try:
            mic5_idx = parts.index('mic5')
            wave_type = parts[mic5_idx + 1]
            chirp_str = parts[mic5_idx + 2]
            chirp_index = int(chirp_str)  # 1-indexed
        except (ValueError, IndexError):
            logger.error(f"文件名格式错误，无法解析: {filename}")
            return False

        logger.info(f"  {wave_type} chirp {chirp_index}: 累积 {len(files)} 个文件")

        # 读取第一个文件获取窗口长度和ch1参考值
        sr, first_data = wavfile.read(files[0])
        window_len = len(first_data)

        # ch1是麦克风响应，用于计算scale_factor
        orig_max = np.max(np.abs(first_data[:, 1].astype(np.float64)))

        # 收集所有文件的ch2（匹配滤波结果）进行累积
        all_segments = []
        for f in files:
            _, data = wavfile.read(f)
            if len(data) != window_len:
                # 补零到相同长度
                if len(data) < window_len:
                    data = np.pad(data, ((0, window_len - len(data)), (0, 0)), mode='constant')
                else:
                    data = data[:window_len]
            all_segments.append(data[:, 2].astype(np.float64))  # ch2: 匹配滤波结果

        # 使用第一个segment作为参考
        ref_segment = all_segments[0]

        # 创建累积数组
        accumulated = np.zeros(window_len, dtype=np.float64)
        accumulated += ref_segment

        # 对其他segment用互相关找时延，线性移位对齐后累加
        for seg in all_segments[1:]:
            delay = self.find_delay_by_crosscorr(ref_segment, seg)
            shifted = self.linear_shift(seg, -delay)
            accumulated += shifted

        # 计算scale_factor
        accum_max = np.max(np.abs(accumulated))
        if accum_max > 0 and orig_max > 0:
            scale = orig_max * 0.8 / accum_max
        else:
            scale = 1.0

        logger.debug(f"    orig_max={orig_max:.2e}, accum_max={accum_max:.2e}, scale={scale:.4f}")

        # 应用scale
        accumulated = accumulated * scale

        # 创建3通道输出
        output_data = np.zeros((window_len, 3), dtype=np.float64)
        output_data[:, 0] = first_data[:, 0].astype(np.float64)  # ch0: 喇叭参考
        output_data[:, 1] = first_data[:, 1].astype(np.float64)  # ch1: 麦克风响应
        output_data[:, 2] = accumulated  # ch2: 相干累积结果

        # clip到int32范围
        int32_max = 2147483647
        output_data = np.clip(output_data, -int32_max, int32_max).astype(np.int32)

        # 生成输出文件名
        # 格式: {波型}_{Chirp编号:02d}_accumulated_{日期}.wav
        output_filename = f"{wave_type}_{chirp_index:02d}_accumulated_{self.timestamp}.wav"
        output_path = self.config.output_dir / output_filename

        wavfile.write(output_path, sr, output_data)
        logger.debug(f"    -> {output_filename}")

        return True

    def run(self) -> bool:
        """运行相干累积"""
        self.config.ensure_dirs()

        # 查找所有per-chirp matched文件
        matched_files = list(self.config.step2_output_dir.glob('*_mic5_*_matched_*.wav'))
        logger.info(f"找到 {len(matched_files)} 个per-chirp matched文件")

        if not matched_files:
            logger.warning("没有找到 matched 文件!")
            return False

        # 按波型和chirp编号分组
        file_groups = defaultdict(list)
        for f in matched_files:
            filename = f.name
            parts = filename.replace('.wav', '').split('_')
            try:
                mic5_idx = parts.index('mic5')
                wave_type = parts[mic5_idx + 1]
                chirp_str = parts[mic5_idx + 2]
                chirp_index = int(chirp_str)
                key = (wave_type, chirp_index)
                file_groups[key].append(f)
            except (ValueError, IndexError):
                logger.warning(f"跳过无法解析的文件: {filename}")
                continue

        logger.info(f"共有 {len(file_groups)} 个波型/chirp组合")

        # 限制每个组合最多8个文件（与原始版本一致）
        NUM_MICS = 8
        for key in file_groups:
            file_groups[key] = sorted(file_groups[key])[:NUM_MICS]

        # 对每个波型chirp组合进行累积
        success_count = 0
        for (wave_type, chirp_index), files in sorted(file_groups.items()):
            if self.accumulate_single(files):
                success_count += 1

        logger.info(f"Step 3 完成: 累积了 {success_count}/{len(file_groups)} 个文件")
        return success_count > 0


def main():
    setup_logging('INFO')

    project_root = Path(__file__).parent.parent
    config = PipelineConfig(project_root)

    accumulator = CoherentAccumulator(config)
    accumulator.run()


if __name__ == '__main__':
    main()
