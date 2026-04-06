"""Step 2: 匹配滤波处理

对per-chirp标准WAV应用匹配滤波，输出per-chirp matched文件
格式: {原始文件名}_mic5_{波型}_{Chirp编号:02d}_matched_{日期}.wav
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy import signal
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.chirp_params import WAVE_PARAMS, WAVE_TYPES, SAMPLE_RATE
from pipeline.config import PipelineConfig
from pipeline.logging import setup_logging, logger


class MatchedFilterProcessor:
    """匹配滤波处理器"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.timestamp = datetime.now().strftime('%Y%m%d')
        self.sr = SAMPLE_RATE

    def process_single(self, wav_path: Path) -> bool:
        """处理单个per-chirp标准WAV文件"""
        # 验证文件
        if not wav_path.exists():
            logger.error(f"文件不存在: {wav_path}")
            return False

        filename = wav_path.name
        if '_matched_' in filename:
            logger.debug(f"跳过已处理的: {filename}")
            return True

        # 读取per-chirp数据（3通道：ch1喇叭，ch2麦克风，ch3占位）
        sr, data = wavfile.read(wav_path)
        window_len = len(data)
        logger.debug(f"处理: {filename}, 形状: {data.shape}")

        # 从文件名解析波型和chirp编号
        # 格式: {raw_name}_mic5_{wave}_{chirp_index:02d}_extracted_{date}.wav
        parts = filename.replace('.wav', '').split('_')
        # 找到mic5的位置，然后提取波型和chirp编号
        try:
            mic5_idx = parts.index('mic5')
            wave_type = parts[mic5_idx + 1]
            chirp_str = parts[mic5_idx + 2]  # 如 "01", "02"
            chirp_index = int(chirp_str)  # 已经是1-indexed
        except (ValueError, IndexError):
            logger.error(f"文件名格式错误，无法解析: {filename}")
            return False

        # 获取波型参数
        params = WAVE_PARAMS[wave_type]
        duration = params['duration']
        delay_min = params['delay_min']
        delay_max = params['delay_max']

        # ch1: 喇叭参考（不变）
        chirp_template = data[:, 0].astype(np.float64)

        # ch2: 麦克风响应
        response = data[:, 1].astype(np.float64)

        if len(chirp_template) == 0 or len(response) == 0:
            logger.warning(f"模板或响应为空: {filename}")
            return False

        # 匹配滤波 (互相关)
        matched = signal.correlate(response, chirp_template, mode='same')

        # 计算scale_factor
        matched_max = np.max(np.abs(matched))
        orig_max = np.max(np.abs(response))

        if matched_max > 0 and orig_max > 0:
            scale_factor = orig_max / matched_max
        else:
            scale_factor = 1.0

        # 应用scale_factor
        matched_scaled = matched * scale_factor

        # 创建3通道输出
        output_data = np.zeros((window_len, 3), dtype=np.float64)
        output_data[:, 0] = chirp_template  # ch1: 喇叭参考
        output_data[:, 1] = response  # ch2: 麦克风响应

        # ch3: 匹配滤波结果，按表格要求的响应窗口起始位置(delay_min)放置
        # 响应窗口长度 = delay_max - delay_min + duration
        resp_window_len = int((delay_max - delay_min + duration) * self.sr)
        delay_min_samples = int(delay_min * self.sr)

        # 找到matched峰值位置
        peak_idx = np.argmax(np.abs(matched_scaled))

        # 从峰值位置提取响应窗口长度的数据
        resp_start = peak_idx - delay_min_samples
        resp_end = resp_start + resp_window_len

        # 边界处理
        if resp_start < 0:
            resp_end -= resp_start
            resp_start = 0
        if resp_end > window_len:
            resp_end = window_len
            resp_start = max(0, resp_end - resp_window_len)

        # 提取matched结果并放入ch3的delay_min位置
        matched_window = matched_scaled[resp_start:resp_end]
        actual_len = len(matched_window)

        # 放入output_data的ch3，从delay_min位置开始
        if delay_min_samples + actual_len <= window_len:
            output_data[delay_min_samples:delay_min_samples + actual_len, 2] = matched_window
        else:
            # 超长时截断
            available_len = window_len - delay_min_samples
            output_data[delay_min_samples:window_len, 2] = matched_window[:available_len]

        # 生成输出文件名
        # 格式: {raw_name}_mic5_{wave}_{chirp_index:02d}_matched_{date}.wav
        # 替换 _extracted_ 为 _matched_
        base_name = filename.replace('.wav', '')
        if '_extracted_' in base_name:
            # 提取日期部分
            parts_split = base_name.split('_extracted_')
            new_filename = f"{parts_split[0]}_matched_{parts_split[1]}"
        else:
            new_filename = base_name.replace('_extracted_', '_matched_')
        new_filename = f"{new_filename}.wav"

        # clip到int32范围并保存
        int32_max = 2147483647
        output_data = np.clip(output_data, -int32_max, int32_max).astype(np.int32)

        output_path = self.config.step2_output_dir / new_filename
        wavfile.write(output_path, sr, output_data)
        logger.debug(f"  -> {new_filename}")

        return True

    def run(self) -> int:
        """运行匹配滤波"""
        # 查找per-chirp标准数据文件
        files = list(self.config.standard_data_dir.glob('*_mic5_*_extracted_*.wav'))
        logger.info(f"找到 {len(files)} 个per-chirp标准数据文件")

        success = 0
        for wav_path in files:
            if self.process_single(wav_path):
                success += 1

        logger.info(f"Step 2 完成: 处理了 {success}/{len(files)} 个文件")
        return success


def main():
    setup_logging('INFO')

    project_root = Path(__file__).parent.parent
    config = PipelineConfig(project_root)

    processor = MatchedFilterProcessor(config)
    processor.run()


if __name__ == '__main__':
    main()
