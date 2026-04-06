"""Step 1: 从原始12通道WAV提取标准数据

将每个原始文件的每个chirp响应提取为独立的3通道标准格式文件
格式: {原始文件名}_mic5_{波型}_{Chirp编号:02d}_extracted_{日期}.wav
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy.io import wavfile

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.chirp_params import WAVE_PARAMS, WAVE_TYPES, SAMPLE_RATE, MIC_CHANNEL
from config.raw_files import VALID_FILES, RAW_DATA_DIR
from pipeline.config import PipelineConfig
from pipeline.logging import setup_logging, logger


class ChirpExtractor:
    """Chirp提取器"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.timestamp = datetime.now().strftime('%Y%m%d')
        self.sr = SAMPLE_RATE

    def extract_single_file(self, raw_file: str) -> int:
        """
        从单个原始文件提取标准数据

        输出每个chirp为独立的3通道文件：
        - ch1: 喇叭激励（对应窗口）
        - ch2: mic响应（对应窗口）
        - ch3: 0（占位）

        Returns:
            成功提取的chirp数量
        """
        raw_path = self.config.raw_data_dir / raw_file
        if not raw_path.exists():
            logger.error(f"原始文件不存在: {raw_path}")
            return 0

        # 读取原始12通道数据
        sr, data = wavfile.read(raw_path)
        logger.info(f"读取: {raw_file}, 形状: {data.shape}")

        # 提取原始文件名（不含扩展名）
        raw_basename = raw_file.replace('.wav', '')
        success_count = 0

        # 按波型、按chirp提取
        for wave_type in WAVE_TYPES:
            params = WAVE_PARAMS[wave_type]
            emission_times = params['emission_times']
            delay_min = params['delay_min']
            delay_max = params['delay_max']
            duration = params['duration']

            for i, emission_time in enumerate(emission_times):
                chirp_index = i + 1  # 1-indexed

                # 计算响应窗口
                resp_start_time = emission_time + delay_min
                resp_end_time = emission_time + delay_max + duration

                # 转换为样本索引
                resp_start = int(round(resp_start_time * sr))
                resp_end = int(round(resp_end_time * sr))

                if resp_end > len(data):
                    resp_end = len(data)
                if resp_start >= len(data):
                    continue

                # 创建3通道输出数组（固定2秒长度）
                TWO_SECONDS = int(2.0 * sr)  # 384000 samples
                output = np.zeros((TWO_SECONDS, 3), dtype=data.dtype)

                # ch1: 喇叭激励（原始通道1，从emission_time开始填充2秒）
                chirp_start = int(emission_time * sr)
                speaker_end = min(chirp_start + TWO_SECONDS, len(data))
                speaker_len = speaker_end - chirp_start
                output[:speaker_len, 0] = data[chirp_start:chirp_start + speaker_len, 0]

                # ch2: mic响应（原始通道7，从emission_time开始填充2秒）
                mic_end = min(chirp_start + TWO_SECONDS, len(data))
                mic_len = mic_end - chirp_start
                output[:mic_len, 1] = data[chirp_start:chirp_start + mic_len, MIC_CHANNEL]

                # ch3: 0（占位，与标准格式一致）

                # 生成输出文件名
                # 格式: {原始文件名}_mic5_{波型}_{Chirp编号:02d}_extracted_{日期}.wav
                output_filename = f"{raw_basename}_mic5_{wave_type}_{chirp_index:02d}_extracted_{self.timestamp}.wav"
                output_path = self.config.standard_data_dir / output_filename

                wavfile.write(output_path, sr, output)
                success_count += 1
                logger.debug(f"  -> {output_filename}")

        logger.info(f"  {raw_file}: 提取了 {success_count} 个chirp文件")
        return success_count

    def run(self) -> int:
        """运行提取流程"""
        self.config.ensure_dirs()

        total_success = 0
        for raw_file in VALID_FILES:
            count = self.extract_single_file(raw_file)
            total_success += count

        logger.info(f"Step 1 完成: 共提取 {total_success} 个标准WAV")
        return total_success


def main():
    setup_logging('INFO')

    project_root = Path(__file__).parent.parent
    config = PipelineConfig(project_root)

    extractor = ChirpExtractor(config)
    extractor.run()


if __name__ == '__main__':
    main()
