"""Step 2: 匹配滤波处理

对per-chirp标准WAV应用匹配滤波，输出per-chirp matched文件
格式: {原始文件名}_{mic}_{波型}_{Chirp编号:02d}_matched_{日期}.wav
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy import signal
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.chirp_params import WAVE_PARAMS, WAVE_TYPES, SAMPLE_RATE, ACTIVE_MICS, get_mic_channel_name, get_wave_params_for_mic, SAVE_FORMAT
from pipeline.config import PipelineConfig
from pipeline.logging import setup_logging, logger


class MatchedFilterProcessor:
    """匹配滤波处理器"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.timestamp = datetime.now().strftime('%Y%m%d')
        self.sr = SAMPLE_RATE

    def process_single(self, wav_path: Path, mic_name: str, mic_idx: int) -> bool:
        """处理单个per-chirp标准文件"""
        # 验证文件
        if not wav_path.exists():
            logger.error(f"文件不存在: {wav_path}")
            return False

        filename = wav_path.name
        if '_matched_' in filename:
            logger.debug(f"跳过已处理的: {filename}")
            return True

        # 读取per-chirp数据（3通道：ch1喇叭，ch2麦克风，ch3占位）
        if wav_path.suffix == '.npz':
            loaded = np.load(wav_path)
            data = loaded['data']
            sr = SAMPLE_RATE
        else:
            sr, data = wavfile.read(wav_path)
        window_len = len(data)
        logger.debug(f"处理: {filename}, 形状: {data.shape}")

        # 从文件名解析波型和chirp编号
        # 格式: {raw_name}_{mic}_{wave}_{chirp_index:02d}_extracted_{date}.wav
        parts = filename.replace('.wav', '').split('_')
        # 找到mic的位置，然后提取波型和chirp编号
        try:
            mic_idx_pos = parts.index(mic_name)
            wave_type = parts[mic_idx_pos + 1]
            chirp_str = parts[mic_idx_pos + 2]  # 如 "01", "02"
            chirp_index = int(chirp_str)  # 已经是1-indexed
        except (ValueError, IndexError):
            logger.error(f"文件名格式错误，无法解析: {filename}")
            return False

        # 获取该MIC对应距离的波型参数
        wave_params = get_wave_params_for_mic(mic_idx)
        params = wave_params[wave_type]
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

        # 响应窗口参数
        delay_min_samples = int(delay_min * self.sr)
        chirp_duration_samples = int(duration * self.sr)
        resp_window_start = delay_min_samples
        resp_window_end = int((delay_max + duration) * self.sr)
        resp_window_len = resp_window_end - resp_window_start

        # ch3: 匹配滤波结果
        # 在响应窗口范围内做互相关
        # chirp模板取chirp长度部分
        # 响应取响应窗口部分
        matched = signal.correlate(
            response[resp_window_start:resp_window_end],
            chirp_template[:chirp_duration_samples],
            mode='full'
        )

        # 计算scale_factor
        matched_max = np.max(np.abs(matched))
        orig_max = np.max(np.abs(response[resp_window_start:resp_window_end]))

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

        # ch3: 匹配滤波结果，从delay_min位置开始放置
        # matched长度 = resp_window_len + chirp_duration_samples - 1
        # 只需要前resp_window_len部分（对应响应窗口）
        matched_len = min(len(matched_scaled), resp_window_len)
        output_data[delay_min_samples:delay_min_samples + matched_len, 2] = matched_scaled[:matched_len]

        # 生成输出文件名
        # 格式: {raw_name}_{mic}_{wave}_{chirp_index:02d}_matched_{date}.{ext}
        # 替换 _extracted_ 为 _matched_
        base_name = filename.replace('.wav', '').replace('.npz', '')
        if '_extracted_' in base_name:
            # 提取日期部分
            parts_split = base_name.split('_extracted_')
            new_filename = f"{parts_split[0]}_matched_{parts_split[1]}"
        else:
            new_filename = base_name.replace('_extracted_', '_matched_')
        ext = 'npz' if SAVE_FORMAT == 'npz' else 'wav'
        new_filename = f"{new_filename}.{ext}"

        # clip到int32范围并保存
        int32_max = 2147483647
        output_data = np.clip(output_data, -int32_max, int32_max).astype(np.int32)

        output_path = self.config.get_step2_dir(mic_name) / new_filename
        if SAVE_FORMAT == 'npz':
            np.savez_compressed(output_path, data=output_data)
        else:
            wavfile.write(output_path, sr, output_data)
        logger.debug(f"  -> {new_filename}")

        return True

    def run(self) -> int:
        """运行匹配滤波"""
        total_success = 0
        total_files = 0

        for mic_idx in ACTIVE_MICS:
            mic_name = get_mic_channel_name(mic_idx)
            # 查找per-chirp标准数据文件（npz或wav格式）
            files = []
            files.extend(self.config.get_step1_dir(mic_name).glob(f'*_{mic_name}_*_extracted_*.npz'))
            files.extend(self.config.get_step1_dir(mic_name).glob(f'*_{mic_name}_*_extracted_*.wav'))
            logger.info(f"{mic_name}: 找到 {len(files)} 个per-chirp标准数据文件")
            total_files += len(files)

            for wav_path in files:
                if self.process_single(wav_path, mic_name, mic_idx):
                    total_success += 1

        logger.info(f"Step 2 完成: 处理了 {total_success}/{total_files} 个文件")
        return total_success


def main():
    setup_logging('INFO')

    project_root = Path(__file__).parent.parent
    config = PipelineConfig(project_root)

    processor = MatchedFilterProcessor(config)
    processor.run()


if __name__ == '__main__':
    main()
