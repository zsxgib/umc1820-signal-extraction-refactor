"""Step 3: 相干累积

对per-chirp matched文件进行相干累积，输出：
1. 标准格式：5个波型文件（每个波型所有chirp在正确时间位置）
2. 102秒完整版本：时间轴拼接后的完整信号
"""

import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
from scipy.io import wavfile
from scipy import signal

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.chirp_params import WAVE_PARAMS, WAVE_TYPES, SAMPLE_RATE, MIC_CHANNEL
from config.raw_files import VALID_FILES
from pipeline.config import PipelineConfig
from pipeline.logging import setup_logging, logger


class CoherentAccumulator:
    """相干累积处理器"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.timestamp = datetime.now().strftime('%Y%m%d')
        self.sr = SAMPLE_RATE
        self.total_samples = int(102.0 * self.sr)  # 102秒

    def find_delay_by_crosscorr(self, ref_signal: np.ndarray, target_signal: np.ndarray) -> int:
        """使用互相关找目标信号相对于参考信号的时延"""
        corr = signal.correlate(target_signal, ref_signal, mode='full')
        n = len(ref_signal)
        center = n - 1
        peak_idx = np.argmax(np.abs(corr))
        delay = peak_idx - center
        return delay

    def linear_shift(self, seg: np.ndarray, offset: int) -> np.ndarray:
        """线性移位（替代np.roll循环移位）"""
        if offset > 0:
            shifted = np.pad(seg, (offset, 0), mode='constant')[:len(seg)]
        elif offset < 0:
            shifted = np.pad(seg, (0, -offset), mode='constant')[-len(seg):]
        else:
            shifted = seg
        return shifted

    def accumulate_per_chirp(self, files: list) -> tuple:
        """
        对一组per-chirp matched文件进行相干累积

        Args:
            files: 同一波型同一chirp编号的所有matched文件路径列表

        Returns:
            tuple: (accumulated_data, window_len) 累积后的数据和窗口长度
        """
        if not files:
            return None, 0

        # 从文件名解析波型和chirp编号
        filename = files[0].name
        parts = filename.replace('.wav', '').split('_')
        try:
            mic5_idx = parts.index('mic5')
            wave_type = parts[mic5_idx + 1]
            chirp_str = parts[mic5_idx + 2]
            chirp_index = int(chirp_str)
        except (ValueError, IndexError):
            logger.error(f"文件名格式错误，无法解析: {filename}")
            return None, 0

        # 读取第一个文件获取窗口长度
        sr, first_data = wavfile.read(files[0])
        window_len = len(first_data)

        # ch1是麦克风响应，用于计算scale_factor
        orig_max = np.max(np.abs(first_data[:, 1].astype(np.float64)))

        # 收集所有文件的ch2（匹配滤波结果）进行累积
        all_segments = []
        for f in files:
            _, data = wavfile.read(f)
            if len(data) != window_len:
                if len(data) < window_len:
                    data = np.pad(data, ((0, window_len - len(data)), (0, 0)), mode='constant')
                else:
                    data = data[:window_len]
            all_segments.append(data[:, 2].astype(np.float64))

        # 使用第一个segment作为参考进行cross-correlation对齐
        ref_segment = all_segments[0]

        # 累积
        accumulated = np.zeros(window_len, dtype=np.float64)
        accumulated += ref_segment

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

        accumulated = accumulated * scale

        return accumulated, window_len

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

        # 限制每个组合最多8个文件
        NUM_MICS = 8
        for key in file_groups:
            file_groups[key] = sorted(file_groups[key])[:NUM_MICS]

        # 创建102秒完整输出缓冲区
        # 第1通道: 喇叭参考
        # 第2通道: mic响应完整102秒（从原始WAV加载）
        # 第3通道: mic参考截取per-chirp
        # 第4通道: 相干累积结果
        output_buffer = np.zeros((self.total_samples, 4), dtype=np.float64)
        logger.info(f"创建输出缓冲区: {self.total_samples} 样本 ({self.total_samples/self.sr:.1f}秒)")

        # 加载原始WAV的完整102秒mic信号（第2通道）
        raw_wav_path = self.config.raw_data_dir / VALID_FILES[0]
        if raw_wav_path.exists():
            _, raw_data = wavfile.read(raw_wav_path)
            mic_full = raw_data[:, MIC_CHANNEL].astype(np.float64)
            output_buffer[:, 1] = mic_full  # 第2通道: 完整102秒mic响应
            logger.info(f"已加载原始mic信号: {len(mic_full)} 样本")

        # 收集每个chirp的speaker参考信号（用于102秒版本的第1通道）
        # key: (wave_type, chirp_index), value: (ch0_data, ch1_data, window_len)
        chirp_signals = {}

        # 按波型处理
        for wave_type in WAVE_TYPES:
            params = WAVE_PARAMS[wave_type]
            emission_times = params['emission_times']
            delay_min = params['delay_min']
            delay_max = params['delay_max']
            duration = params['duration']

            logger.info(f"处理波型: {wave_type}")

            # 创建该波型的累积buffer（响应窗口长度）
            first_emission = emission_times[0]
            first_resp_start_time = first_emission + delay_min
            first_resp_end_time = first_emission + delay_max
            first_start_sample = int(first_resp_start_time * self.sr)
            first_end_sample = int(first_resp_end_time * self.sr) + int(duration * self.sr)
            wave_window_len = first_end_sample - first_start_sample

            # 波型累积buffer
            wave_buffer = np.zeros(wave_window_len, dtype=np.float64)

            # 对该波型的每个chirp进行累积
            for chirp_idx, emission_time in enumerate(emission_times):
                chirp_index = chirp_idx + 1  # 1-indexed
                key = (wave_type, chirp_index)

                if key not in file_groups:
                    logger.warning(f"  {wave_type} chirp {chirp_index}: 没有找到文件")
                    continue

                files = file_groups[key]

                # 读取speaker参考和麦克风信号（从第一个文件）
                sr_ref, ref_data = wavfile.read(files[0])
                ref_ch0 = ref_data[:, 0].astype(np.float64)
                ref_ch1 = ref_data[:, 1].astype(np.float64)
                chirp_signals[key] = (ref_ch0, ref_ch1, len(ref_ch0))

                accumulated, window_len = self.accumulate_per_chirp(files)

                if accumulated is None:
                    continue

                # 计算该chirp在时间轴上的位置
                resp_start_time = emission_time + delay_min
                resp_end_time = emission_time + delay_max
                start_sample = int(resp_start_time * self.sr)
                end_sample = int(resp_end_time * self.sr) + int(duration * self.sr)

                # 找到累积结果中的峰值位置
                abs_accum = np.abs(accumulated)
                peak_idx = np.argmax(abs_accum)

                # 从峰值位置提取响应窗口长度的数据
                response_len = end_sample - start_sample  # 响应窗口长度
                actual_len = min(response_len, wave_window_len)

                # 从峰值位置开始提取
                if peak_idx + actual_len <= len(accumulated):
                    wave_buffer[:actual_len] += accumulated[peak_idx:peak_idx + actual_len]
                else:
                    # 峰值太靠后，提取到末尾
                    available = len(accumulated) - peak_idx
                    wave_buffer[:available] += accumulated[peak_idx:]

                logger.debug(f"    {wave_type} chirp {chirp_index}: 累积了 {len(files)} 个文件")

            # 计算scale_factor
            orig_max = np.max(np.abs(output_buffer[:, 1]))
            accum_max = np.max(np.abs(wave_buffer))

            if accum_max > 0 and orig_max > 0:
                scale = orig_max * 0.8 / accum_max
            else:
                scale = 1.0

            logger.debug(f"  {wave_type}: orig_max={orig_max:.2e}, accum_max={accum_max:.2e}, scale={scale:.4f}")

            wave_buffer = wave_buffer * scale

            # 将波型累积结果写入output_buffer的第4通道
            # 期望：每个波型只有1个信号（10个chirp叠加），放在第一个chirp位置
            end_sample = min(first_start_sample + wave_window_len, self.total_samples)
            actual_len = end_sample - first_start_sample
            output_buffer[first_start_sample:end_sample, 3] = wave_buffer[:actual_len]

            # 保存标准格式文件（该波型所有chirp在正确时间位置）
            # 创建4通道输出
            wave_output = np.zeros((wave_window_len, 4), dtype=np.float64)

            # 第1通道: 喇叭参考，从 0ms 开始（对应 emission_time）
            # 第2通道: 麦克风响应per-chirp，从 delay_min 开始（对应 emission_time + delay_min）
            # 第3通道: 麦克风响应per-chirp（参考）
            # 第4通道: 累积结果
            delay_min_samples = int(delay_min * self.sr)
            chirp_duration_samples = int(duration * self.sr)

            # 从第一个 chirp 的数据填充
            first_key = (wave_type, 1)
            if first_key in chirp_signals:
                ref_ch0, ref_ch1, _ = chirp_signals[first_key]
                # 第1通道: 喇叭 chirp 从 0 开始
                actual_ch0_len = min(chirp_duration_samples, len(ref_ch0), wave_window_len)
                wave_output[:actual_ch0_len, 0] = ref_ch0[:actual_ch0_len]
                # 第2通道: 麦克风响应从 delay_min 开始
                actual_ch1_len = min(wave_window_len - delay_min_samples, len(ref_ch1))
                if actual_ch1_len > 0:
                    wave_output[delay_min_samples:delay_min_samples + actual_ch1_len, 1] = ref_ch1[:actual_ch1_len]
                # 第3通道: 同第2通道，mic参考
                if actual_ch1_len > 0:
                    wave_output[delay_min_samples:delay_min_samples + actual_ch1_len, 2] = ref_ch1[:actual_ch1_len]

            wave_output[:, 3] = wave_buffer

            # clip并保存
            int32_max = 2147483647
            wave_output = np.clip(wave_output, -int32_max, int32_max).astype(np.int32)

            output_filename = f"{wave_type}_accumulated_{self.timestamp}.wav"
            output_path = self.config.output_dir / output_filename
            wavfile.write(output_path, self.sr, wave_output)
            logger.info(f"  -> {output_filename}")

        # 填充102秒版本的第1通道（喇叭参考）和第3通道（mic参考per-chirp）
        logger.info("填充102秒版本的第1通道和第3通道...")
        for (wave_type, chirp_index), (ch0_data, ch1_data, window_len) in chirp_signals.items():
            params = WAVE_PARAMS[wave_type]
            emission_times = params['emission_times']
            delay_min = params['delay_min']
            duration = params['duration']

            chirp_idx = chirp_index - 1  # 转为0-indexed
            if chirp_idx >= len(emission_times):
                continue
            emission_time = emission_times[chirp_idx]

            # 第1通道: 喇叭参考从emission_time开始
            start_sample = int(emission_time * self.sr)

            # 第1通道: 长度是 chirp 持续时间
            ch0_len = int(duration * self.sr)
            actual_ch0_len = min(ch0_len, len(ch0_data), self.total_samples - start_sample)
            if actual_ch0_len > 0:
                output_buffer[start_sample:start_sample + actual_ch0_len, 0] = ch0_data[:actual_ch0_len]

            # 第3通道: mic参考per-chirp从emission_time + delay_min开始
            resp_start_sample = int((emission_time + delay_min) * self.sr)
            actual_ch1_len = min(window_len, len(ch1_data), self.total_samples - resp_start_sample)
            if actual_ch1_len > 0:
                output_buffer[resp_start_sample:resp_start_sample + actual_ch1_len, 2] = ch1_data[:actual_ch1_len]

        # 生成102秒完整版本
        output_filename_full = f"coherent_accumulation_{self.timestamp}.wav"
        output_path_full = self.config.output_dir / output_filename_full

        output_int = np.clip(output_buffer, -2147483647, 2147483647).astype(np.int32)
        wavfile.write(output_path_full, self.sr, output_int)
        logger.info(f"Step 3 完成: {output_filename_full}")
        logger.info(f"  形状: {output_int.shape}")
        logger.info(f"  时长: {len(output_int)/self.sr:.1f}秒")

        return True


def main():
    setup_logging('INFO')

    project_root = Path(__file__).parent.parent
    config = PipelineConfig(project_root)

    accumulator = CoherentAccumulator(config)
    accumulator.run()


if __name__ == '__main__':
    main()
