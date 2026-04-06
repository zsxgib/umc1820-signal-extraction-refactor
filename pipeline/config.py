"""Pipeline 路径配置"""

from pathlib import Path
from config.chirp_params import ACTIVE_MICS, get_mic_channel_name

class PipelineConfig:
    """Pipeline 路径配置"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.raw_data_dir = Path('/home/zsx/umc1820-2026-03-22-1/umc1820_refactor/save-sound')
        self.step1_base_dir = project_root / 'step1_extracted'   # Step 1: extracted文件根目录
        self.step2_base_dir = project_root / 'step2_matched'     # Step 2: matched文件根目录
        self.step3_base_dir = project_root / 'step3_accumulated' # Step 3: accumulated文件根目录

        # 为每个MIC创建子目录
        self.mic_dirs = {}
        for mic_idx in ACTIVE_MICS:
            mic_name = get_mic_channel_name(mic_idx)
            self.mic_dirs[mic_name] = {
                'step1': self.step1_base_dir / mic_name,
                'step2': self.step2_base_dir / mic_name,
                'step3': self.step3_base_dir / mic_name,
            }

    def get_step1_dir(self, mic_name: str) -> Path:
        """获取指定MIC的Step1输出目录"""
        return self.mic_dirs[mic_name]['step1']

    def get_step2_dir(self, mic_name: str) -> Path:
        """获取指定MIC的Step2输出目录"""
        return self.mic_dirs[mic_name]['step2']

    def get_step3_dir(self, mic_name: str) -> Path:
        """获取指定MIC的Step3输出目录"""
        return self.mic_dirs[mic_name]['step3']

    def ensure_dirs(self):
        """确保目录存在"""
        self.step1_base_dir.mkdir(parents=True, exist_ok=True)
        self.step2_base_dir.mkdir(parents=True, exist_ok=True)
        self.step3_base_dir.mkdir(parents=True, exist_ok=True)
        for mic_name, dirs in self.mic_dirs.items():
            dirs['step1'].mkdir(parents=True, exist_ok=True)
            dirs['step2'].mkdir(parents=True, exist_ok=True)
            dirs['step3'].mkdir(parents=True, exist_ok=True)