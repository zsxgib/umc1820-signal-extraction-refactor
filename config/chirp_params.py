"""波型参数定义"""

WAVE_PARAMS = {
    'PS': {
        'freq_start': 1000,
        'freq_end': 2000,
        'duration': 0.020,
        'delay_min': 0.0173,
        'delay_max': 0.0479,
        'emission_times': [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
    },
    'SV': {
        'freq_start': 800,
        'freq_end': 1200,
        'duration': 0.025,
        'delay_min': 0.035,
        'delay_max': 0.085,
        'emission_times': [22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0, 40.0],
    },
    'SH': {
        'freq_start': 800,
        'freq_end': 1200,
        'duration': 0.025,
        'delay_min': 0.0382,
        'delay_max': 0.0882,
        'emission_times': [42.0, 44.0, 46.0, 48.0, 50.0, 52.0, 54.0, 56.0, 58.0, 60.0],
    },
    'A0H': {
        'freq_start': 500,
        'freq_end': 800,
        'duration': 0.025,
        'delay_min': 0.065,
        'delay_max': 0.145,
        'emission_times': [62.0, 64.0, 66.0, 68.0, 70.0, 72.0, 74.0, 76.0, 78.0, 80.0],
    },
    'A0L': {
        'freq_start': 100,
        'freq_end': 300,
        'duration': 0.100,
        'delay_min': 0.200,
        'delay_max': 1.300,
        'emission_times': [82.0, 84.0, 86.0, 88.0, 90.0, 92.0, 94.0, 96.0, 98.0, 100.0],
    },
}

WAVE_TYPES = ['PS', 'SV', 'SH', 'A0H', 'A0L']
SAMPLE_RATE = 192000
NUM_MICS = 8
SPEAKER_CHANNELS = [0, 1]  # 喇叭左/右

# 同时处理的MIC列表 (1-indexed通道号)
# MIC索引 -> 1-indexed通道号: MIC1=第3通道, MIC2=第4通道, ..., MIC8=第10通道
ACTIVE_MICS = [1, 2, 3, 4, 5, 6, 7, 8]  # 处理全部8个MIC

# 存储格式配置: 'npz' 或 'wav'
# npz: 压缩比约48x，节省98%空间
# wav: 标准格式，文件更大
SAVE_FORMAT = 'npz'

def get_mic_channel_name(mic_index: int) -> str:
    """根据MIC索引(1-8)返回通道名"""
    return f'mic{mic_index}'

def get_mic_channel_1indexed(mic_index: int) -> int:
    """根据MIC索引(1-8)返回1-indexed通道号"""
    return mic_index + 2  # MIC1=第3通道, MIC8=第10通道

# MIC距离配置（1-indexed）
# 默认全部设为50米，后续可根据实际情况调整
MIC_DISTANCES = {
    1: 50,  # MIC1 在50米
    2: 50,  # MIC2 在50米
    3: 50,  # MIC3 在50米
    4: 50,  # MIC4 在50米
    5: 50,  # MIC5 在50米
    6: 50,  # MIC6 在50米
    7: 50,  # MIC7 在50米
    8: 50,  # MIC8 在50米
}

# 按距离索引的完整参数表（来自HDPE膜声学检测完整参数表）
# delay_min, delay_max = 截取窗口的起始和结束时间（秒）
WAVE_PARAMS_30M = {
    'PS': {'delay_min': 0.0036, 'delay_max': 0.034, 'duration': 0.020},
    'SV': {'delay_min': 0.005, 'delay_max': 0.055, 'duration': 0.025},
    'SH': {'delay_min': 0.0066, 'delay_max': 0.0566, 'duration': 0.025},
    'A0H': {'delay_min': 0.0275, 'delay_max': 0.090, 'duration': 0.025},
    'A0L': {'delay_min': 0.050, 'delay_max': 0.700, 'duration': 0.100},
}

WAVE_PARAMS_40M = {
    'PS': {'delay_min': 0.0082, 'delay_max': 0.0386, 'duration': 0.020},
    'SV': {'delay_min': 0.015, 'delay_max': 0.065, 'duration': 0.025},
    'SH': {'delay_min': 0.0171, 'delay_max': 0.0671, 'duration': 0.025},
    'A0H': {'delay_min': 0.040, 'delay_max': 0.110, 'duration': 0.025},
    'A0L': {'delay_min': 0.100, 'delay_max': 0.900, 'duration': 0.100},
}

WAVE_PARAMS_50M = {
    'PS': {'delay_min': 0.0127, 'delay_max': 0.0433, 'duration': 0.020},
    'SV': {'delay_min': 0.025, 'delay_max': 0.075, 'duration': 0.025},
    'SH': {'delay_min': 0.0276, 'delay_max': 0.0776, 'duration': 0.025},
    'A0H': {'delay_min': 0.0525, 'delay_max': 0.125, 'duration': 0.025},
    'A0L': {'delay_min': 0.150, 'delay_max': 1.100, 'duration': 0.100},
}

WAVE_PARAMS_60M = {
    'PS': {'delay_min': 0.0173, 'delay_max': 0.0479, 'duration': 0.020},
    'SV': {'delay_min': 0.035, 'delay_max': 0.085, 'duration': 0.025},
    'SH': {'delay_min': 0.0382, 'delay_max': 0.0882, 'duration': 0.025},
    'A0H': {'delay_min': 0.065, 'delay_max': 0.145, 'duration': 0.025},
    'A0L': {'delay_min': 0.200, 'delay_max': 1.300, 'duration': 0.100},
}

WAVE_PARAMS_70M = {
    'PS': {'delay_min': 0.0218, 'delay_max': 0.0526, 'duration': 0.020},
    'SV': {'delay_min': 0.045, 'delay_max': 0.095, 'duration': 0.025},
    'SH': {'delay_min': 0.0487, 'delay_max': 0.0987, 'duration': 0.025},
    'A0H': {'delay_min': 0.0775, 'delay_max': 0.165, 'duration': 0.025},
    'A0L': {'delay_min': 0.250, 'delay_max': 1.500, 'duration': 0.100},
}

WAVE_PARAMS_80M = {
    'PS': {'delay_min': 0.0264, 'delay_max': 0.0572, 'duration': 0.020},
    'SV': {'delay_min': 0.055, 'delay_max': 0.105, 'duration': 0.025},
    'SH': {'delay_min': 0.0592, 'delay_max': 0.1092, 'duration': 0.025},
    'A0H': {'delay_min': 0.090, 'delay_max': 0.185, 'duration': 0.025},
    'A0L': {'delay_min': 0.300, 'delay_max': 1.700, 'duration': 0.100},
}

WAVE_PARAMS_90M = {
    'PS': {'delay_min': 0.0309, 'delay_max': 0.0619, 'duration': 0.020},
    'SV': {'delay_min': 0.065, 'delay_max': 0.115, 'duration': 0.025},
    'SH': {'delay_min': 0.0697, 'delay_max': 0.1197, 'duration': 0.025},
    'A0H': {'delay_min': 0.1025, 'delay_max': 0.205, 'duration': 0.025},
    'A0L': {'delay_min': 0.350, 'delay_max': 1.900, 'duration': 0.100},
}

WAVE_PARAMS_100M = {
    'PS': {'delay_min': 0.0355, 'delay_max': 0.0655, 'duration': 0.020},
    'SV': {'delay_min': 0.075, 'delay_max': 0.125, 'duration': 0.025},
    'SH': {'delay_min': 0.0803, 'delay_max': 0.1303, 'duration': 0.025},
    'A0H': {'delay_min': 0.115, 'delay_max': 0.225, 'duration': 0.025},
    'A0L': {'delay_min': 0.400, 'delay_max': 2.100, 'duration': 0.100},
}

WAVE_PARAMS_BY_DISTANCE = {
    30: WAVE_PARAMS_30M,
    40: WAVE_PARAMS_40M,
    50: WAVE_PARAMS_50M,
    60: WAVE_PARAMS_60M,
    70: WAVE_PARAMS_70M,
    80: WAVE_PARAMS_80M,
    90: WAVE_PARAMS_90M,
    100: WAVE_PARAMS_100M,
}

def get_distance_for_mic(mic_index: int) -> int:
    """根据MIC索引(1-8)返回该MIC的距离"""
    return MIC_DISTANCES.get(mic_index, 50)

def get_wave_params_for_mic(mic_index: int) -> dict:
    """
    获取指定MIC对应距离的完整波型参数
    合并freq_start, freq_end, emission_times(与距离无关)
    和delay_min, delay_max, duration(与距离相关)
    """
    distance = get_distance_for_mic(mic_index)
    distance_params = WAVE_PARAMS_BY_DISTANCE[distance]

    result = {}
    for wave_type, params in distance_params.items():
        result[wave_type] = {
            'freq_start': WAVE_PARAMS[wave_type]['freq_start'],
            'freq_end': WAVE_PARAMS[wave_type]['freq_end'],
            'emission_times': WAVE_PARAMS[wave_type]['emission_times'],
            'delay_min': params['delay_min'],
            'delay_max': params['delay_max'],
            'duration': params['duration'],
        }
    return result
