"""自定义异常类"""


class EvaluatorError(Exception):
    """所有画质评测异常的基类。"""
    pass


class ConfigError(EvaluatorError):
    """配置文件错误（字段缺失、类型不对、值越界）。"""
    pass


class ImageLoadError(EvaluatorError):
    """图像加载失败。"""
    pass


class ROIDetectionError(EvaluatorError):
    """ROI 检测失败（找不到色卡、测试卡等）。"""
    pass


class MetricError(EvaluatorError):
    """指标计算异常。"""
    pass
