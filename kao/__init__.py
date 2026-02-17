"""KAO: Parsimonious Symbolic Regression via Typed Quadratic Operators"""
try:
    from .KAO_v3_1 import *  # 按实际导出内容调整
except ImportError:
    pass  # deap / kneed 等重型依赖未安装时，仍可导入 kao.shared_classes
