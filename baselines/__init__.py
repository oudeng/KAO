"""Baseline symbolic regression methods — unified interface"""
from abc import ABC, abstractmethod
import numpy as np
import re


class BaselineSR(ABC):
    """所有基线方法的统一接口"""

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_test: np.ndarray = None, y_test: np.ndarray = None,
            time_budget: float = 60.0, random_state: int = 42,
            **kwargs) -> dict:
        """
        运行符号回归，返回结果字典：
        {
            'expression': str,         # 符号表达式字符串
            'y_pred_train': ndarray,   # 训练集预测
            'y_pred_test': ndarray,    # 测试集预测（如果提供了 X_test）
            'complexity': int,         # 方法原生复杂度（节点数或项数）
            'complexity_chars': int,   # 统一字符长度代理指标
            'runtime': float,          # 实际运行秒数
            'model': object,           # 原始模型对象
        }
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """方法名称"""
        pass

    @staticmethod
    def compute_complexity_chars(expr_str: str) -> int:
        """统一的字符长度复杂度代理"""
        cleaned = re.sub(r'[\s\(\)]', '', str(expr_str))
        return len(cleaned)
