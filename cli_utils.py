"""
命令行参数解析工具模块
提供统一的工具函数和解析器，参考 LLaMAFactory 风格
"""
import argparse
from typing import Any, Dict


def set_nested_value(config: dict, path: str, value: Any) -> None:
    """
    使用点号路径设置嵌套字典的值（参考 LLaMAFactory 实现）
    
    Args:
        config: 配置字典
        path: 点号分隔的路径，如 'model.model_name_or_path'
        value: 要设置的值
    """
    keys = path.split('.')
    current = config
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def get_nested_value(config: dict, path: str, default: Any = None) -> Any:
    """
    使用点号路径获取嵌套字典的值
    
    Args:
        config: 配置字典
        path: 点号分隔的路径
        default: 默认值
    
    Returns:
        配置值或默认值
    """
    keys = path.split('.')
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def setup_argument_parser(description: str, args_config: Dict[str, Dict[str, Any]]) -> argparse.ArgumentParser:
    """
    设置命令行参数解析器
    
    Args:
        description: 程序描述
        args_config: 参数配置字典，格式为：
            {
                'arg_name': {
                    'type': type,
                    'default': default_value,
                    'help': 'help text'
                }
            }
    
    Returns:
        配置好的 ArgumentParser 实例
    """
    parser = argparse.ArgumentParser(description=description)
    for arg_name, arg_config in args_config.items():
        parser.add_argument(
            f'--{arg_name}',
            type=arg_config['type'],
            default=arg_config.get('default'),
            help=arg_config.get('help', '')
        )
    return parser

