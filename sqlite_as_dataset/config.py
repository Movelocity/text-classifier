import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据库配置
DATABASE_URL = "sqlite:///./annotation.db"

# 服务器配置
HOST = "0.0.0.0"
PORT = 8000

# 日志配置
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 分页配置
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 1000 