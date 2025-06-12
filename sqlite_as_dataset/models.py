"""
文本标注系统的数据库模型。

本模块定义了以下 SQLAlchemy 模型：
- AnnotationData: 存储带有关联标签的文本（无重复）
- Label: 存储带有 id 和标签字符串的标签信息
"""

from sqlalchemy import Column, Integer, String, Text, create_engine, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

Base = declarative_base()


class AnnotationData(Base):
    """
    存储文本标注数据的模型。
    
    Attributes:
        id: 主键（自动生成）
        text: 文本内容（不允许重复，已建立索引）
        labels: 与文本关联的标签的逗号分隔字符串（已建立索引）
    """
    __tablename__ = "annotation_data"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(Text, nullable=False, unique=True, index=True)  # 添加索引提高查询性能
    labels = Column(String, nullable=True, index=True)  # 添加索引提高标签搜索性能
    
    # 显式定义复合索引，优化多条件查询
    __table_args__ = (
        Index('ix_text_labels', 'text', 'labels'),  # 复合索引用于同时搜索文本和标签
        Index('ix_labels_partial', 'labels'),  # 标签部分匹配索引
    )


class Label(Base):
    """
    标签管理的模型。

    Attributes:
        id: 标签的唯一标识符
        label: 标签字符串（已建立唯一索引）
        description: 标签描述（可选）
    """
    __tablename__ = "labels"

    id = Column(Integer, primary_key=True)
    label = Column(String, nullable=False, unique=True, index=True)  # 添加索引
    description = Column(Text, nullable=True)  # 标签描述
    groups = Column(Text, nullable=True)  # 标签分组 aaa/bbb/ccc

from .config import DATABASE_URL

# 优化后的数据库引擎配置
engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False,
        "timeout": 20,  # 增加超时时间
        "isolation_level": None,  # 启用自动提交模式以提高性能
    },
    poolclass=StaticPool,  # 使用静态连接池
    pool_pre_ping=True,  # 连接前检查连接有效性
    echo=False,  # 生产环境关闭SQL日志
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables():
    """创建所有数据库表和索引。"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """获取 FastAPI 的数据库会话依赖。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 