"""
文本标注 API 的 Pydantic schemas。

本模块定义了以下请求和响应模型：
- 标注数据操作
- 标签管理
- 数据导入操作
"""

from typing import List, Optional, Union
from pydantic import BaseModel, Field, validator, model_validator


class AnnotationDataBase(BaseModel):
    """标注数据的基础 schema。"""
    text: str = Field(..., description="文本内容")
    labels: Optional[str] = Field(None, description="多标签，存储为逗号分隔字符串")
    
    @validator('labels')
    def validate_labels(cls, v):
        """验证标签格式，确保逗号分隔的标签格式正确"""
        if v is None or v == '':
            return None
        # 清理标签：去除空白，过滤空标签，去重
        labels = [label.strip() for label in v.split(',')]
        labels = [label for label in labels if label]  # 移除空标签
        labels = list(dict.fromkeys(labels))  # 去重并保持顺序
        return ', '.join(labels) if labels else None


class AnnotationDataCreate(AnnotationDataBase):
    """创建新标注数据的 schema。"""
    pass


class AnnotationDataUpdate(BaseModel):
    """更新标注数据的 schema。"""
    labels: Optional[str] = Field(None, description="逗号分隔的标签")


class AnnotationDataResponse(AnnotationDataBase):
    """标注数据响应的 schema。"""
    id: int = Field(..., description="唯一标识符")
    
    class Config:
        from_attributes = True


class AnnotationDataList(BaseModel):
    """分页标注数据列表的 schema。"""
    items: List[AnnotationDataResponse]
    total: int = Field(..., description="记录总数")
    page: int = Field(..., description="当前页码")
    per_page: int = Field(..., description="每页记录数")


class LabelBase(BaseModel):
    """标签的基础 schema。"""
    label: str = Field(..., description="标签字符串")
    description: Optional[str] = Field(None, description="标签描述")
    groups: Optional[str] = Field(None, description="标签分组 aaa/bbb/ccc")


class LabelCreate(LabelBase):
    """创建新标签的 schema。"""
    id: Optional[int] = Field(None, description="可选的自定义 ID")


class LabelUpdate(LabelBase):
    """更新标签的 schema。"""
    pass


class LabelResponse(LabelBase):
    """标签响应的 schema。"""
    id: int = Field(..., description="唯一标识符")
    
    class Config:
        from_attributes = True


class ImportStats(BaseModel):
    """导入操作统计的 schema。"""
    files_processed: int = Field(..., description="处理的文件数")
    records_imported: int = Field(..., description="导入的记录数")
    duplicates_merged: int = Field(..., description="合并的重复项数")


class ImportRequest(BaseModel):
    """导入请求的 schema。"""
    file_path: str = Field(..., description="要导入的文件路径")


class TextImportRequest(BaseModel):
    """文本导入请求的 schema。"""
    texts: List[str] = Field(..., description="要导入的文本列表")


class BulkLabelRequest(BaseModel):
    """批量标注请求的 schema。"""
    text_ids: List[int] = Field(..., description="要标注的文本 ID 列表")
    labels: str = Field(..., description="要应用的多标签，逗号分隔格式")
    
    @validator('labels')
    def validate_labels(cls, v):
        """验证标签格式"""
        if not v or not v.strip():
            raise ValueError("标签不能为空")
        # 清理标签
        labels = [label.strip() for label in v.split(',')]
        labels = [label for label in labels if label]
        if not labels:
            raise ValueError("至少需要一个有效标签")
        labels = list(dict.fromkeys(labels))  # 去重
        return ', '.join(labels)


class SearchRequest(BaseModel):
    """文本搜索请求的 schema。"""
    query: Optional[str] = Field(None, description="文本必须包含的关键词（模糊搜索、正则搜索，暂不开发）")
    exclude_query: Optional[str] = Field(None, description="文本不能包含的关键词（模糊搜索、正则搜索，暂不开发）")
    keywords: Optional[List[str]] = Field(None, description="文本必须包含的关键词数组（精确包含搜索）")
    exclude_keywords: Optional[List[str]] = Field(None, description="文本不能包含的关键词数组（精确包含搜索）")
    labels: Optional[str] = Field(None, description="文本必须包含的逗号分隔标签")
    exclude_labels: Optional[str] = Field(None, description="文本不能包含的逗号分隔标签")
    unlabeled_only: bool = Field(False, description="仅返回未标注文本")
    page: int = Field(1, description="页码", ge=1)
    per_page: int = Field(50, description="每页记录数", ge=1, le=1000)


class LabelStats(BaseModel):
    """标签统计的 schema。"""
    label: str = Field(..., description="标签名称")
    count: int = Field(..., description="带有此标签的文本数")


class SystemStats(BaseModel):
    """系统统计的 schema。"""
    total_texts: int = Field(..., description="总文本数")
    labeled_texts: int = Field(..., description="已标注文本数")
    unlabeled_texts: int = Field(..., description="未标注文本数")
    total_labels: int = Field(..., description="总唯一标签数")
    label_statistics: List[LabelStats] = Field(..., description="每个标签的统计")


class BulkLabelUpdateRequest(BaseModel):
    """批量标签更新请求的 schema。"""
    search_criteria: Optional[SearchRequest] = Field(None, description="搜索条件（与text_ids二选一）")
    text_ids: Optional[List[int]] = Field(None, description="要更新的文本ID列表（与search_criteria二选一）")
    labels_to_add: Optional[str] = Field(None, description="要添加的标签，逗号分隔")
    labels_to_remove: Optional[str] = Field(None, description="要删除的标签，逗号分隔")
    
    @validator('labels_to_add')
    def validate_labels_to_add(cls, v):
        """验证要添加的标签格式"""
        if v is None or v == '':
            return None
        # 清理标签
        labels = [label.strip() for label in v.split(',')]
        labels = [label for label in labels if label]
        if not labels:
            return None
        labels = list(dict.fromkeys(labels))  # 去重
        return ', '.join(labels)
    
    @validator('labels_to_remove')
    def validate_labels_to_remove(cls, v):
        """验证要删除的标签格式"""
        if v is None or v == '':
            return None
        # 清理标签
        labels = [label.strip() for label in v.split(',')]
        labels = [label for label in labels if label]
        if not labels:
            return None
        labels = list(dict.fromkeys(labels))  # 去重
        return ', '.join(labels)
    
    @model_validator(mode='after')
    def validate_operation(self):
        """验证操作参数的完整性"""
        # 必须提供搜索条件或ID列表之一
        if not self.search_criteria and not self.text_ids:
            raise ValueError('必须提供search_criteria或text_ids之一')
        
        # text_ids如果提供则不能为空
        if self.text_ids is not None and len(self.text_ids) == 0:
            raise ValueError('text_ids不能为空列表')
        
        # 必须提供添加或删除操作之一
        if not self.labels_to_add and not self.labels_to_remove:
            raise ValueError('必须提供labels_to_add或labels_to_remove之一')
        
        return self


class BulkLabelUpdateResponse(BaseModel):
    """批量标签更新响应的 schema。"""
    updated_count: int = Field(..., description="更新的记录数量")
    message: str = Field(..., description="操作结果描述")


# 数据生成相关schemas
class GenerateRequest(BaseModel):
    """数据生成请求的 schema。"""
    api_key: str = Field(..., description="大模型API Key")
    base_url: str = Field(..., description="大模型API Base URL")
    model: str = Field(default="gpt-3.5-turbo", description="模型名称")
    system_prompt: str = Field(..., description="系统提示词")
    user_prompt: str = Field(..., description="用户提示词")
    count: int = Field(default=10, ge=1, le=100, description="生成数量")
    parse_regex: Optional[str] = Field(None, description="解析正则表达式")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="生成温度")
    max_tokens: Optional[int] = Field(None, ge=1, le=4096, description="最大token数")


class GeneratedText(BaseModel):
    """生成的文本 schema。"""
    text: str = Field(..., description="生成的文本内容")
    labels: Optional[str] = Field(None, description="解析出的标签")
    raw_output: str = Field(..., description="原始模型输出")


class GenerateResponse(BaseModel):
    """数据生成响应的 schema。"""
    success: bool = Field(..., description="是否成功")
    generated_count: int = Field(..., description="成功生成的数量")
    texts: List[GeneratedText] = Field(..., description="生成的文本列表")
    error: Optional[str] = Field(None, description="错误信息")


class GenerateStatus(BaseModel):
    """生成状态 schema。"""
    status: str = Field(..., description="状态: generating, completed, cancelled, error")
    progress: int = Field(..., description="进度 (0-100)")
    current_count: int = Field(..., description="当前已生成数量")
    total_count: int = Field(..., description="目标总数量")
    message: Optional[str] = Field(None, description="状态消息")
    error: Optional[str] = Field(None, description="错误信息") 