# -*- coding: utf-8 -*-  # 保证中文注释在不同环境下正常显示

from __future__ import annotations  # 允许在类型注解中引用本模块稍后定义的类型

import enum  # 提供枚举类型，用于描述字段的标准类别
from dataclasses import dataclass  # dataclass 装饰器，快速定义只包含数据的类
from pathlib import Path  # Path 对象，方便跨平台处理文件路径
import re
from typing import List, Optional, Sequence  # 类型注解泛型，描述返回的列表/序列类型

import pandas as pd  # pandas 用于读取并处理字段说明的 CSV 文件


class FieldValueType(enum.Enum):
    """定义 Lag-Llama 数据管道可识别的标准字段类型。"""

    CATEGORY = "category"  # 类别型字段（用于离散嵌入）
    NUMERIC = "numeric"  # 连续数值型字段
    BINARY = "binary"  # 二值字段（0/1）
    LIST_NUMERIC = "list_numeric"  # 数值列表字段
    LIST_BINARY = "list_binary"  # 二值列表字段
    DATE = "date"  # 日期字段
    TARGET = "target"  # 预测目标字段


@dataclass(frozen=True)
class FieldSpec:
    """存储 schema 中单个字段的元信息。"""

    name: str  # 字段英文名
    raw_type: str  # 原始类型描述
    time_scope: str  # 时间粒度/范围说明
    module: str  # 所属业务模块
    required: bool  # 是否必填
    grain: str  # 数据粒度（门店级、商品级等）
    coop_type: str  # 静态/动态等合作类型描述
    description: str  # 字段的中文说明

    @property
    def is_time_dependent(self) -> bool:
        """根据 time_scope 判断字段是否随时间变化。"""
        scope = (self.time_scope or "").strip()  # 清理字符串
        if not scope or scope == "-":  # time_scope 为空或 '-' 视为静态字段
            return False
        return True  # 其它情况默认是动态字段

    @property
    def canonical_type(self) -> FieldValueType:
        """把字段原始类型映射到统一的枚举值。"""
        raw = (self.raw_type or "").strip().lower()  # 统一为小写方便匹配
        if raw in {"cat", "category", "categorical"}:  # 各种类别型写法
            return FieldValueType.CATEGORY
        if raw == "binary":  # 二值型
            return FieldValueType.BINARY
        if raw in {"numeric", "number"} or raw.startswith("numeric"):  # 数值型
            return FieldValueType.NUMERIC
        if raw.startswith("list<numeric"):  # 数值列表
            return FieldValueType.LIST_NUMERIC
        if raw.startswith("list<binary"):  # 二值列表
            return FieldValueType.LIST_BINARY
        if raw in {"date", "datetime"}:  # 日期/日期时间
            return FieldValueType.DATE
        if "target" in raw:  # 包含 target 字样即视为目标字段
            return FieldValueType.TARGET
        raise ValueError(f"Unrecognised field type {self.raw_type!r} for {self.name!r}")  # 未识别时抛出异常

    @property
    def is_static(self) -> bool:
        """是否静态字段。"""
        return not self.is_time_dependent

    @property
    def is_dynamic(self) -> bool:
        """是否动态字段。"""
        return self.is_time_dependent

    @property
    def is_target(self) -> bool:
        """是否目标字段。"""
        if self.canonical_type is FieldValueType.TARGET:  # 直接标记为 target 的情况
            return True
        coop = (self.coop_type or "").lower()  # coop_type 里可能包含 target 关键词
        return "target" in coop or self.name.lower() in {"target", "sale_amount"}  # 字段名匹配常用指标

    @property
    def is_date(self) -> bool:
        """是否日期字段。"""
        return self.canonical_type is FieldValueType.DATE

    @property
    def covariate_class(self) -> Optional[str]:
        """根据“协变量分类”列判断所属类别。"""
        label = (self.coop_type or "").strip()
        if not label:
            return None
        normalized = re.sub(r"\s+", "", label)
        mapping = {
            "静态类别协变量": "static_cat",
            "静态实数协变量": "static_real",
            "动态类别协变量": "dynamic_cat",
            "动态实数协变量": "dynamic_real",
        }
        return mapping.get(normalized)


@dataclass(frozen=True)
class SchemaSpec:
    """封装整张 schema 表，并提供便捷的字段筛选方法。"""

    fields: Sequence[FieldSpec]  # 所有字段的 FieldSpec 列表

    @property
    def static_fields(self) -> List[FieldSpec]:
        """返回所有静态字段（排除目标、日期）。"""
        return [f for f in self.fields if f.is_static and not f.is_target and not f.is_date]

    @property
    def dynamic_fields(self) -> List[FieldSpec]:
        """返回所有动态字段（排除目标、日期）。"""
        return [f for f in self.fields if f.is_dynamic and not f.is_target and not f.is_date]

    @property
    def target_field(self) -> FieldSpec:
        """获取预测目标字段，没有则抛出异常。"""
        for field in self.fields:  # 遍历所有字段
            if field.is_target:  # 找到第一个目标字段
                return field
        raise RuntimeError("Schema spec does not define a target field.")  # 未找到目标字段时提示

    @property
    def date_field(self) -> FieldSpec:
        """获取日期字段，没有则抛出异常。"""
        for field in self.fields:
            if field.is_date:
                return field
        raise RuntimeError("Schema spec does not define a calendar date field.")  # 未找到日期字段时提示

    @property
    def static_categorical_fields(self) -> List[FieldSpec]:
        """返回所有静态类别字段。"""
        return [
            f
            for f in self.static_fields
            if f.canonical_type is FieldValueType.CATEGORY
        ]

    @property
    def static_numeric_fields(self) -> List[FieldSpec]:
        """返回所有静态数值字段。"""
        return [
            f
            for f in self.static_fields
            if f.canonical_type in {FieldValueType.NUMERIC, FieldValueType.BINARY}
        ]

    @property
    def dynamic_categorical_fields(self) -> List[FieldSpec]:
        """返回所有动态类别字段。"""
        return [
            f
            for f in self.dynamic_fields
            if f.canonical_type is FieldValueType.CATEGORY
        ]

    @property
    def dynamic_numeric_fields(self) -> List[FieldSpec]:
        """返回所有动态数值字段（含列表数值）。"""
        return [
            f
            for f in self.dynamic_fields
            if f.canonical_type
            in {
                FieldValueType.NUMERIC,
                FieldValueType.BINARY,
                FieldValueType.LIST_NUMERIC,
            }
        ]


def _decode_required_flag(value: object) -> bool:
    """把“是否必填”列的文本值转换成布尔值。"""
    text = str(value or "").strip()  # 统一字符串表现
    return text in {"是", "必需", "必须", "1", "true", "True"}  # 中文/英文多种表达形式都算必填


def load_schema(path: Path) -> SchemaSpec:
    """从 CSV 文件加载 schema，并转换成 SchemaSpec 对象。"""

    df = pd.read_csv(path, encoding="gbk")  # 读取字段说明表
    if "Unnamed: 8" in df.columns:  # 某些导出版本最后多出空列，需要去掉
        df = df.drop(columns=["Unnamed: 8"])

    specs: List[FieldSpec] = []  # 用于收集 FieldSpec 实例
    for _, row in df.iterrows():  # 逐行遍历 CSV
        field = FieldSpec(  # 根据每一行创建 FieldSpec
            name=str(row.iloc[0]).strip(),  # 第 0 列：字段名
            raw_type=str(row.iloc[1]).strip(),  # 第 1 列：类型描述
            time_scope=str(row.iloc[2]).strip(),  # 第 2 列：时间粒度
            module=str(row.iloc[3]).strip(),  # 第 3 列：业务模块
            required=_decode_required_flag(row.iloc[4]),  # 第 4 列：必填标记
            grain=str(row.iloc[5]).strip(),  # 第 5 列：粒度
            coop_type=str(row.iloc[6]).strip(),  # 第 6 列：静态/动态说明
            description=str(row.iloc[7]).strip(),  # 第 7 列：中文描述
        )
        specs.append(field)  # 保存字段定义

    return SchemaSpec(specs)  # 使用收集到的字段生成 SchemaSpec


__all__ = ["FieldSpec", "FieldValueType", "SchemaSpec", "load_schema"]  # 控制模块对外暴露的符号
