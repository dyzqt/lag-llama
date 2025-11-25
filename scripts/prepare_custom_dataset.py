"""Lag-Llama 自定义零售数据集准备脚本，逐行附带中文注释解释实现思路。"""

from __future__ import annotations  # 允许类型注解引用稍后声明的类型

import argparse  # 解析命令行参数
import json  # 处理 JSON 序列化
import math  # 数学工具，用于检测 NaN
import re  # 正则匹配 list<...> 字段
import sys  # 修改模块搜索路径
from dataclasses import dataclass  # 快速定义数据类
from pathlib import Path  # 跨平台文件路径
from typing import Dict, Iterable, List, Optional, Sequence, Tuple  # 类型注解泛型

import numpy as np  # 数值计算库
import pandas as pd  # 表格处理库

REPO_ROOT = Path(__file__).resolve().parents[1]  # 计算仓库根目录
if str(REPO_ROOT) not in sys.path:  # 若根目录未加入 sys.path
    sys.path.insert(0, str(REPO_ROOT))  # 动态插入，便于导入 data.schema_spec

from data.schema_spec import FieldSpec, FieldValueType, SchemaSpec, load_schema  # 导入 schema 解析工具

CYCLICAL_FEATURES = {  # 定义需要生成正余弦编码的周期特征
    "dow_cyclical_sin_cos": ("dow", "week", 7),  # 星期周期固定 7 天
    "dom_cyclical_sin_cos": ("dom", "month", None),  # 月份周期长度按当月天数确定
    "doy_cyclical_sin_cos": ("doy", "year", None),  # 年度周期长度按闰年确定
    "month_cyclical_sin_cos": ("month", "year_month", 12),  # 12 个月固定周期
    "quarter_cyclical_sin_cos": ("quarter", "year_quarter", 4),  # 4 个季度固定周期
}

LIST_COLUMN_RE = re.compile(r"^list<(numeric|binary)>", flags=re.IGNORECASE)  # 识别列表类型字段


@dataclass
class ProcessedFeatures:
    """保存静态/动态特征划分结果以及类别映射。"""

    static_cat_columns: List[str]  # 静态类别字段
    static_real_columns: List[str]  # 静态数值字段
    dynamic_numeric_columns: List[str]  # 动态数值字段（一并包含展开后的列表列）
    dynamic_categorical_columns: List[str]  # 动态类别字段
    expanded_dynamic_columns: List[str]  # 记录展开生成的列
    mappings: Dict[str, Dict[str, int]]  # 类别取值到编码的映射
    list_column_types: Dict[str, str]  # 记录列表字段目标类型（numeric / categorical）


def _safe_float(value: object) -> Optional[float]:
    """尝试将任意值转换为 float，失败时返回 None。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _infer_static_dynamic_columns(schema: SchemaSpec, df: pd.DataFrame) -> ProcessedFeatures:
    """根据 schema 定义，识别静态/动态字段并准备类别编码。"""
    static_cat: List[str] = []  # 收集静态类别列
    static_real: List[str] = []  # 收集静态数值列
    dynamic_numeric: List[str] = []  # 收集动态数值列
    dynamic_cat: List[str] = []  # 收集动态类别列
    list_column_types: Dict[str, str] = {}
    mappings: Dict[str, Dict[str, int]] = {}  # 类别编码映射

    for field in schema.fields:  # 遍历所有字段定义
        if field.name not in df.columns:  # 数据缺少该列，直接跳过
            continue
        if field.is_target or field.is_date:  # 目标/日期由专门逻辑处理
            continue

        covariate_class = field.covariate_class

        if field.canonical_type == FieldValueType.LIST_NUMERIC:
            target_class = covariate_class or "dynamic_real"
            if target_class == "dynamic_cat":
                dynamic_cat.append(field.name)
                list_column_types[field.name] = "categorical"
            else:
                dynamic_numeric.append(field.name)
                list_column_types[field.name] = "numeric"
            continue

        if covariate_class == "static_cat":
            static_cat.append(field.name)
            continue
        if covariate_class == "static_real":
            static_real.append(field.name)
            continue
        if covariate_class == "dynamic_cat":
            dynamic_cat.append(field.name)
            continue
        if covariate_class == "dynamic_real":
            dynamic_numeric.append(field.name)
            continue

        if field.is_static:  # 静态字段根据类型分类
            if field.canonical_type == FieldValueType.CATEGORY:
                static_cat.append(field.name)
            elif field.canonical_type in {FieldValueType.NUMERIC, FieldValueType.BINARY}:
                static_real.append(field.name)
        else:
            if field.canonical_type == FieldValueType.CATEGORY:
                dynamic_cat.append(field.name)
            else:
                dynamic_numeric.append(field.name)

    def _is_numeric_id(value: object) -> bool:
        """检查值是否为纯数字ID（可以转换为整数）。"""
        if pd.isna(value):
            return False
        try:
            str_val = str(value).strip()
            # 尝试转换为整数，如果能转换且转换后字符串相同，则为数字ID
            int_val = int(float(str_val))  # 先转float再转int，处理"100.0"这种情况
            return str(int_val) == str_val
        except (ValueError, TypeError):
            return False

    for col in static_cat:  # 对静态类别字段进行编码
        # 检查是否为数字ID字段
        sample_values = df[col].dropna().unique()[:10]  # 取前10个样本值检查
        is_numeric_id_field = all(_is_numeric_id(val) for val in sample_values) if len(sample_values) > 0 else False
        
        if is_numeric_id_field:
            # 对于数字ID，直接使用原始值，不进行重新编码
            df[f"{col}__code"] = df[col].apply(
                lambda x: int(float(str(x))) if pd.notna(x) and _is_numeric_id(x) else 0
            ).astype("int32")
            # 创建映射：原始值 -> 原始值（保持原样）
            unique_values = sorted(df[col].dropna().unique(), key=lambda x: int(float(str(x))) if _is_numeric_id(x) else float('inf'))
            mappings[col] = {str(value): int(float(str(value))) for value in unique_values if _is_numeric_id(value)}
        else:
            # 对于文字类别，进行编码
            codes, uniques = pd.factorize(df[col], sort=True)  # 生成编码与唯一取值
            df[f"{col}__code"] = codes.astype("int32")  # 将编码存入新列
            mappings[col] = {str(value): int(idx) for idx, value in enumerate(uniques)}  # 保存映射

    for col in dynamic_cat:  # 动态类别字段处理
        # 检查是否为数字ID字段
        sample_values = df[col].dropna().unique()[:10]
        is_numeric_id_field = all(_is_numeric_id(val) for val in sample_values) if len(sample_values) > 0 else False
        
        if is_numeric_id_field:
            # 对于数字ID，直接使用原始值
            df[col] = df[col].apply(
                lambda x: int(float(str(x))) if pd.notna(x) and _is_numeric_id(x) else 0
            ).astype("int32")
            unique_values = sorted(df[col].dropna().unique(), key=lambda x: int(float(str(x))) if _is_numeric_id(x) else float('inf'))
            mappings[col] = {str(value): int(float(str(value))) for value in unique_values if _is_numeric_id(value)}
        else:
            # 对于文字类别，进行编码
            codes, uniques = pd.factorize(df[col], sort=True)
            df[col] = codes.astype("int32")
            mappings[col] = {str(value): int(idx) for idx, value in enumerate(uniques)}

    return ProcessedFeatures(
        static_cat_columns=static_cat,
        static_real_columns=static_real,
        dynamic_numeric_columns=dynamic_numeric,
        dynamic_categorical_columns=dynamic_cat,
        expanded_dynamic_columns=[],
        mappings=mappings,
        list_column_types=list_column_types,
    )


def _parse_list_value(value: object) -> List[float]:
    """将列表字段的各种表示转换为浮点数列表，自动忽略无法解析的值。"""
    if isinstance(value, (list, tuple, np.ndarray)):  # 原本就是序列
        result: List[float] = []
        for item in value:
            number = _safe_float(item)
            if number is not None:
                result.append(number)
        return result

    if value is None or (isinstance(value, float) and math.isnan(value)):  # 空值处理
        return []

    text = str(value).strip()  # 统一转为字符串
    if not text:  # 空字符串
        return []

    if text.startswith("[") and text.endswith("]"):  # JSON 格式列表
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = eval(text)  # noqa: S307  # 兜底解析 Python 表达式
        iterable = parsed if isinstance(parsed, (list, tuple)) else [parsed]
    else:
        iterable = text.split(",")  # 按逗号拆分

    result: List[float] = []
    for item in iterable:
        number = _safe_float(item if not isinstance(item, str) else item.strip())
        if number is not None:
            result.append(number)
    return result


def _expand_list_columns(df: pd.DataFrame, features: ProcessedFeatures) -> ProcessedFeatures:
    """
    处理 list<...> 字段。
    将所有列表类型字段展开成多列，每个时间点的每个元素作为一个独立特征。
    """
    new_dynamic_names: List[str] = []  # 保存新生成的列名
    
    for column, value_kind in list(features.list_column_types.items()):  # 遍历动态列表时复制一份避免原地修改
        if column not in df.columns:
            continue
        field_series = df[column]
        parsed_lists = field_series.apply(_parse_list_value)  # 先整体解析，避免重复计算
        sample = next((lst for lst in parsed_lists if lst), None)  # 找一个非空样本推断长度
        if sample is None:
            continue  # 整列都为空则跳过

        length = len(sample)
        
        # 所有列表类型都展开成多列
        expanded_cols = [f"{column}__{idx:02d}" for idx in range(length)]  # 生成列名
        for idx in range(length):  # 逐位置展开列表
            df[expanded_cols[idx]] = parsed_lists.apply(
                lambda lst, pos=idx: lst[pos] if len(lst) > pos else 0.0
            )
        new_dynamic_names.extend(expanded_cols)
        df.drop(columns=[column], inplace=True)
        
        # 根据value_kind决定添加到哪个特征列表
        if value_kind == "categorical":
            if column in features.dynamic_categorical_columns:
                features.dynamic_categorical_columns.remove(column)
            features.dynamic_categorical_columns.extend(expanded_cols)
        else:
            # numeric类型添加到dynamic_numeric_columns
            if column in features.dynamic_numeric_columns:
                features.dynamic_numeric_columns.remove(column)
            features.dynamic_numeric_columns.extend(expanded_cols)
        
        del features.list_column_types[column]

    # 将展开的列信息存储到ProcessedFeatures中
    features.expanded_dynamic_columns = new_dynamic_names
    
    return features


def _ensure_calendar_features(df: pd.DataFrame, date_col: str) -> None:
    """补充日期相关的基础字段，便于生成周期特征。"""
    df[date_col] = pd.to_datetime(df[date_col])  # 统一转换为 datetime
    df["dow"] = df[date_col].dt.weekday  # 星期几
    df["dom"] = df[date_col].dt.day  # 月份中的第几天
    df["doy"] = df[date_col].dt.dayofyear  # 年中的第几天
    df["month"] = df[date_col].dt.month  # 月份编号
    df["quarter"] = df[date_col].dt.quarter  # 季度编号
    df["days_in_month"] = df[date_col].dt.days_in_month  # 当月总天数
    df["days_in_year"] = np.where(df[date_col].dt.is_leap_year, 366, 365)  # 闰年 366 天


def _append_cyclical_features(df: pd.DataFrame) -> List[str]:
    """根据 CYCLICAL_FEATURES 定义创建 sin/cos 周期特征。"""
    created_columns: List[str] = []
    for feature_name, (base_col, category, period) in CYCLICAL_FEATURES.items():
        if base_col not in df.columns:
            continue

        if period is None:  # 需要动态确定周期长度
            if category == "month":
                denom = df["days_in_month"]
            elif category == "year":
                denom = df["days_in_year"]
            else:
                denom = df.get(f"{base_col}_period", 1)
        else:
            denom = period

        if category == "month":
            angle = 2 * np.pi * df["dom"] / df["days_in_month"].clip(lower=1)
        elif category == "year":
            angle = 2 * np.pi * df["doy"] / df["days_in_year"].clip(lower=1)
        elif category == "year_month":
            angle = 2 * np.pi * (df["month"] - 1) / 12
        elif category == "year_quarter":
            angle = 2 * np.pi * (df["quarter"] - 1) / 4
        elif category == "week":
            angle = 2 * np.pi * df["dow"] / 7
        else:
            angle = 2 * np.pi * df[base_col] / (denom if isinstance(denom, (int, float)) else denom)

        sin_col = f"{feature_name}_sin"
        cos_col = f"{feature_name}_cos"
        df[sin_col] = np.sin(angle)
        df[cos_col] = np.cos(angle)
        created_columns.extend([sin_col, cos_col])

    return created_columns


def _build_dynamic_feature_matrix(
    df: pd.DataFrame,
    dynamic_columns: Sequence[str],
    group_df: pd.DataFrame,
    *,
    value_kind: str = "float",
) -> List[List[float]]:
    """
    将动态列抽取成二维列表，符合 GluonTS 输入格式。
    """
    arrays: List[List[float]] = []
    is_float = value_kind == "float"
    fill_value = 0.0 if is_float else 0
    dtype = "float32" if is_float else "int32"

    for column in dynamic_columns:
        if column not in group_df.columns:
            arrays.append([fill_value] * len(group_df))
            continue

        # 所有列都作为标量值处理
        if is_float:
            converted: List[float] = []
            for val in group_df[column]:
                number = _safe_float(val)
                converted.append(number if number is not None else fill_value)
        else:
            converted = []
            for val in group_df[column]:
                if pd.isna(val):
                    converted.append(fill_value)
                else:
                    try:
                        converted.append(int(val))
                    except (TypeError, ValueError):
                        converted.append(fill_value)
        arrays.append(np.asarray(converted, dtype=dtype).tolist())
    return arrays


def _make_record(
    group_df: pd.DataFrame,
    static_info: ProcessedFeatures,
    dynamic_numeric_columns: Sequence[str],
    dynamic_categorical_columns: Sequence[str],
    target_col: str,
    prediction_length: int,
    mode: str,
) -> Dict:
    """把单个实体的时间序列转为 GluonTS 记录。"""
    assert mode in {"train", "test"}
    working_df = group_df.copy()
    if mode == "train":
        working_df = working_df.iloc[: len(working_df) - prediction_length]

    target = working_df[target_col].astype("float32").tolist()
    if not target:
        return {}

    item_id = "|".join(str(working_df.iloc[0][col]) for col in static_info.static_cat_columns)
    if not static_info.static_cat_columns:
        item_id = str(working_df.index[0])

    record: Dict = {
        "item_id": item_id,
        "start": working_df.iloc[0]["_dt"].isoformat(),
        "target": target,
    }

    if static_info.static_cat_columns:
        record["feat_static_cat"] = [
            int(working_df.iloc[0][f"{col}__code"]) for col in static_info.static_cat_columns
        ]

    if static_info.static_real_columns:
        record["feat_static_real"] = [
            float(working_df.iloc[0][col]) for col in static_info.static_real_columns
        ]

    if dynamic_numeric_columns:
        record["feat_dynamic_real"] = _build_dynamic_feature_matrix(
            df=working_df,
            dynamic_columns=dynamic_numeric_columns,
            group_df=working_df,
            value_kind="float",
        )
    if dynamic_categorical_columns:
        record["feat_dynamic_cat"] = _build_dynamic_feature_matrix(
            df=working_df,
            dynamic_columns=dynamic_categorical_columns,
            group_df=working_df,
            value_kind="int",
        )

    return record


def _prepare_records(
    df: pd.DataFrame,
    schema: SchemaSpec,
    group_keys: Sequence[str],
    target_col: str,
    prediction_length: int,
) -> Tuple[List[Dict], List[Dict], ProcessedFeatures]:
    """根据 schema 处理数据并生成训练/测试记录。"""
    processed = _infer_static_dynamic_columns(schema, df)
    processed = _expand_list_columns(df, processed)
    cyclical_columns = _append_cyclical_features(df)
    processed.dynamic_numeric_columns.extend(cyclical_columns)
    dynamic_numeric_columns = processed.dynamic_numeric_columns
    dynamic_categorical_columns = processed.dynamic_categorical_columns

    train_records: List[Dict] = []
    test_records: List[Dict] = []

    grouped = df.groupby(list(group_keys), sort=False, as_index=False)
    for _, group_df in grouped:
        sorted_group = group_df.sort_values("_dt").reset_index(drop=True)
        if len(sorted_group) <= prediction_length:
            continue

        train_record = _make_record(
            group_df=sorted_group,
            static_info=processed,
            dynamic_numeric_columns=dynamic_numeric_columns,
            dynamic_categorical_columns=dynamic_categorical_columns,
            target_col=target_col,
            prediction_length=prediction_length,
            mode="train",
        )
        if train_record:
            train_records.append(train_record)

        test_record = _make_record(
            group_df=sorted_group,
            static_info=processed,
            dynamic_numeric_columns=dynamic_numeric_columns,
            dynamic_categorical_columns=dynamic_categorical_columns,
            target_col=target_col,
            prediction_length=prediction_length,
            mode="test",
        )
        if test_record:
            test_records.append(test_record)

    return train_records, test_records, processed


def _write_jsonl_gz(path: Path, records: Iterable[Dict]) -> None:
    """以 gzip 格式写出 jsonl 文件。"""
    import gzip  # 延迟导入，非写文件场景无需加载

    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fp:
        for record in records:
            json.dump(record, fp, ensure_ascii=False)
            fp.write("\n")


def prepare_dataset_from_schema(
    schema_path: Path,
    data_path: Path,
    output_dir: Path,
    freq: str,
    prediction_length: int,
    group_keys: Sequence[str],
    date_col: str,
    target_col: str,
) -> None:
    """主流程：读 schema、合并数据，输出 Lag-Llama 可用的数据集。"""
    schema = load_schema(schema_path)
    if data_path.is_dir():  # 若 --data 指向目录，则遍历其中的 CSV
        frames: List[pd.DataFrame] = []
        for csv_path in sorted(data_path.glob("*.csv")):
            frames.append(pd.read_csv(csv_path))
        if not frames:
            raise ValueError(f"No CSV files found in directory {data_path!s}")
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(data_path)

    if date_col not in df.columns:
        raise ValueError(f"Date column {date_col!r} not found in data.")
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col!r} not found in data.")
    missing_keys = [col for col in group_keys if col not in df.columns]
    if missing_keys:
        raise ValueError(f"Grouping keys missing from data: {missing_keys}")

    df["_dt"] = pd.to_datetime(df[date_col])
    _ensure_calendar_features(df, "_dt")

    train_records, test_records, features = _prepare_records(
        df=df,
        schema=schema,
        group_keys=group_keys,
        target_col=target_col,
        prediction_length=prediction_length,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl_gz(output_dir / "train" / "data.json.gz", train_records)
    _write_jsonl_gz(output_dir / "test" / "data.json.gz", test_records)

    # 构建 metadata
    metadata = {
        "freq": freq,
        "prediction_length": prediction_length,
        "target": target_col,
        "date": date_col,
        "feat_static_cat": [
            {"name": col, "cardinality": len(features.mappings[col])}
            for col in features.static_cat_columns
        ],
        "feat_static_real": [{"name": col} for col in features.static_real_columns],
        "feat_dynamic_real": [{"name": col} for col in features.dynamic_numeric_columns],
        "dynamic_feature_source": "schema_v2",
    }
    if features.dynamic_categorical_columns:
        metadata["feat_dynamic_cat"] = [
            {"name": col, "cardinality": len(features.mappings.get(col, {}))}
            for col in features.dynamic_categorical_columns
        ]
    if features.mappings:
        metadata["categorical_mappings"] = features.mappings

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2, ensure_ascii=False)

    print(
        f"[prepare_custom_dataset] Wrote {len(train_records)} train records "
        f"and {len(test_records)} test records to {output_dir}"
    )


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, required=True, help="字段说明 CSV 的路径")
    parser.add_argument("--data", type=Path, required=True, help="原始宽表或目录路径")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="输出 GluonTS 数据集的目录",
    )
    parser.add_argument("--freq", type=str, default="D", help="时间序列频率，例如 D/H")
    parser.add_argument(
        "--prediction-length",
        type=int,
        default=28,
        dest="prediction_length",
        help="预测窗口长度，用于切分 train/test",
    )
    parser.add_argument(
        "--group-keys",
        nargs="+",
        required=True,
        help="唯一标识一条时间序列的列名，例如 city_id store_id product_id",
    )
    parser.add_argument(
        "--date-col",
        default="dt",
        help="原始数据中的日期列名称，默认 dt",
    )
    parser.add_argument(
        "--target-col",
        default="sale_amount",
        help="预测目标列名称，默认 sale_amount",
    )
    return parser.parse_args()


def main() -> None:
    """脚本入口：解析参数并调用主流程。"""
    args = parse_args()
    prepare_dataset_from_schema(
        schema_path=args.schema,
        data_path=args.data,
        output_dir=args.output,
        freq=args.freq,
        prediction_length=args.prediction_length,
        group_keys=args.group_keys,
        date_col=args.date_col,
        target_col=args.target_col,
    )


if __name__ == "__main__":
    main()
