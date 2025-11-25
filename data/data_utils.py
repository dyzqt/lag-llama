# Copyright 2024 Arjun Ashok
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import random
import warnings
import json
import os
import gzip
from pathlib import Path

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

import numpy as np
import pandas as pd
from tqdm import tqdm
from gluonts.dataset.common import ListDataset
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.transform import InstanceSampler
from pandas.tseries.frequencies import to_offset

from data.read_new_dataset import get_ett_dataset, create_train_dataset_without_last_k_timesteps, TrainDatasets, \
    MetaData


class CombinedDatasetIterator:
    # 负责在多个数据集之间按照权重随机抽样，从而实现多数据集混合训练
    def __init__(self, datasets, seed, weights):
        self._datasets = [iter(el) for el in datasets]
        self._weights = weights
        self._rng = random.Random(seed)

    def __next__(self):
        (dataset,) = self._rng.choices(self._datasets, weights=self._weights, k=1)
        return next(dataset)


class CombinedDataset:
    # 将多个 ListDataset 合并成一个迭代器，训练时可按权重轮询不同数据源
    def __init__(self, datasets, seed=None, weights=None):
        self._seed = seed
        self._datasets = datasets
        self._weights = weights
        n_datasets = len(datasets)
        if weights is None:
            self._weights = [1 / n_datasets] * n_datasets

    def __iter__(self):
        return CombinedDatasetIterator(self._datasets, self._seed, self._weights)

    def __len__(self):
        return sum([len(ds) for ds in self._datasets])


class SingleInstanceSampler(InstanceSampler):
    # 在每条时间序列中随机选择一个合法窗口，避免长序列被采样得过多
    """
    Randomly pick a single valid window in the given time series.
    This fix the bias in ExpectedNumInstanceSampler which leads to varying sampling frequency
    of time series of unequal length, not only based on their length, but when they were sampled.
    """

    def __init__(self, min_past: int = 0, min_future: int = 0, **kwargs):
        super().__init__(min_past=min_past, min_future=min_future, **kwargs)

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)
        window_size = b - a + 1
        if window_size <= 0:
            return np.array([], dtype=int)
        indices = np.random.randint(window_size, size=1)
        return indices + a


def _count_timesteps(
        left: pd.Timestamp, right: pd.Timestamp, delta: pd.DateOffset
) -> int:
    """
    Count how many timesteps there are between left and right, according to the given timesteps delta.
    If the number if not integer, round down.
    """
    # This is due to GluonTS replacing Timestamp by Period for version 0.10.0.
    # Original code was tested on version 0.9.4
    if type(left) == pd.Period:
        left = left.to_timestamp()
    if type(right) == pd.Period:
        right = right.to_timestamp()
    assert (
            right >= left
    ), f"Case where left ({left}) is after right ({right}) is not implemented in _count_timesteps()."
    try:
        return (right - left) // delta
    except TypeError:
        # For MonthEnd offsets, the division does not work, so we count months one by one.
        for i in range(10000):
            if left + (i + 1) * delta > right:
                return i
        else:
            raise RuntimeError(
                f"Too large difference between both timestamps ({left} and {right}) for _count_timesteps()."
            )


from pathlib import Path
from gluonts.dataset.common import ListDataset
from gluonts.dataset.repository.datasets import get_dataset


def create_train_dataset_last_k_percentage(
        raw_train_dataset,
        freq,
        k=100
):
    # Get training data
    train_data = []
    for i, series in enumerate(raw_train_dataset):
        s_train = series.copy()
        number_of_values = int(len(s_train["target"]) * k / 100)
        train_start_index = len(s_train["target"]) - number_of_values
        s_train["target"] = s_train["target"][train_start_index:]
        train_data.append(s_train)

    train_data = ListDataset(train_data, freq=freq)

    return train_data


def _slice_dynamic_features(entry, start_idx=None, end_idx=None):
    slice_obj = slice(start_idx, end_idx)
    for key in ("feat_dynamic_real", "feat_dynamic_cat"):
        values = entry.get(key)
        if values is None or len(values) == 0:
            continue
        if isinstance(values, np.ndarray):
            values = values.tolist()
        if len(values) == 0:
            continue
        entry[key] = [np.asarray(feature)[slice_obj].tolist() for feature in values]


def _read_metadata_dict(dataset_path, name):
    dataset_dir = Path(dataset_path) / name
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def get_covariate_feature_sizes(dataset_path, name):
    """
    读取 metadata.json，统计不同类型协变量的数量。
    返回 dict，键包括 dynamic_real/static_real/dynamic_cat/static_cat。
    """
    metadata_dict = _read_metadata_dict(dataset_path, name)
    if metadata_dict is None:
        return {
            "dynamic_real": 0,
            "dynamic_cat": 0,
            "static_real": 0,
            "static_cat": 0,
        }

    return {
        "dynamic_real": len(metadata_dict.get("feat_dynamic_real", [])),
        "dynamic_cat": len(metadata_dict.get("feat_dynamic_cat", [])),
        "static_real": len(metadata_dict.get("feat_static_real", [])),
        "static_cat": len(metadata_dict.get("feat_static_cat", [])),
    }


def get_dynamic_feat_size(dataset_path, name):
    """
    保留历史接口，仅返回动态实值协变量的数量。
    """
    return get_covariate_feature_sizes(dataset_path, name)["dynamic_real"]


def _load_custom_json_dataset(dataset_path, name):
    dataset_dir = Path(dataset_path) / name
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        return None

    with metadata_path.open("r", encoding="utf-8") as fp:
        metadata_dict = json.load(fp)

    def _prepare_metadata(metadata_dict):
        meta_kwargs = {}
        meta_kwargs["freq"] = metadata_dict["freq"]
        if "prediction_length" in metadata_dict:
            meta_kwargs["prediction_length"] = metadata_dict["prediction_length"]

        target = metadata_dict.get("target")
        if isinstance(target, str):
            meta_kwargs["target"] = {"name": target}
        elif isinstance(target, dict):
            meta_kwargs["target"] = target
        elif target is not None:
            warnings.warn("Unsupported target specification in metadata, ignoring.")

        def _convert_basic_list(key):
            features = metadata_dict.get(key, [])
            result = []
            for feature in features:
                if not isinstance(feature, dict):
                    continue
                name = feature.get("name")
                if not name:
                    continue
                result.append({"name": name})
            return result

        def _convert_cat_list(key):
            features = metadata_dict.get(key, [])
            result = []
            for feature in features:
                if not isinstance(feature, dict):
                    continue
                name = feature.get("name")
                if not name:
                    continue
                cardinality = feature.get("cardinality")
                if cardinality is None:
                    warnings.warn(f"No cardinality specified for categorical feature '{name}'.")
                    continue
                result.append({"name": name, "cardinality": str(cardinality)})
            return result

        meta_kwargs["feat_static_real"] = _convert_basic_list("feat_static_real")
        meta_kwargs["feat_dynamic_real"] = _convert_basic_list("feat_dynamic_real")
        meta_kwargs["feat_static_cat"] = _convert_cat_list("feat_static_cat")
        meta_kwargs["feat_dynamic_cat"] = _convert_cat_list("feat_dynamic_cat")
        return MetaData(**meta_kwargs)

    metadata = _prepare_metadata(metadata_dict)

    def _load_split(split):
        for filename in ("data.json.gz", "data.json"):
            candidate = dataset_dir / split / filename
            if candidate.exists():
                if candidate.suffix == ".gz":
                    with gzip.open(candidate, "rt", encoding="utf-8") as fp:
                        return [json.loads(line) for line in fp if line.strip()]
                else:
                    with candidate.open("r", encoding="utf-8") as fp:
                        return [json.loads(line) for line in fp if line.strip()]
        raise FileNotFoundError(
            f"Could not locate data.json(.gz) for split '{split}' in {dataset_dir}"
        )

    try:
        train_data = _load_split("train")
        test_data = _load_split("test")
    except FileNotFoundError as err:
        warnings.warn(str(err))
        return None

    train_ds = ListDataset(train_data, freq=metadata.freq)
    test_ds = ListDataset(test_data, freq=metadata.freq)
    return TrainDatasets(metadata=metadata, train=train_ds, test=test_ds)


def create_train_and_val_datasets_with_dates(
        name,
        dataset_path,
        data_id,
        history_length,
        prediction_length=None,
        num_val_windows=None,
        val_start_date=None,
        train_start_date=None,
        freq=None,
        last_k_percentage=None
):
    """
    核心入口：根据历史窗口/验证窗口划分训练与验证数据集，并统计样本点数量。
    兼容官方公共数据与自定义 JSON 数据，必要时裁剪动态协变量。
    Train Start date is assumed to be the start of the series if not provided
    Freq is not given is inferred from the data
    We can use ListDataset to just group multiple time series - https://github.com/awslabs/gluonts/issues/695
    """

    if name in ("ett_h1", "ett_h2", "ett_m1", "ett_m2"):
        path = os.path.join(dataset_path, "ett_datasets")
        raw_dataset = get_ett_dataset(name, path)
    elif name in ("cpu_limit_minute", "cpu_usage_minute", \
                  "function_delay_minute", "instances_minute", \
                  "memory_limit_minute", "memory_usage_minute", \
                  "platform_delay_minute", "requests_minute"):
        path = os.path.join(dataset_path, "huawei/" + name + ".json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Huawei dataset '{name}' not found at expected path: {path}\n"
                f"Expected structure: {dataset_path}/huawei/{name}.json\n"
                f"Please ensure the dataset exists at the specified path or check that --dataset_path is correct."
            )
        with open(path, "r") as f:
            data = json.load(f)
        metadata = MetaData(**data["metadata"])
        train_data = [x for x in data["train"] if type(x["target"][0]) != str]
        test_data = [x for x in data["test"] if type(x["target"][0]) != str]
        train_ds = ListDataset(train_data, freq=metadata.freq)
        test_ds = ListDataset(test_data, freq=metadata.freq)
        raw_dataset = TrainDatasets(metadata=metadata, train=train_ds, test=test_ds)
    elif name in ("beijing_pm25", "AirQualityUCI", "beijing_multisite"):
        path = os.path.join(dataset_path, "air_quality/" + name + ".json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Air quality dataset '{name}' not found at expected path: {path}\n"
                f"Expected structure: {dataset_path}/air_quality/{name}.json\n"
                f"Please ensure the dataset exists at the specified path or check that --dataset_path is correct."
            )
        with open(path, "r") as f:
            data = json.load(f)
        metadata = MetaData(**data["metadata"])
        train_test_data = [x for x in data["data"] if type(x["target"][0]) != str]
        full_dataset = ListDataset(train_test_data, freq=metadata.freq)
        train_ds = create_train_dataset_without_last_k_timesteps(full_dataset, freq=metadata.freq, k=24)
        raw_dataset = TrainDatasets(metadata=metadata, train=train_ds, test=full_dataset)
    else:
        raw_dataset = _load_custom_json_dataset(dataset_path, name)
        if raw_dataset is None:
            raw_dataset = get_dataset(name, path=Path(dataset_path))

    if prediction_length is None:
        prediction_length = raw_dataset.metadata.prediction_length
    if freq is None:
        freq = raw_dataset.metadata.freq
    timestep_delta = pd.tseries.frequencies.to_offset(freq)
    raw_train_dataset = raw_dataset.train

    if not num_val_windows and not val_start_date:
        raise Exception("Either num_val_windows or val_start_date must be provided")
    if num_val_windows and val_start_date:
        raise Exception("Either num_val_windows or val_start_date must be provided")

    max_train_end_date = None

    # Get training data
    total_train_points = 0
    train_data = []
    min_required_length = history_length + prediction_length
    filtered_count = 0
    for i, series in enumerate(raw_train_dataset):
        s_train = series.copy()
        if val_start_date is not None:
            train_end_index = _count_timesteps(
                series["start"] if not train_start_date else train_start_date,
                val_start_date,
                timestep_delta,
            )
        else:
            train_end_index = len(series["target"]) - num_val_windows
        train_end_index = max(0, train_end_index)
        # Compute train_start_index based on last_k_percentage
        if last_k_percentage:
            number_of_values = int(len(s_train["target"]) * last_k_percentage / 100)
            train_start_index = max(train_end_index - number_of_values, 0)
        else:
            train_start_index = 0
        s_train["start"] = series["start"] + train_start_index * timestep_delta
        s_train["target"] = series["target"][train_start_index:train_end_index]

        # Filter out sequences that are too short
        if len(s_train["target"]) < min_required_length:
            filtered_count += 1
            continue

        _slice_dynamic_features(s_train, train_start_index, train_end_index)
        s_train["item_id"] = i
        s_train["data_id"] = data_id
        train_data.append(s_train)
        total_train_points += len(s_train["target"])

        # Calculate the end date
        end_date = s_train["start"] + timestep_delta * (len(s_train["target"]) - 1)
        if max_train_end_date is None or end_date > max_train_end_date:
            max_train_end_date = end_date

    if filtered_count > 0:
        warnings.warn(
            f"Filtered out {filtered_count} training sequences that were shorter than "
            f"min_required_length={min_required_length}. Remaining sequences: {len(train_data)}"
        )
    if len(train_data) == 0:
        raise ValueError(
            f"No training sequences remain after filtering. All sequences were shorter than "
            f"min_required_length={min_required_length}. Consider reducing context_length, "
            f"prediction_length, or num_validation_windows."
        )
    train_data = ListDataset(train_data, freq=freq)

    # Get validation data
    total_val_points = 0
    total_val_windows = 0
    val_data = []
    val_filtered_count = 0
    for i, series in enumerate(raw_train_dataset):
        s_val = series.copy()
        if val_start_date is not None:
            train_end_index = _count_timesteps(
                series["start"], val_start_date, timestep_delta
            )
        else:
            train_end_index = len(series["target"]) - num_val_windows
        val_start_index = train_end_index - prediction_length - history_length
        s_val["start"] = series["start"] + val_start_index * timestep_delta
        s_val["target"] = series["target"][val_start_index:]

        # Filter out sequences that are too short
        if len(s_val["target"]) < min_required_length:
            val_filtered_count += 1
            continue

        _slice_dynamic_features(s_val, val_start_index, None)
        s_val["item_id"] = i
        s_val["data_id"] = data_id
        val_data.append(s_val)
        total_val_points += len(s_val["target"])
        total_val_windows += len(s_val["target"]) - prediction_length - history_length
    if val_filtered_count > 0:
        warnings.warn(
            f"Filtered out {val_filtered_count} validation sequences that were shorter than "
            f"min_required_length={min_required_length}. Remaining sequences: {len(val_data)}"
        )
    val_data = ListDataset(val_data, freq=freq)

    total_points = (
            total_train_points
            + total_val_points
            - (len(raw_train_dataset) * (prediction_length + history_length))
    )

    return (
        train_data,
        val_data,
        total_train_points,
        total_val_points,
        total_val_windows,
        max_train_end_date,
        total_points,
    )


def create_test_dataset(
        name, dataset_path, history_length, freq=None, data_id=None
):
    """
    构建测试阶段所需的 ListDataset，只保留最近的 history_length+prediction_length 窗口。
    对自定义数据集同样裁剪协变量，方便推理时使用。
    For now, only window per series is used.
    make_evaluation_predictions automatically only predicts for the last "prediction_length" timesteps
    NOTE / TODO: For datasets where the test set has more series (possibly due to more timestamps), \
    we should check if we only use the last N series where N = series per single timestamp, or if we should do something else.
    """

    if name in ("ett_h1", "ett_h2", "ett_m1", "ett_m2"):
        path = os.path.join(dataset_path, "ett_datasets")
        dataset = get_ett_dataset(name, path)
    elif name in ("cpu_limit_minute", "cpu_usage_minute", \
                  "function_delay_minute", "instances_minute", \
                  "memory_limit_minute", "memory_usage_minute", \
                  "platform_delay_minute", "requests_minute"):
        path = os.path.join(dataset_path, "huawei/" + name + ".json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Huawei dataset '{name}' not found at expected path: {path}\n"
                f"Expected structure: {dataset_path}/huawei/{name}.json\n"
                f"Please ensure the dataset exists at the specified path or check that --dataset_path is correct."
            )
        with open(path, "r") as f:
            data = json.load(f)
        metadata = MetaData(**data["metadata"])
        train_data = [x for x in data["train"] if type(x["target"][0]) != str]
        test_data = [x for x in data["test"] if type(x["target"][0]) != str]
        train_ds = ListDataset(train_data, freq=metadata.freq)
        test_ds = ListDataset(test_data, freq=metadata.freq)
        dataset = TrainDatasets(metadata=metadata, train=train_ds, test=test_ds)
    elif name in ("beijing_pm25", "AirQualityUCI", "beijing_multisite"):
        path = os.path.join(dataset_path, "air_quality/" + name + ".json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Air quality dataset '{name}' not found at expected path: {path}\n"
                f"Expected structure: {dataset_path}/air_quality/{name}.json\n"
                f"Please ensure the dataset exists at the specified path or check that --dataset_path is correct."
            )
        with open(path, "r") as f:
            data = json.load(f)
        metadata = MetaData(**data["metadata"])
        train_test_data = [x for x in data["data"] if type(x["target"][0]) != str]
        full_dataset = ListDataset(train_test_data, freq=metadata.freq)
        train_ds = create_train_dataset_without_last_k_timesteps(full_dataset, freq=metadata.freq, k=24)
        dataset = TrainDatasets(metadata=metadata, train=train_ds, test=full_dataset)
    else:
        dataset = _load_custom_json_dataset(dataset_path, name)
        if dataset is None:
            dataset = get_dataset(name, path=Path(dataset_path))

    if freq is None:
        freq = dataset.metadata.freq
    prediction_length = dataset.metadata.prediction_length
    timestep_delta = to_offset(freq)
    data = []
    total_points = 0
    window_size = history_length + prediction_length
    for i, series in enumerate(dataset.test):
        series_copy = copy.deepcopy(series)
        offset = len(series["target"]) - window_size
        if offset > 0:
            series_copy["target"] = series["target"][-window_size:]
            series_copy["start"] = series["start"] + offset * timestep_delta
            _slice_dynamic_features(series_copy, len(series["target"]) - window_size, len(series["target"]))
        series_copy["item_id"] = i
        series_copy["data_id"] = data_id
        data.append(series_copy)
        total_points += len(series_copy["target"])
    return ListDataset(data, freq=freq), prediction_length, total_points
