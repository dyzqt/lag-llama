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

from typing import Any, Dict, Iterable, Optional

import warnings
import pytorch_lightning as pl
import torch

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import as_stacked_batches
from gluonts.dataset.stat import calculate_dataset_statistics
from gluonts.itertools import Cyclic
from gluonts.time_feature import (
    get_lags_for_frequency,
    time_features_from_frequency_str,
)
from gluonts.torch.distributions import StudentTOutput, NegativeBinomialOutput
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import (
    AddObservedValuesIndicator,
    AddTimeFeatures,
    Chain,
    DummyValueImputation,
    ExpectedNumInstanceSampler,
    InstanceSampler,
    InstanceSplitter,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
)

from gluon_utils.gluon_ts_distributions.implicit_quantile_network import (
    ImplicitQuantileNetworkOutput,
)
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from lag_llama.gluon.lightning_module import LagLlamaLightningModule

PREDICTION_INPUT_NAMES = [
    "past_target",
    "past_observed_values",
]
TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
    "future_target",
    "future_observed_values",
]


class LagLlamaEstimator(PyTorchLightningEstimator):
    """
    An estimator training a ConvTSMixer model for forecasting.

    This class is uses the model defined in ``ConvTSMixerModel``,
    and wraps it into a ``ConvTSMixerLightningModule`` for training
    purposes: training is performed using PyTorch Lightning's ``pl.Trainer``
    class.

    Parameters
    ----------
    prediction_length
        Length of the prediction horizon.
    context_length
        Number of time steps prior to prediction time that the model
        takes as inputs (default: ``10 * prediction_length``).
    lr
        Learning rate (default: ``1e-3``).
    weight_decay
        Weight decay regularization parameter (default: ``1e-8``).
    distr_output
        Distribution to use to evaluate observations and sample predictions
        (default: StudentTOutput()).
    loss
        Loss to be optimized during training
        (default: ``NegativeLogLikelihood()``).
    batch_norm
        Whether to apply batch normalization.
    batch_size
        The size of the batches to be used for training (default: 32).
    num_batches_per_epoch
        Number of batches to be processed in each training epoch
            (default: 50).
    trainer_kwargs
        Additional arguments to provide to ``pl.Trainer`` for construction.
    train_sampler
        Controls the sampling of windows during training.
    validation_sampler
        Controls the sampling of windows during validation.
    use_single_pass_sampling
        If True, use a single forward pass and sample N times from the saved distribution, much more efficient.
        If False, perform N forward passes and maintain N parallel prediction paths, this is true probalistic forecasting.
            (default: False)
    """

    @validated()
    def __init__(
            self,
            prediction_length: int,
            context_length: Optional[int] = None,
            input_size: int = 1,
            n_layer: int = 1,
            n_embd_per_head: int = 32,
            n_head: int = 4,
            max_context_length: int = 2048,
            rope_scaling=None,
            scaling: Optional[str] = "mean",
            lr: float = 1e-3,
            weight_decay: float = 1e-8,
            # Augmentations arguments
            aug_prob: float = 0.1,
            freq_mask_rate: float = 0.1,
            freq_mixing_rate: float = 0.1,
            jitter_prob: float = 0.0,
            jitter_sigma: float = 0.03,
            scaling_prob: float = 0.0,
            scaling_sigma: float = 0.1,
            rotation_prob: float = 0.0,
            permutation_prob: float = 0.0,
            permutation_max_segments: int = 5,
            permutation_seg_mode: str = "equal",
            magnitude_warp_prob: float = 0.0,
            magnitude_warp_sigma: float = 0.2,
            magnitude_warp_knot: int = 4,
            time_warp_prob: float = 0.0,
            time_warp_sigma: float = 0.2,
            time_warp_knot: int = 4,
            window_slice_prob: float = 0.0,
            window_slice_reduce_ratio: float = 0.9,
            window_warp_prob: float = 0.0,
            window_warp_window_ratio: float = 0.1,
            window_warp_scales: list = [0.5, 2.0],
            # Continuning model arguments
            distr_output: str = "studentT",
            loss: DistributionLoss = NegativeLogLikelihood(),
            num_parallel_samples: int = 100,
            batch_size: int = 32,
            num_batches_per_epoch: int = 50,
            trainer_kwargs: Optional[Dict[str, Any]] = None,
            train_sampler: Optional[InstanceSampler] = None,
            validation_sampler: Optional[InstanceSampler] = None,
            time_feat: bool = False,
            dynamic_feat_size: int = 0,
            use_covariates: bool = False,
            dropout: float = 0.0,
            lags_seq: list = ["Q", "M", "W", "D", "H", "T", "S"],
            data_id_to_name_map: dict = {},
            use_cosine_annealing_lr: bool = False,
            cosine_annealing_lr_args: dict = {},
            track_loss_per_series: bool = False,
            ckpt_path: Optional[str] = None,
            nonnegative_pred_samples: bool = False,
            use_single_pass_sampling: bool = False,
            device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
            covariate_field_sizes: Optional[Dict[str, int]] = None,
    ) -> None:
        default_trainer_kwargs = {"max_epochs": 100}
        if trainer_kwargs is not None:
            default_trainer_kwargs.update(trainer_kwargs)
        super().__init__(trainer_kwargs=default_trainer_kwargs)

        self.scaling = scaling
        self.input_size = input_size
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.max_context_length = max_context_length

        lag_indices = []
        for freq in lags_seq:
            try:
                lag_indices.extend(
                    get_lags_for_frequency(freq_str=freq, num_default_lags=1)
                )
            except Exception as e:
                warnings.warn(
                    f"Invalid frequency '{freq}' in lags_seq, skipping. Error: {str(e)}"
                )
                continue

        if len(lag_indices):
            lag_indices = sorted(set(lag_indices))
            # Limit max lag to avoid requiring too much history
            # For daily data, reasonable max lag is around 365 days (1 year)
            # For other frequencies, limit to context_length * 3 to allow some seasonality
            # But cap at 365 to avoid requiring too much history
            ctx_len = context_length or 64
            max_lag_limit = min(365, ctx_len * 3)

            original_max = max(lag_indices)
            lag_indices = [lag for lag in lag_indices if lag <= max_lag_limit]

            if len(lag_indices) == 0:
                # If all lags were filtered out, use a default small lag
                warnings.warn(
                    f"All lag indices exceeded limit {max_lag_limit}. Using default lag of 1."
                )
                lag_indices = [1]
            elif original_max > max_lag_limit:
                warnings.warn(
                    f"Some lag indices exceeded limit {max_lag_limit} (max was {original_max}). "
                    f"Filtered lags to max {max_lag_limit}."
                )

            # Store original lag indices for past_length calculation
            self.max_lag_original = max(lag_indices)
            # Convert to 0-based indexing for model use
            self.lags_seq = [lag_index - 1 for lag_index in lag_indices]
        else:
            self.lags_seq = []
            self.max_lag_original = 0

        self.n_head = n_head
        self.n_layer = n_layer
        self.n_embd_per_head = n_embd_per_head
        self.rope_scaling = rope_scaling

        self.lr = lr
        self.weight_decay = weight_decay
        if distr_output == "studentT":
            distr_output = StudentTOutput()
        elif distr_output == "neg_bin":
            distr_output = NegativeBinomialOutput()
        elif distr_output == "iqn":
            distr_output = ImplicitQuantileNetworkOutput()
        self.distr_output = distr_output
        self.num_parallel_samples = num_parallel_samples
        self.loss = loss
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.nonnegative_pred_samples = nonnegative_pred_samples
        self.use_single_pass_sampling = use_single_pass_sampling

        self.train_sampler = train_sampler or ExpectedNumInstanceSampler(
            num_instances=1.0,
            min_future=prediction_length,
            min_instances=1,
        )
        self.validation_sampler = validation_sampler or ValidationSplitSampler(
            min_future=prediction_length
        )

        self.aug_prob = aug_prob
        self.freq_mask_rate = freq_mask_rate
        self.freq_mixing_rate = freq_mixing_rate
        self.jitter_prob = jitter_prob
        self.jitter_sigma = jitter_sigma
        self.scaling_prob = scaling_prob
        self.scaling_sigma = scaling_sigma
        self.rotation_prob = rotation_prob
        self.permutation_prob = permutation_prob
        self.permutation_max_segments = permutation_max_segments
        self.permutation_seg_mode = permutation_seg_mode
        self.magnitude_warp_prob = magnitude_warp_prob
        self.magnitude_warp_sigma = magnitude_warp_sigma
        self.magnitude_warp_knot = magnitude_warp_knot
        self.time_warp_prob = time_warp_prob
        self.time_warp_sigma = time_warp_sigma
        self.time_warp_knot = time_warp_knot
        self.window_slice_prob = window_slice_prob
        self.window_slice_reduce_ratio = window_slice_reduce_ratio
        self.window_warp_prob = window_warp_prob
        self.window_warp_window_ratio = window_warp_window_ratio
        self.window_warp_scales = window_warp_scales
        self.track_loss_per_series = track_loss_per_series

        # -------- 协变量开关检查 --------
        # 当用户要求启用协变量但数据集中没有动态特征时，给出警告并禁用。
        if use_covariates and dynamic_feat_size == 0:
            warnings.warn(
                "use_covariates is True but dynamic_feat_size is 0. Disabling covariates."
            )
        self.time_feat = time_feat
        self.dynamic_feat_size = dynamic_feat_size if dynamic_feat_size is not None else 0
        self.use_covariates = use_covariates and self.dynamic_feat_size > 0
        self.dropout = dropout
        self.data_id_to_name_map = data_id_to_name_map
        self.ckpt_path = ckpt_path
        if covariate_field_sizes is None:
            covariate_field_sizes = {
                "dynamic_real": self.dynamic_feat_size if self.use_covariates else 0,
                "dynamic_cat": 0,
                "static_real": 0,
                "static_cat": 0,
            }
        self.covariate_field_sizes = {
            "dynamic_real": covariate_field_sizes.get("dynamic_real", 0),
            "dynamic_cat": covariate_field_sizes.get("dynamic_cat", 0),
            "static_real": covariate_field_sizes.get("static_real", 0),
            "static_cat": covariate_field_sizes.get("static_cat", 0),
        }
        self.has_dynamic_real = self.covariate_field_sizes["dynamic_real"] > 0
        self.has_dynamic_cat = self.covariate_field_sizes["dynamic_cat"] > 0
        self.has_static_real = self.covariate_field_sizes["static_real"] > 0
        self.has_static_cat = self.covariate_field_sizes["static_cat"] > 0

        self.use_cosine_annealing_lr = use_cosine_annealing_lr
        self.cosine_annealing_lr_args = cosine_annealing_lr_args
        self.device = device

    @classmethod
    def derive_auto_fields(cls, train_iter):
        stats = calculate_dataset_statistics(train_iter)

        return {
            "num_feat_dynamic_real": stats.num_feat_dynamic_real,
            "num_feat_static_cat": len(stats.feat_static_cat),
            "cardinality": [len(cats) for cats in stats.feat_static_cat],
        }

    def create_transformation(self) -> Transformation:
        if self.time_feat:
            return Chain(
                [
                    AddTimeFeatures(
                        start_field=FieldName.START,
                        target_field=FieldName.TARGET,
                        output_field=FieldName.FEAT_TIME,
                        time_features=time_features_from_frequency_str("S"),
                        pred_length=self.prediction_length,
                    ),
                    AddObservedValuesIndicator(
                        target_field=FieldName.TARGET,
                        output_field=FieldName.OBSERVED_VALUES,
                        imputation_method=DummyValueImputation(0.0),
                    ),
                ]
            )
        else:
            return Chain(
                [
                    AddObservedValuesIndicator(
                        target_field=FieldName.TARGET,
                        output_field=FieldName.OBSERVED_VALUES,
                        imputation_method=DummyValueImputation(0.0),
                    ),
                ]
            )

    def create_lightning_module(self, use_kv_cache: bool = False) -> pl.LightningModule:
        if self.ckpt_path is not None:
            # Load checkpoint to get the original model_kwargs
            ckpt = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
            ckpt_model_kwargs = ckpt["hyper_parameters"]["model_kwargs"].copy()

            # Use checkpoint's model_kwargs to ensure feature_size matches
            # But allow overriding context_length, prediction_length, and rope_scaling
            model_kwargs = {
                "input_size": ckpt_model_kwargs["input_size"],
                "context_length": self.context_length,  # Allow override
                "max_context_length": ckpt_model_kwargs.get("max_context_length", self.max_context_length),
                "lags_seq": ckpt_model_kwargs["lags_seq"],  # Critical: must match checkpoint
                "n_layer": ckpt_model_kwargs["n_layer"],
                "n_embd_per_head": ckpt_model_kwargs["n_embd_per_head"],
                "n_head": ckpt_model_kwargs["n_head"],
                "scaling": ckpt_model_kwargs["scaling"],
                "distr_output": self.distr_output,  # Can use current if compatible
                "num_parallel_samples": self.num_parallel_samples,
                "rope_scaling": self.rope_scaling,  # Allow override
                "time_feat": ckpt_model_kwargs.get("time_feat", self.time_feat),  # Must match checkpoint
                "dynamic_feat_size": ckpt_model_kwargs.get("dynamic_feat_size", 0),  # Must match checkpoint
                "use_covariates": ckpt_model_kwargs.get("use_covariates", False),  # Must match checkpoint
                "dropout": ckpt_model_kwargs.get("dropout", self.dropout),
            }
        else:
            model_kwargs = {
                "input_size": self.input_size,
                "context_length": self.context_length,
                "max_context_length": self.max_context_length,
                "lags_seq": self.lags_seq,
                "n_layer": self.n_layer,
                "n_embd_per_head": self.n_embd_per_head,
                "n_head": self.n_head,
                "scaling": self.scaling,
                "distr_output": self.distr_output,
                "num_parallel_samples": self.num_parallel_samples,
                "rope_scaling": self.rope_scaling,
                "time_feat": self.time_feat,
                "dynamic_feat_size": self.dynamic_feat_size,
                "use_covariates": self.use_covariates,
                "dropout": self.dropout,
            }

        if self.ckpt_path is not None:
            return LagLlamaLightningModule.load_from_checkpoint(
                checkpoint_path=self.ckpt_path,
                map_location=self.device,
                strict=False,
                loss=self.loss,
                lr=self.lr,
                weight_decay=self.weight_decay,
                context_length=self.context_length,
                prediction_length=self.prediction_length,
                model_kwargs=model_kwargs,
                # Augmentations
                aug_prob=self.aug_prob,
                freq_mask_rate=self.freq_mask_rate,
                freq_mixing_rate=self.freq_mixing_rate,
                jitter_prob=self.jitter_prob,
                jitter_sigma=self.jitter_sigma,
                scaling_prob=self.scaling_prob,
                scaling_sigma=self.scaling_sigma,
                rotation_prob=self.rotation_prob,
                permutation_prob=self.permutation_prob,
                permutation_max_segments=self.permutation_max_segments,
                permutation_seg_mode=self.permutation_seg_mode,
                magnitude_warp_prob=self.magnitude_warp_prob,
                magnitude_warp_sigma=self.magnitude_warp_sigma,
                magnitude_warp_knot=self.magnitude_warp_knot,
                time_warp_prob=self.time_warp_prob,
                time_warp_sigma=self.time_warp_sigma,
                time_warp_knot=self.time_warp_knot,
                window_slice_prob=self.window_slice_prob,
                window_slice_reduce_ratio=self.window_slice_reduce_ratio,
                window_warp_prob=self.window_warp_prob,
                window_warp_window_ratio=self.window_warp_window_ratio,
                window_warp_scales=self.window_warp_scales,
                use_kv_cache=use_kv_cache,
                data_id_to_name_map=self.data_id_to_name_map,
                use_cosine_annealing_lr=self.use_cosine_annealing_lr,
                cosine_annealing_lr_args=self.cosine_annealing_lr_args,
                track_loss_per_series=self.track_loss_per_series,
                nonnegative_pred_samples=self.nonnegative_pred_samples,
                covariate_field_sizes=self.covariate_field_sizes,
            )
        else:
            return LagLlamaLightningModule(
                loss=self.loss,
                lr=self.lr,
                weight_decay=self.weight_decay,
                context_length=self.context_length,
                prediction_length=self.prediction_length,
                model_kwargs=model_kwargs,
                # Augmentations
                aug_prob=self.aug_prob,
                freq_mask_rate=self.freq_mask_rate,
                freq_mixing_rate=self.freq_mixing_rate,
                jitter_prob=self.jitter_prob,
                jitter_sigma=self.jitter_sigma,
                scaling_prob=self.scaling_prob,
                scaling_sigma=self.scaling_sigma,
                rotation_prob=self.rotation_prob,
                permutation_prob=self.permutation_prob,
                permutation_max_segments=self.permutation_max_segments,
                permutation_seg_mode=self.permutation_seg_mode,
                magnitude_warp_prob=self.magnitude_warp_prob,
                magnitude_warp_sigma=self.magnitude_warp_sigma,
                magnitude_warp_knot=self.magnitude_warp_knot,
                time_warp_prob=self.time_warp_prob,
                time_warp_sigma=self.time_warp_sigma,
                time_warp_knot=self.time_warp_knot,
                window_slice_prob=self.window_slice_prob,
                window_slice_reduce_ratio=self.window_slice_reduce_ratio,
                window_warp_prob=self.window_warp_prob,
                window_warp_window_ratio=self.window_warp_window_ratio,
                window_warp_scales=self.window_warp_scales,
                use_kv_cache=use_kv_cache,
                data_id_to_name_map=self.data_id_to_name_map,
                use_cosine_annealing_lr=self.use_cosine_annealing_lr,
                cosine_annealing_lr_args=self.cosine_annealing_lr_args,
                track_loss_per_series=self.track_loss_per_series,
                nonnegative_pred_samples=self.nonnegative_pred_samples,
                covariate_field_sizes=self.covariate_field_sizes,
            )

    def _get_time_series_fields(self):
        # 生成传递给 InstanceSplitter 的字段列表：基础的观测值、时间特征以及动态协变量
        fields = [FieldName.OBSERVED_VALUES]
        if self.time_feat:
            fields.append(FieldName.FEAT_TIME)
        if self.use_covariates and self.has_dynamic_real:
            fields.append(FieldName.FEAT_DYNAMIC_REAL)
        if self.use_covariates and self.has_dynamic_cat:
            fields.append(FieldName.FEAT_DYNAMIC_CAT)
        return fields

    def _additional_field_names(self):
        # dataloader 输出时附加的张量名，与上面的字段保持一一对应
        extra_fields = []
        if self.time_feat:
            extra_fields += ["past_time_feat", "future_time_feat"]
        if self.use_covariates:
            if self.has_dynamic_real:
                extra_fields += ["past_feat_dynamic_real", "future_feat_dynamic_real"]
            if self.has_dynamic_cat:
                extra_fields += ["past_feat_dynamic_cat", "future_feat_dynamic_cat"]
            if self.has_static_real:
                extra_fields += ["feat_static_real"]
            if self.has_static_cat:
                extra_fields += ["feat_static_cat"]
        return extra_fields

    def _create_instance_splitter(self, module: LagLlamaLightningModule, mode: str):
        # 根据训练/验证/测试模式创建窗口切分器，决定可用的时间序列字段
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]
        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=self.context_length + (self.max_lag_original if hasattr(self, 'max_lag_original') else (
                max(self.lags_seq) + 1 if len(self.lags_seq) > 0 else 0)),
            future_length=self.prediction_length,
            time_series_fields=self._get_time_series_fields(),
            dummy_value=self.distr_output.value_in_support,
        )

    def create_training_data_loader(
            self,
            data: Dataset,
            module: LagLlamaLightningModule,
            shuffle_buffer_length: Optional[int] = None,
            **kwargs,
    ) -> Iterable:
        data = Cyclic(data).stream()
        instances = self._create_instance_splitter(module, "training").apply(
            data, is_train=True
        )
        field_names = TRAINING_INPUT_NAMES + self._additional_field_names()
        # as_stacked_batches 会把所需字段堆叠成张量，这里附带时间特征与协变量
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            shuffle_buffer_length=shuffle_buffer_length,
            field_names=field_names,
            output_type=torch.tensor,
            num_batches_per_epoch=self.num_batches_per_epoch,
        )

    def create_validation_data_loader(
            self,
            data: Dataset,
            module: LagLlamaLightningModule,
            **kwargs,
    ) -> Iterable:
        instances = self._create_instance_splitter(module, "validation").apply(
            data, is_train=True
        )
        field_names = TRAINING_INPUT_NAMES + self._additional_field_names()
        # 验证阶段同样保持额外字段，方便 Lightning 计算 loss
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            field_names=field_names,
            output_type=torch.tensor,
        )

    def create_predictor(
            self,
            transformation: Transformation,
            module,
    ) -> PyTorchPredictor:
        prediction_splitter = self._create_instance_splitter(module, "test")
        # 推理阶段的输入名必须和前面 dataloader 输出一致，否则 Predictor 无法取值
        input_names = list(PREDICTION_INPUT_NAMES)
        if self.time_feat:
            input_names += ["past_time_feat", "future_time_feat"]
        if self.use_covariates:
            if self.has_dynamic_real:
                input_names += ["past_feat_dynamic_real", "future_feat_dynamic_real"]
            if self.has_dynamic_cat:
                input_names += ["past_feat_dynamic_cat", "future_feat_dynamic_cat"]
            if self.has_static_real:
                input_names += ["feat_static_real"]
            if self.has_static_cat:
                input_names += ["feat_static_cat"]
        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=input_names,
            prediction_net=module,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
