# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fast Image processor class for Beit."""

from typing import Optional, Union

import torch
from torchvision.transforms import functional as F

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    pil_torch_interpolation_mapping,
    validate_kwargs,
)
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    add_start_docstrings,
)


class BeitFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    do_reduce_labels: Optional[bool]


@add_start_docstrings(
    "Constructs a fast Beit image processor.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    """
    do_reduce_labels (`bool`, *optional*, defaults to `self.do_reduce_labels`):
        Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0
        is used for background, and background itself is not included in all classes of a dataset (e.g.
        ADE20k). The background label will be replaced by 255.
    """,
)
class BeitImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 224, "width": 224}
    default_to_square = True
    crop_size = {"height": 224, "width": 224}
    do_resize = True
    do_center_crop = False
    do_rescale = True
    do_normalize = True
    do_reduce_labels = False
    valid_kwargs = BeitFastImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[BeitFastImageProcessorKwargs]):
        super().__init__(**kwargs)

    def reduce_label(self, label: list["torch.Tensor"]):
        label = torch.where(label == 0, torch.tensor(255, dtype=label.dtype), label)
        label = label - 1
        label = torch.where(label == 254, torch.tensor(255, dtype=label.dtype), label)

        return label

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_reduce_labels: bool,
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ) -> BatchFeature:
        if do_reduce_labels:
            images = self.reduce_label(images)

        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(image=stacked_images, size=size, interpolation=interpolation)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_center_crop:
                stacked_images = self.center_crop(stacked_images, crop_size)
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images
        return processed_images

    def _preprocess_images(self, images: ImageInput, **kwargs):
        return self._preprocess(
            images,
            **kwargs,
        )

    def _preprocess_segmentation_maps(
        self,
        segmentation_maps,
        **kwargs,
    ):
        """Preprocesses a single segmentation map."""
        # Add an axis to the segmentation maps for transformations.
        kwargs["do_normalize"] = False
        kwargs["do_rescale"] = False
        kwargs["input_data_format"] = ChannelDimension.FIRST
        segmentation_maps = self._preprocess(images=segmentation_maps, **kwargs)
        segmentation_maps = [seg_map.to(torch.uint64) for seg_map in segmentation_maps]
        return segmentation_maps

    def __call__(self, images, segmentation_maps=None, **kwargs):
        # Overrides the `__call__` method of the `Preprocessor` class such that the images and segmentation maps can both
        # be passed in as positional arguments.
        return super().__call__(images, segmentation_maps=segmentation_maps, **kwargs)

    @add_start_docstrings(
        "Constructs a fast Beit image processor.",
        BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
        """
        do_reduce_labels (`bool`, *optional*, defaults to `self.do_reduce_labels`):
            Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0
            is used for background, and background itself is not included in all classes of a dataset (e.g.
            ADE20k). The background label will be replaced by 255.
        """,
    )
    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: Optional[ImageInput] = None,
        **kwargs: Unpack[DefaultFastImageProcessorKwargs],
    ) -> BatchFeature:
        validate_kwargs(captured_kwargs=kwargs.keys(), valid_processor_keys=self.valid_kwargs.__annotations__.keys())
        # Set default kwargs from self. This ensures that if a kwarg is not provided
        # by the user, it gets its default value from the instance, or is set to None.
        for kwarg_name in self.valid_kwargs.__annotations__:
            kwargs.setdefault(kwarg_name, getattr(self, kwarg_name, None))

        # Extract parameters that are only used for preparing the input images
        do_convert_rgb = kwargs.pop("do_convert_rgb")
        input_data_format = kwargs.pop("input_data_format")
        device = kwargs.pop("device")
        # Prepare input images
        images = self._prepare_input_images(
            images=images, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device
        )

        # Prepare segmentation maps
        if segmentation_maps:
            segmentation_maps = self._prepare_input_images(
                images=segmentation_maps,
                do_convert_rgb=do_convert_rgb,
                input_data_format=input_data_format,
                device=device,
            )

        # Update kwargs that need further processing before being validated
        kwargs = self._further_process_kwargs(**kwargs)

        # Validate kwargs
        self._validate_preprocess_kwargs(**kwargs)

        # torch resize uses interpolation instead of resample
        resample = kwargs.pop("resample")
        kwargs["interpolation"] = (
            pil_torch_interpolation_mapping[resample] if isinstance(resample, (PILImageResampling, int)) else resample
        )

        # Pop kwargs that are not needed in _preprocess
        kwargs.pop("default_to_square")
        kwargs.pop("data_format")

        images = self._preprocess_images(
            images=images,
            **kwargs,
        )
        data = {"pixel_values": images}

        if segmentation_maps:
            segmentation_maps = self._preprocess_segmentation_maps(
                segmentation_maps=segmentation_maps,
                **kwargs,
            )
            data["labels"] = segmentation_maps

        return BatchFeature(data=data, tensor_type=kwargs["return_tensors"])


__all__ = ["BeitImageProcessorFast"]
