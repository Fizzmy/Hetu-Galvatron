# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import abc
from typing import Dict, Tuple

import torch
from torch import Tensor


class BaseInferenceContext(abc.ABC):
    """Base class for inference contexts.

    Currently extended by `StaticInferenceContext` and `DynamicInferenceContext`.
    Extend this class for any future contexts types.
    """

    @abc.abstractmethod
    def is_static_batching(self) -> bool:
        """Return `True` if context uses static batching."""
        pass

    def is_dynamic_batching(self) -> bool:
        """Return `True` if context uses dynamic batching."""
        return not self.is_static_batching()


class StaticInferenceContext(BaseInferenceContext):
    """Static inference context for managing KV cache during autoregressive generation.

    Args:
        max_batch_size (int): Max supported batch size.
        max_sequence_length (int): Max supported sequence length.
    """

    def __init__(self, max_batch_size: int, max_sequence_length: int):
        self.max_sequence_length = max_sequence_length
        self.max_batch_size = max_batch_size
        self.current_batch_size = max_batch_size
        self.sequence_len_offset = 0
        self.batch_size_offset = 0
        self.key_value_memory_dict: Dict[int, Tuple[Tensor, Tensor]] = {}
        self.decode_mode = False

    def is_static_batching(self) -> bool:
        return True

    def enable_prefill_mode(self):
        self.decode_mode = False

    def enable_decode_mode(self):
        self.decode_mode = True

    def is_decode_only(self):
        return self.decode_mode

    def reset(self):
        self.current_batch_size = self.max_batch_size
        self.sequence_len_offset = 0
        self.batch_size_offset = 0
        self.key_value_memory_dict.clear()
        self.enable_prefill_mode()

    def swap_key_value_dict(self, batch_idx):
        if len(self.key_value_memory_dict) == 0:
            raise ValueError("should not swap when dict is empty")
        for layer_number in self.key_value_memory_dict.keys():
            inference_key_memory, inference_value_memory = self.key_value_memory_dict[layer_number]
            assert len(batch_idx) == inference_key_memory.shape[1]
            new_inference_key_memory = inference_key_memory[:, batch_idx]
            new_inference_value_memory = inference_value_memory[:, batch_idx]
            self.key_value_memory_dict[layer_number] = (
                new_inference_key_memory,
                new_inference_value_memory,
            )

    def __str__(self):
        return (
            f"StaticInferenceContext(max_seq_len={self.max_sequence_length}, "
            f"max_batch_size={self.max_batch_size}, "
            f"sequence_len_offset={self.sequence_len_offset}, "
            f"decode_mode={self.decode_mode})"
        )
