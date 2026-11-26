# Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

from __future__ import annotations
from typing import List


class Model:
    def __init__(
        self,
        name: str,
        inputs: List[TensorIO],
        outputs: List[TensorIO],
        gpu: int = 0,
        cpu: int = 1,
        max_batch_size: int = 0,
        warm_up: bool = False,
    ):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.gpu = gpu
        self.cpu = cpu
        self.max_batch_size = max_batch_size
        self.warm_up = warm_up

    class TensorIO:
        def __init__(
            self, name: str, datatype: str, dims: List, label_filename: str = None
        ):
            self.name = name
            self.datatype = datatype
            self.dims = dims
            self.label_filename = label_filename


class LiteRTTritonModel(Model):
    def __init__(
        self,
        name: str,
        inputs: List[Model.TensorIO],
        outputs: List[Model.TensorIO],
        litert_num_threads: int = None,
        gpu: int = 0,
        cpu: int = 1,
        max_batch_size: int = 0,
        warm_up: bool = False,
        edgetpu: bool = False,
        edgetpu_parameters: dict = {},
    ):
        super().__init__(
            name,
            inputs,
            outputs,
            gpu=gpu,
            cpu=cpu,
            max_batch_size=max_batch_size,
            warm_up=warm_up,
        )
        self.litert_num_threads = litert_num_threads
        self.edgetpu = edgetpu
        self.edgetpu_parameters = edgetpu_parameters

    def set_thread_count(self, num_threads: int):
        self.litert_num_threads = num_threads
