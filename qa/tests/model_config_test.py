# Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest

from tritonclient.utils import InferenceServerException

from helpers.triton_model_config import Model, LiteRTTritonModel
from helpers.helper_functions import (
    load_model,
)


@pytest.mark.parametrize(
    "model_config",
    [
        LiteRTTritonModel(
            "add",
            [Model.TensorIO("X_input_xxx", "TYPE_FP32", [1])],
            [Model.TensorIO("ADD_TOP", "TYPE_FP32", [1])],
        ),
    ],
)
def test_incorrect_input_name(
    tritonserver_client,
    model_config: LiteRTTritonModel,
):
    with pytest.raises(InferenceServerException):
        load_model(
            tritonserver_client.client,
            model_config,
        )


@pytest.mark.parametrize(
    "model_config",
    [
        LiteRTTritonModel(
            "add",
            [Model.TensorIO("X_input", "TYPE_FP32", [1])],
            [Model.TensorIO("ADD_TOP_xxx", "TYPE_FP32", [1])],
        ),
    ],
)
def test_incorrect_output_name(
    tritonserver_client,
    model_config: LiteRTTritonModel,
):
    with pytest.raises(InferenceServerException):
        load_model(
            tritonserver_client.client,
            model_config,
        )


@pytest.mark.parametrize(
    "model_config",
    [
        LiteRTTritonModel(
            "add",
            [
                Model.TensorIO("X_input", "TYPE_FP32", [1]),
                Model.TensorIO("extra_input", "TYPE_FP32", [1]),
            ],
            [Model.TensorIO("ADD_TOP", "TYPE_FP32", [1])],
        ),
    ],
)
def test_extra_input(
    tritonserver_client,
    model_config: LiteRTTritonModel,
):
    with pytest.raises(InferenceServerException):
        load_model(
            tritonserver_client.client,
            model_config,
        )


@pytest.mark.parametrize(
    "model_config",
    [
        LiteRTTritonModel(
            "add",
            [Model.TensorIO("X_input", "TYPE_FP32", [1])],
            [
                Model.TensorIO("ADD_TOP", "TYPE_FP32", [1]),
                Model.TensorIO("extra_output", "TYPE_FP32", [1]),
            ],
        ),
    ],
)
def test_extra_output(
    tritonserver_client,
    model_config: LiteRTTritonModel,
):
    with pytest.raises(InferenceServerException):
        load_model(
            tritonserver_client.client,
            model_config,
        )


@pytest.mark.parametrize(
    "model_config",
    [
        LiteRTTritonModel(
            "add",
            [Model.TensorIO("X_input", "TYPE_INT32", [1])],
            [Model.TensorIO("ADD_TOP", "TYPE_FP32", [1])],
        ),
        LiteRTTritonModel(
            "add",
            [Model.TensorIO("X_input", "TYPE_FP32", [1])],
            [Model.TensorIO("ADD_TOP", "TYPE_INT32", [1])],
        ),
    ],
)
def test_incorrect_datatypes(
    tritonserver_client,
    model_config: LiteRTTritonModel,
):
    with pytest.raises(InferenceServerException):
        load_model(
            tritonserver_client.client,
            model_config,
        )


@pytest.mark.parametrize(
    "model_config",
    [
        LiteRTTritonModel(
            "add",
            [Model.TensorIO("X_input", "TYPE_FP32", [5])],
            [Model.TensorIO("ADD_TOP", "TYPE_FP32", [5])],
        ),
    ],
)
def test_incorrect_shapes(
    tritonserver_client,
    model_config: LiteRTTritonModel,
):
    with pytest.raises(InferenceServerException):
        load_model(
            tritonserver_client.client,
            model_config,
        )


@pytest.mark.parametrize(
    "model_config",
    [
        LiteRTTritonModel(
            "add",
            [Model.TensorIO("X_input", "TYPE_FP32", [1])],
            [Model.TensorIO("ADD_TOP", "TYPE_FP32", [1])],
            litert_num_threads=-2,
        ),
        LiteRTTritonModel(
            "add",
            [Model.TensorIO("X_input", "TYPE_FP32", [1])],
            [Model.TensorIO("ADD_TOP", "TYPE_FP32", [1])],
            edgetpu=True,
            edgetpu_parameters={"invalid_name": 3},
        ),
        LiteRTTritonModel(
            "add",
            [Model.TensorIO("X_input", "TYPE_FP32", [1])],
            [Model.TensorIO("ADD_TOP", "TYPE_FP32", [1])],
            edgetpu=True,
            edgetpu_parameters={"performance": "Invalid"},
        ),
    ],
)
def test_invalid_runtime_params(
    tritonserver_client,
    model_config: LiteRTTritonModel,
):
    with pytest.raises(InferenceServerException):
        load_model(
            tritonserver_client.client,
            model_config,
        )
