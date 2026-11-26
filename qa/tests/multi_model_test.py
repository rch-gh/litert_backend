# Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest
from typing import List


from helpers.triton_model_config import Model, LiteRTTritonModel
from helpers.helper_functions import (
    load_model,
    send_inference_request,
    unload_model,
    is_server_ready,
)


@pytest.mark.parametrize(
    "model_configs",
    [
        [
            LiteRTTritonModel(
                "mobilenet_v1_1.0_224",
                [Model.TensorIO("input", "TYPE_FP32", [1, 224, 224, 3])],
                [
                    Model.TensorIO(
                        "MobilenetV1/Predictions/Reshape_1",
                        "TYPE_FP32",
                        [1, 1001],
                    )
                ],
                max_batch_size=0,
                litert_num_threads=1,
            ),
            LiteRTTritonModel(
                "add",
                [Model.TensorIO("X_input", "TYPE_FP32", [1])],
                [Model.TensorIO("ADD_TOP", "TYPE_FP32", [1])],
                litert_num_threads=3,
            ),
        ],
        [
            LiteRTTritonModel(
                "mobilenet_v1_1.0_224",
                [Model.TensorIO("input", "TYPE_FP32", [1, 224, 224, 3])],
                [
                    Model.TensorIO(
                        "MobilenetV1/Predictions/Reshape_1",
                        "TYPE_FP32",
                        [1, 1001],
                    )
                ],
                max_batch_size=0,
                litert_num_threads=3,
            ),
            LiteRTTritonModel(
                "add",
                [Model.TensorIO("X_input", "TYPE_FP32", [1])],
                [Model.TensorIO("ADD_TOP", "TYPE_FP32", [1])],
                litert_num_threads=1,
            ),
        ],
        [
            LiteRTTritonModel(
                "inceptionv3_dynamic",
                [Model.TensorIO("serving_default_inputs:0", "TYPE_FP32", [-1, -1, 3])],
                [
                    Model.TensorIO(
                        "StatefulPartitionedCall:0",
                        "TYPE_FP32",
                        [1001],
                    )
                ],
            )
        ],
    ],
)
def test_differing_thread_counts(
    tritonserver_client,
    model_configs: List[LiteRTTritonModel],
):
    # Load each model in sequence
    for model_config in model_configs:
        load_model(
            tritonserver_client.client,
            model_config,
        )

    assert is_server_ready(tritonserver_client.client)

    # Request inference from each model in sequence
    for model_config in model_configs:
        send_inference_request(
            tritonserver_client.client, tritonserver_client.module, model_config
        )

    # After both models have inference requested, assert triton has not segfaulted
    assert is_server_ready(tritonserver_client.client)

    # Now unload models and make sure everything still behaves
    for model_config in model_configs:
        unload_model(tritonserver_client.client, model_config.name)

    assert is_server_ready(
        tritonserver_client.client
    ), "Triton never became ready after unload"
