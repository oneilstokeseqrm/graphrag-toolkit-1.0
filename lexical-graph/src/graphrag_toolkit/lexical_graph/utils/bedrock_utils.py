# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import llama_index.llms.bedrock_converse.utils
from typing import Any, Callable

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

logger = logging.getLogger(__name__)



def _create_retry_decorator(client: Any, max_retries: int) -> Callable[[Any], Any]:
    min_seconds = 4
    max_seconds = 30
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 30 seconds, then 30 seconds afterwards
    try:
        import boto3  # noqa
    except ImportError as e:
        raise ImportError(
            "boto3 package not found, install with 'pip install boto3'"
        ) from e
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_random_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
            retry_if_exception_type(client.exceptions.ThrottlingException) | 
            retry_if_exception_type(client.exceptions.ModelTimeoutException) |
            retry_if_exception_type(client.exceptions.ModelErrorException)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    
llama_index.llms.bedrock_converse.utils._create_retry_decorator = _create_retry_decorator
