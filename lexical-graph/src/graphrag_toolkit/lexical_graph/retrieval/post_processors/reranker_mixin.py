# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import List, Tuple

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

class RerankerMixin(ABC):

    @property
    @abstractmethod
    def batch_size(self):
        pass

    @abstractmethod
    def rerank_pairs(self, pairs: List[Tuple[str, str]], batch_size: int = 128) -> List[float]:
        pass
