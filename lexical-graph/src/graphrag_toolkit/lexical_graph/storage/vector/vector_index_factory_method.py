# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import abc
from typing import List

from graphrag_toolkit.lexical_graph.storage.vector.vector_index import VectorIndex

class VectorIndexFactoryMethod():
    @abc.abstractmethod
    def try_create(self, index_names:List[str], vector_index_info:str, **kwargs) -> List[VectorIndex]:
        raise NotImplementedError