# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pipe import Pipe

def _sink():
    def _sink_from(generator):
        for item in generator:
            pass
    return Pipe(_sink_from)

sink = _sink()