# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import setuptools

deps = []
with open('./requirements.txt') as f:
    for line in f.readlines():
        if not line.strip():
            continue
        deps.append(line.strip())

setuptools.setup(
    name='graphrag-toolkit',
    description="AWS GraphRAG Toolkit",
    packages=setuptools.find_packages(where="."),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=deps,
    version="2.5.0"
)