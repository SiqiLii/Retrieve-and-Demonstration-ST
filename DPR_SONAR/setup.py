#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from setuptools import setup


setup(
        name="dpr_sonar",
    install_requires=[
        "faiss-cpu>=1.6.1",
        "filelock",
        "numpy",
        "regex",
        "torch>=1.5.0",
        "transformers>=4.3",
        "tqdm>=4.27",
        "wget",
        "spacy>=2.1.8",
        "hydra-core>=1.0.0",
        "omegaconf>=2.0.1",
        "jsonlines",
        "soundfile",
        "editdistance",
    ],
)