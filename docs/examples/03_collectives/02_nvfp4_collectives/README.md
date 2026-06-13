<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# NVFP4 Collectives Example

This example validates `ncclAllReduceNvfp4` and `ncclAlltoAllNvfp4` against the
standard NCCL collectives using fp16 and bf16 user buffers. It exercises exact
block counts, tail blocks, eager launches, CUDA graph capture, and a few
negative API checks.
