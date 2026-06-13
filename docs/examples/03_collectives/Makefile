#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

# NCCL Collective Examples
EXAMPLES = 01_allreduce 02_nvfp4_collectives 03_nvfp4_allreduce_mpi

# Default target
all: $(EXAMPLES)

# Build individual examples
$(EXAMPLES):
	$(MAKE) -C $@

# Clean all build artifacts
clean:
	for example in $(EXAMPLES); do \
		$(MAKE) -C $$example clean; \
	done

# Test all examples
test: all
	for example in $(EXAMPLES); do \
		echo "Testing $$example..."; \
		$(MAKE) -C $$example test; \
	done

# Help
help:
	@echo "NCCL Collective Communication Examples"
	@echo "====================================="
	@echo ""
	@echo "Targets:"
	@echo "  all     - Build all examples"
	@echo "  clean   - Clean all build artifacts"
	@echo "  test    - Test all examples"
	@echo "  help    - Show this help"
	@echo ""
	@echo "Examples:"
	@echo "  01_allreduce - AllReduce collective operation"
	@echo "  02_nvfp4_collectives - NVFP4 allreduce and alltoall verification"
	@echo "  03_nvfp4_allreduce_mpi - MPI correctness and scalability harness for NVFP4 allreduce"
	@echo ""
	@echo "To build/run individual examples:"
	@echo "  make -C 01_allreduce"
	@echo "  make -C 02_nvfp4_collectives"
	@echo "  make -C 03_nvfp4_allreduce_mpi"

.PHONY: all clean test help $(EXAMPLES)
