#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

MPI = 1

include ../../../../makefiles/common.mk
include ../../../../makefiles/examples.mk

TARGET = nvfp4_allreduce_mpi
SOURCES = main.cc
OBJECTS = $(SOURCES:.cc=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(MPICXX) $(CXXFLAGS) $(OBJECTS) $(LIBRARIES) $(LDFLAGS) -o $@
	@echo "Built target $@"

%.o: %.cc
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

test: $(TARGET)
	$(MPIRUN) -np 2 ./$(TARGET) --mode correctness --correctness-bytes 4MB,4194306

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all test clean
