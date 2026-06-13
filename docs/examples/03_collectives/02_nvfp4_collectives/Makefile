#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

include ../../../../makefiles/common.mk
include ../../../../makefiles/examples.mk

TARGET = nvfp4_collectives
SOURCES = main.cc
OBJECTS = $(SOURCES:.cc=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(LIBRARIES) $(LDFLAGS) -lpthread -o $@
	@echo "Built target $@"

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

test: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all test clean
