#
# SPDX-License-Identifier: Apache-2.0
#

MPI = 1

include ../../../../makefiles/common.mk
include ../../../../makefiles/examples.mk

TARGETS = dtype_allreduce_scaling_mpi dtype_value_count_scaling_mpi dtype_value_count_single_case_mpi dtype_error_compare_mpi
OBJECTS = main.o value_count_scaling.o single_case_value_count_scaling.o dtype_error_compare.o dtype_error_compare_kernels.o

all: $(TARGETS)

dtype_allreduce_scaling_mpi: main.o
	$(MPICXX) $(CXXFLAGS) main.o $(LIBRARIES) $(LDFLAGS) -o $@
	@echo "Built target $@"

dtype_value_count_scaling_mpi: value_count_scaling.o
	$(MPICXX) $(CXXFLAGS) value_count_scaling.o $(LIBRARIES) $(LDFLAGS) -o $@
	@echo "Built target $@"

dtype_value_count_single_case_mpi: single_case_value_count_scaling.o
	$(MPICXX) $(CXXFLAGS) single_case_value_count_scaling.o $(LIBRARIES) $(LDFLAGS) -o $@
	@echo "Built target $@"

dtype_error_compare_mpi: dtype_error_compare.o dtype_error_compare_kernels.o
	$(MPICXX) $(CXXFLAGS) dtype_error_compare.o dtype_error_compare_kernels.o $(LIBRARIES) $(LDFLAGS) -o $@
	@echo "Built target $@"

%.o: %.cu
	$(NVCC) $(NVCUFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cc
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGETS)

.PHONY: all clean
