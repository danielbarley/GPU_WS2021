CUDA_ROOT=$(CUDA_HOME)
INC=-I./inc -I. -I$(CUDA_ROOT)/include
LIB=-L$(CUDA_ROOT)/lib64
NVCC=$(CUDA_ROOT)/bin/nvcc

.PHONY: build
build: ./bin/reduction

.PHONY: clean
clean:
	rm -f ./bin/*
	
.PHONY: rebuild
rebuild: clean build

./bin/reduction: ./src/main.cpp ./src/kernel.cu
	$(NVCC) -O2 --compiler-options "-O2 -Wall -Wextra" -o $@ $^ $(INC) $(LIB)
