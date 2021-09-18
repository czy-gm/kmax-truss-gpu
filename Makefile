CUDA_DIR=/usr/local/cuda
SOURCE_DIR=src
CXX=g++
CXX_FLAGS=-fopenmp -std=c++14 -Wall -O3 -Iinclude
NVCC=${CUDA_DIR}/bin/nvcc
NVCC_FLAGS=-gencode=arch=compute_70,code=sm_70 -std=c++14 -O3 -Iinclude

all: kt

kt: main.o kt.o link.o
	${CXX} ${CXX_FLAGS} -o kt main.o kt.o link.o ${SOURCE_DIR}/log.cpp -L ${CUDA_DIR}/lib64 -lcuda -lcudart

main.o: ${SOURCE_DIR}/main.cpp
	${CXX} ${CXX_FLAGS} -o main.o -c ${SOURCE_DIR}/main.cpp

kt.o: src/kt.cu
	${NVCC} ${NVCC_FLAGS} -o kt.o -dc ${SOURCE_DIR}/kt.cu

link.o: kt.o
	${NVCC} ${NVCC_FLAGS} -o link.o -dlink kt.o

clean:
	rm -f *.o kt