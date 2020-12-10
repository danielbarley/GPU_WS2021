#include <chrono>
#include <iostream>
#include <random>
#include <string>

#include <argparse.hpp>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 5120000 / 8
#endif /* ifndef BLOCK_SIZE */

void mmul_unoptimized(double const *A, double const *B, double *C, size_t dim);

int main(int argc, const char **argv) {
	// get args
	argparse::ArgumentParser ap;
	ap.addArgument("-s", "--size", 1, false);
	ap.addArgument("-p", "--print", 1, false);
	ap.parse(argc, argv);

	size_t size = ap.retrieve<size_t>("size");
	bool printing = ap.retrieve<bool>("print");
	// setup matrices
	double *A = (double *)calloc(size * size, sizeof(double));
	double *B = (double *)calloc(size * size, sizeof(double));
	double *C = (double *)calloc(size * size, sizeof(double));
	for (size_t i = 0; i < size; ++i) {
		for (size_t j = 0; j < size; ++j) {
			A[i * size + j] = i + j;
		}
	}
	for (size_t i = 0; i < size; ++i) {
		for (size_t j = 0; j < size; ++j) {
			B[i * size + j] = i * j;
		}
	}

	mmul_unoptimized(A, B, C, size);
	auto start_unopt = std::chrono::high_resolution_clock::now();
	mmul_unoptimized(A, B, C, size);
	auto stop_unopt = std::chrono::high_resolution_clock::now();
	auto time_unopt = std::chrono::duration_cast<std::chrono::nanoseconds>(
			      stop_unopt - start_unopt)
			      .count();

	double FLOP = size * size * size * 2.0;
	if (printing) {
		std::cout << "======================" << std::endl;
		std::cout << "A:" << std::endl;
		for (size_t i = 0; i < size; ++i) {
			for (size_t j = 0; j < size; ++j) {
				std::cout << A[i * size + j] << ", ";
			}
			std::cout << std::endl;
		}
		std::cout << "----------------------" << std::endl;
		std::cout << "B:" << std::endl;
		for (size_t i = 0; i < size; ++i) {
			for (size_t j = 0; j < size; ++j) {
				std::cout << B[i * size + j] << ", ";
			}
			std::cout << std::endl;
		}
		std::cout << "----------------------" << std::endl;
		std::cout << "C:" << std::endl;
		for (size_t i = 0; i < size; ++i) {
			for (size_t j = 0; j < size; ++j) {
				std::cout << C[i * size + j] << ", ";
			}
			std::cout << std::endl;
		}
		std::cout << "----------------------" << std::endl;
		std::cout << "Unoptimized :" << std::endl;
		std::cout << "Time " << time_unopt << " ns" << std::endl;
		std::cout << "=> " << (FLOP / time_unopt) << "GFLOP/s"
			  << std::endl;
		std::cout << "======================" << std::endl;
	} else {
		std::cout << size << " " << time_unopt << " "
			  << (FLOP / time_unopt) << std::endl;
	}

	free(A);
	free(B);
	free(C);

	return 0;
}

void mmul_unoptimized(double const *A, double const *B, double *C, size_t dim) {
	/*
	 * Naive matrix multiply for square matrices (NxN)
	 * A, B: input matrices
	 * C: output matrix (has to be initialized to zero)
	 * dim: number of rows (or columns in this case) of the matrix
	 */
	for (size_t i = 0; i < dim; ++i) {
		for (size_t j = 0; j < dim; ++j) {
			for (size_t k = 0; k < dim; ++k) {
				C[i * dim + j] +=
				    A[i * dim + k] * B[k * dim + j];
			}
		}
	}
}
