#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define SIZE 100

//Prototypes
__global__ void addVectors(int* A, int* B, int* result);

int main() {
	//Allocates memory for the vectors
	int* A;
	int* B;
	int* result;
	cudaMallocManaged(&A, SIZE * sizeof(int));
	cudaMallocManaged(&B, SIZE * sizeof(int));
	cudaMallocManaged(&result, SIZE * sizeof(int));

	//Initializes the vectors
	for (int i = 0; i < SIZE; i++) {
		*(A + i) = i;
		*(B + i) = i * i;
	}

	//Adds the vectors together
	addVectors<<<1, SIZE>>>(A, B, result);
	cudaDeviceSynchronize();

	//Print the result
	printf("Resulting vector: [");
	for (int i = 0; i < SIZE; i++) {
		printf("%s%d", i == 0 ? "" : ", ", *(result + i));
	}
	printf("]\n");

	//Frees the vectors
	cudaFree(A);
	cudaFree(B);
	cudaFree(result);

	return 0;
}

//Adds two vectors together
__global__ void addVectors(int* A, int* B, int* result) {
	*(result + threadIdx.x) = *(A + threadIdx.x) + *(B + threadIdx.x);
}