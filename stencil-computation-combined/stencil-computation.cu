#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <omp.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//Prototypes
static float* executeSerialImplementation(void);
static void computeSerialGrid(float* read, float* write);
static float* executeOpenMPImplementation(void);
static void computeOpenMPGrid(float* read, float* write);
static float* executeCUDAImplementation(void);
static __global__ void computeCUDAGrid(float* read, float* write);
static void fillGrid(float* grid);
static void swapGrids(float** read, float** write);
static void verify(float* a, float* b);

//Control variables
const int SIZE = 30000;									//Length and width of inner grid in threads
const int DIM = SIZE + 2;								//Length and width of the entire grid in threads
const int GRID_SIZE = 1500;								//Length and width of inner grid in blocks
const int BLOCK_SIZE = 20;								//Length and width of block in threads
const size_t MEM_SIZE = sizeof(float) * DIM * DIM;		//Amount of memory used by a single grid
const int TIME_STEPS = 10;								//Number of time steps to perform
const int PINNED = 0;									//Controls using pinned and unpinned CUDA host memory
const int NUM_THREADS = 16;								//Controls the number of threads used by OpenMP

//Performs the execution of each implementation and the comparison of each result
int main(void) {
	//Executes each implementation and gets each result
	//float* serial = executeSerialImplementation();
	//float* omp = executeOpenMPImplementation();
	executeCUDAImplementation();

	//Compares the results of each implementation
	//Transitively, if a == b and b == c, we already know a == c
	//verify(serial, omp);
	//verify(omp, cuda);

	//Frees the write grids after verification
	//free(serial);
	//free(omp);
	//if (PINNED) {
	//	cudaFreeHost(cuda);
	//}
	//else {
	//	free(cuda);
	//}

	return 0;
}

//Executes the serial implementation of the stencil computation
static float* executeSerialImplementation(void) {
	//Allocate memory for the read and write grid
	float* read = (float*) malloc(MEM_SIZE);
	float* write = (float*) malloc(MEM_SIZE);
	assert(read != NULL && write != NULL);

	//Fill in the read grid
	fillGrid(read);

	//These variables keep track of the amount of time it takes for the grid computation to take
	double serialComputeTime = 0;
	double start;
	double end;

	//Compute the write grid TIME_STEPS times
	//Finally, swap the grids 1 last time to get result into write grid
	for (int i = 0; i < TIME_STEPS; i++) {
		start = omp_get_wtime();
		computeSerialGrid(read, write);
		end = omp_get_wtime();

		serialComputeTime += end - start;
		swapGrids(&read, &write);
	}
	swapGrids(&read, &write);

	//Prints out the timing information
	printf("Total serial compute time: %.5lf seconds\n", serialComputeTime);

	//Free all but the write grid
	free(read);
	
	return write;
}

//Computes the write grid for the serial implementation
static void computeSerialGrid(float* read, float* write) {
	//The result of a cell is the sum of its neighbors
	for (int y = 1; y < DIM - 1; y++) {
		for (int x = 1; x < DIM - 1; x++) {
			write[DIM * y + x] = read[DIM * (y - 1) + x] + read[DIM * (y + 1) + x] + read[DIM * y + x - 1] + read[DIM * y + x + 1];
		}
	}
}

//Executes the OpenMP implementation of the stencil computation
static float* executeOpenMPImplementation(void) {
	//Disables dynamic teams to force the max number of threads to always be used
	omp_set_dynamic(0);

	//Sets the max number of threads to use for all parallel operations
	omp_set_num_threads(NUM_THREADS);

	//Allocate memory for the read and write grids
	float* read = (float*) malloc(MEM_SIZE);
	float* write = (float*) malloc(MEM_SIZE);
	assert(read != NULL && write != NULL);

	//Fill in the read grid
	fillGrid(read);

	//These variables keep track of the amount of time it takes for the grid computation to take
	double ompComputeTime = 0;
	double start;
	double end;

	//Compute the write grid TIME_STEPS times
	//Finally, swap the grids 1 last time to get result into write grid
	for (int i = 0; i < TIME_STEPS; i++) {
		start = omp_get_wtime();
		computeOpenMPGrid(read, write);
		end = omp_get_wtime();

		ompComputeTime += end - start;
		swapGrids(&read, &write);
	}
	swapGrids(&read, &write);

	//Prints out the timing information
	printf("Total OpenMP compute time: %.5lf seconds\n", ompComputeTime);

	//Frees all but the write grid
	free(read);
	
	return write;
}

//Computes the write grid of the OpenMP implementation
static void computeOpenMPGrid(float* read, float* write) {
	//The result of a cell is the sum of its neighbors
	#pragma omp parallel for collapse(2)
	for (int y = 1; y < DIM - 1; y++) {
		for (int x = 1; x < DIM - 1; x++) {
			write[DIM * y + x] = read[DIM * (y - 1) + x] + read[DIM * (y + 1) + x] + read[DIM * y + x - 1] + read[DIM * y + x + 1];
		}
	}
}

//Performs the CUDA implementation of the stencil computation
static float* executeCUDAImplementation(void) {
	//The total number of threads must match the amount specified by the GRID_SIZE and BLOCK_SIZE
	assert(GRID_SIZE * BLOCK_SIZE == SIZE);

	//Allocates space on the host for the grids in pinned or paged memory, depending on PINNED
	//Using	pinned memory is much faster, only being limited by the speed of the PCI-E bus
	//Pinned memory, however, takes longer to allocate
	float* hostRead;
	float* hostWrite;
	if (PINNED) {
		assert(cudaMallocHost((void**) &hostRead, MEM_SIZE) == cudaSuccess);
		assert(cudaMallocHost((void**) &hostWrite, MEM_SIZE) == cudaSuccess);
	}
	else {
		double start = omp_get_wtime();
		hostRead = (float*) malloc(MEM_SIZE);
		hostWrite = (float*) malloc(MEM_SIZE);
		double end = omp_get_wtime();
		printf("Total CUDA host allocation time: %.5lf seconds\n", end - start);
		assert(hostRead != NULL && hostWrite != NULL);
	}

	//Fills in the read grid serially
	fillGrid(hostRead);

	//Allocates space on the device for the grids, and copies over the input grid
	float* deviceRead;
	float* deviceWrite;
	assert(cudaMalloc(&deviceRead, MEM_SIZE) == cudaSuccess);
	assert(cudaMalloc(&deviceWrite, MEM_SIZE) == cudaSuccess);
	assert(cudaMemcpy(deviceRead, hostRead, MEM_SIZE, cudaMemcpyHostToDevice) == cudaSuccess);

	//Calls the computeGrid kernel TIME_STEPS times, swapping input and output each time
	//Finally, swap the grids 1 last time to get the result into the deviceWrite grid
	dim3 gridDimensions(GRID_SIZE, GRID_SIZE);
	dim3 blockDimensions(BLOCK_SIZE, BLOCK_SIZE);
	for (int i = 0; i < TIME_STEPS; i++) {
		computeCUDAGrid<<<gridDimensions, blockDimensions>>>(deviceRead, deviceWrite);
		cudaDeviceSynchronize();
		swapGrids(&deviceRead, &deviceWrite);
	}
	swapGrids(&deviceRead, &deviceWrite);

	//Copies over the result (now in deviceWrite) from device to host
	assert(cudaMemcpy(hostWrite, deviceWrite, MEM_SIZE, cudaMemcpyDeviceToHost) == cudaSuccess);

	//Frees all but the host write grid
	if (PINNED) {
		cudaFreeHost(hostRead);
		cudaFreeHost(hostWrite);
	}
	else {
		double start = omp_get_wtime();
		free(hostRead);
		free(hostWrite);
		double end = omp_get_wtime();
		printf("Total CUDA host free time: %.5lf seconds\n", end - start);
	}
	cudaFree(deviceRead);
	cudaFree(deviceWrite);

	return NULL;
}

//Performs a parallelized stencil computation using CUDA cores
static __global__ void computeCUDAGrid(float* read, float* write) {
	//Retrieve the thread's position in the grid
	//The position is offset by 1 in the x and y directions to remove boundary checks
	int x = blockDim.x * blockIdx.x + threadIdx.x + 1;
	int y = blockDim.y * blockIdx.y + threadIdx.y + 1;

	//Writes the sum of the neighbors to the cell
	write[DIM * y + x] = read[DIM * (y - 1) + x] + read[DIM * (y + 1) + x] + read[DIM * y + x - 1] + read[DIM * y + x + 1];
}

//Fills in a grid serially with sample data
static void fillGrid(float* grid) {
	//Fills in the leftmost and rightmost columns
	for (int y = 0; y < DIM; y++) {
		grid[DIM * y] = grid[DIM * y + DIM - 1] = 0;
	}

	//Fills in the top and bottom rows
	for (int x = 0; x < DIM; x++) {
		grid[x] = grid[DIM * (DIM - 1) + x] = 0;
	}

	//Fills in each spot of the inner grid with 1.1
	for (int y = 1; y < DIM - 1; y++) {
		for (int x = 1; x < DIM - 1; x++) {
			grid[DIM * y + x] = 1.1;
		}
	}
}

//Swaps the pointers of two grids
static void swapGrids(float** read, float** write) {
	float* temp = *read;
	*read = *write;
	*write = temp;
}

//Verifies that two grids are equal to each other
static void verify(float* a, float* b) {
	assert(memcmp(a, b, MEM_SIZE) == 0);
}