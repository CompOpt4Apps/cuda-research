#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <omp.h>

#define SIZE 30000				//Length and width of inner grid in threads
#define DIM (SIZE + 2)			//Length and width of the entire grid in threads
#define GRID_SIZE 1500 			//Length and width of inner grid in blocks
#define BLOCK_SIZE 20 			//Length and width of block in threads
#define MEM_SIZE (sizeof(float) * DIM * DIM)
#define TIME_STEPS 1
#define PINNED 0

void fillGrid(float* grid);
__global__ void computeGrid(float* read, float* write);
void swapGrids(float** read, float** write);
void printGrid(float* grid, const char* name);

void computeGrid2(float* read, float* write) {
	//The result of a cell is the sum of its neighbors
	#pragma omp parallel for collapse(2)
	for (int y = 1; y < DIM - 1; y++) {
		for (int x = 1; x < DIM - 1; x++) {
			write[DIM * y + x] = read[DIM * (y - 1) + x] + read[DIM * (y + 1) + x] + read[DIM * y + x - 1] + read[DIM * y + x + 1];
		}
	}
}

float* getOMPResult() {
	//Disables dynamic teams to force the max number of threads to always be used
	omp_set_dynamic(0);

	//Sets the max number of threads to use for all parallel operations
	omp_set_num_threads(8);

	//Allocate memory for the read and write grid
	float* read = (float*) malloc(MEM_SIZE);
	float* write = (float*) malloc(MEM_SIZE);
	assert(read != NULL && write != NULL);

	//Fill in the read grid
	fillGrid(read);
	//Compute the write grid TIME_STEPS times
	//Finally, swap the grids 1 last time to get result into write grid
	for (int i = 0; i < TIME_STEPS; i++) {
		computeGrid2(read, write);
		swapGrids(&read, &write);
	}
	swapGrids(&read, &write);

	return write;
}

int main(void) {
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
		hostRead = (float*) malloc(MEM_SIZE);
		hostWrite = (float*) malloc(MEM_SIZE);
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
		computeGrid<<<gridDimensions, blockDimensions>>>(deviceRead, deviceWrite);
		cudaDeviceSynchronize();
		swapGrids(&deviceRead, &deviceWrite);
	}
	swapGrids(&deviceRead, &deviceWrite);

	//Copies over the result (now in deviceRead) from device to host
	assert(cudaMemcpy(hostWrite, deviceWrite, MEM_SIZE, cudaMemcpyDeviceToHost) == cudaSuccess);
	assert(memcmp(hostWrite, getOMPResult(), MEM_SIZE) == 0);

	//Print out state of the grid
	//printGrid(hostWrite, "Result");

	//Frees both the host and device grids
	if (PINNED) {
		cudaFreeHost(hostRead);
		cudaFreeHost(hostWrite);
	}
	else {
		free(hostRead);
		free(hostWrite);
	}
	cudaFree(deviceRead);
	cudaFree(deviceWrite);

	return 0;
}

//Fills in the grid based on a thread's position in the grid
void fillGrid(float* grid) {
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

//Performs a parallelized stencil computation
__global__ void computeGrid(float* read, float* write) {
	//Retrieve the thread's position in the grid
	//The position is offset by 1 in the x and y directions to remove boundary checks
	int x = blockDim.x * blockIdx.x + threadIdx.x + 1;
	int y = blockDim.y * blockIdx.y + threadIdx.y + 1;

	//Writes the sum of the neighbors to the cell
	write[DIM * y + x] = read[DIM * (y - 1) + x] + read[DIM * (y + 1) + x] + read[DIM * y + x - 1] + read[DIM * y + x + 1];
}

//Swaps the pointers of two grids
void swapGrids(float** read, float** write) {
	float* temp = *read;
	*read = *write;
	*write = temp;
}

//Prints out the state of the internal grid (can also print entire grid)
void printGrid(float* grid, const char* name) {
	//Prints the name of the grid
	printf("<<< %s >>>\n\n", name);

	//Prints the inner grid
	for (int y = 1; y < DIM - 1; y++) {
		for (int x = 1; x < DIM - 1; x++) {
			printf("%-25.3f", grid[DIM * y + x]);
		}

		printf("\n");
	}
}