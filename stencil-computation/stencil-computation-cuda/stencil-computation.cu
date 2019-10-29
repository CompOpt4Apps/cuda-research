#include <stdlib.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define THREAD_DIM 12
#define GRID_DIM 2
#define BLOCK_DIM ((THREAD_DIM - 2) / GRID_DIM)
#define MEM_SIZE (sizeof(int) * THREAD_DIM * THREAD_DIM)
#define TIME_STEPS 1000000

void fillGrid(int* grid);
__global__ void computeGrid(int* read, int* write);
void swapGrids(int** read, int** write);
void printGrid(int* grid, int skip, const char* name);

int main() {
	//Checks for a valid grid dimension
	if ((THREAD_DIM - 2) % GRID_DIM != 0) {
		printf("Error: bad grid/thread dimensions.\n");
		return 1;
	}

	//Allocates space on the host for the grids in pinned memory
	//Using	memory is much faster, only being limited by the speed of the PCI-E bus
	int* hostRead;
	int* hostWrite;
	cudaMallocHost((void**) &hostRead, MEM_SIZE);
	cudaMallocHost((void**) &hostWrite, MEM_SIZE);

	//Fills in the read grid serially
	fillGrid(hostRead);

	//Allocates space on the device for the grids, and copies over the input grid
	int* deviceRead;
	int* deviceWrite;
	cudaMalloc(&deviceRead, MEM_SIZE);
	cudaMalloc(&deviceWrite, MEM_SIZE);
	cudaMemcpy(deviceRead, hostRead, MEM_SIZE, cudaMemcpyHostToDevice);

	//Calls the computeGrid kernel TIME_STEPS times, swapping input and output each time
	dim3 gridDimension(GRID_DIM, GRID_DIM);
	dim3 blockDimension(BLOCK_DIM, BLOCK_DIM);
	for (int i = 0; i < TIME_STEPS; i++) {
		computeGrid<<<gridDimension, blockDimension>>>(deviceRead, deviceWrite);
		cudaDeviceSynchronize();
		swapGrids(&deviceRead, &deviceWrite);
	}

	//Copies over the result (now in deviceRead) from device to host
	cudaMemcpy(hostWrite, deviceRead, MEM_SIZE, cudaMemcpyDeviceToHost);
	printGrid(hostWrite, 1, "Result");

	//Frees both the host and device grids
	cudaFreeHost(hostRead);
	cudaFreeHost(hostWrite);
	cudaFree(deviceRead);
	cudaFree(deviceWrite);

	return 0;
}

//Fills a grid with a border of zeroes and inner portion with x * y
void fillGrid(int* grid) {
	//Fills in the leftmost and rightmost columns
	for (int y = 0; y < THREAD_DIM; y++) {
		grid[THREAD_DIM * y] = grid[THREAD_DIM * y + THREAD_DIM - 1] = 0;
	}

	//Fills in the top and bottom rows
	for (int x = 0; x < THREAD_DIM; x++) {
		grid[x] = grid[THREAD_DIM * (THREAD_DIM - 1) + x] = 0;
	}

	//Fills in each inner cell in the grid with a 1
	for (int y = 1; y < THREAD_DIM - 1; y++) {
		for (int x = 1; x < THREAD_DIM - 1; x++) {
			grid[THREAD_DIM * y + x] = 1;
		}
	}
}

//Performs a parallelized stencil computation
__global__ void computeGrid(int* read, int* write) {
	//Retrieve the thread's position in the grid
	int x = blockDim.x * blockIdx.x + threadIdx.x + 1;
	int y = blockDim.y * blockIdx.y + threadIdx.y + 1;

	//Writes the sum of the neighbors to the cell
	write[THREAD_DIM * y + x] = read[THREAD_DIM * (y - 1) + x] + read[THREAD_DIM * (y + 1) + x] + read[THREAD_DIM * y + x - 1] + read[THREAD_DIM * y + x + 1];
}

//Swaps the pointers of two grids
void swapGrids(int** read, int** write) {
	int* temp = *read;
	*read = *write;
	*write = temp;
}

//Prints out the state of the internal grid (can also print entire grid)
void printGrid(int* grid, int skip, const char* name) {
	printf("<<< %s >>>\n\n", name);

	for (int y = skip; y < THREAD_DIM - skip; y++) {
		for (int x = skip; x < THREAD_DIM - skip; x++) {
			printf("%-15d", grid[THREAD_DIM * y + x]);
		}

		printf("\n");
	}
}