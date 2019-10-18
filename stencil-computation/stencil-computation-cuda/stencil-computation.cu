#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>

constexpr auto THREAD_DIM = 18000; 				//Square dimensions of the ENTIRE grid, including inner grid, whose dimension is 2 less
constexpr auto GRID_DIM = 10;					//Square dimensions of the grid in blocks
#define BLOCK_DIM (THREAD_DIM / GRID_DIM)		//Square dimensions of a single block, which only works if THREAD_DIM % GRID_DIM == 0

//Prototypes
__global__ void fillGrid(int* grid);
__global__ void computeGrid(int* grid, int* result);
void printGrid(int* grid, const char* name);

static int createOutput = 0;

int main() {
	//Check for errors in grid size definitions
	if (THREAD_DIM % GRID_DIM != 0) {
		printf("Error: bad grid size definitions.\n");
		return 1;
	}

	//Allocate memory for the grid (1D array) and the result grid
	int* grid;
	int* result;
	cudaMallocManaged(&grid, THREAD_DIM * THREAD_DIM * sizeof(int));
	cudaMallocManaged(&result, THREAD_DIM * THREAD_DIM * sizeof(int));
	if (grid == NULL || result == NULL) {
		printf("Error: memory allocation failed.\n");
		return 1;
	}

	//Sets up the CUDA events to handle timing
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float fillTime = 0;
	float computeTime = 0;

	//Fill in the grid, keeping time
	cudaEventRecord(start);
	fillGrid<<<dim3(GRID_DIM, GRID_DIM), dim3(BLOCK_DIM, BLOCK_DIM)>>>(grid);
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&fillTime, start, stop);

	//Compute the grid, keeping time
	cudaEventRecord(start);
	computeGrid<<<dim3(GRID_DIM, GRID_DIM), dim3(BLOCK_DIM, BLOCK_DIM)>>>(grid, result);
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&computeTime, start, stop);

	//Destroys the timing events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//Prints the elapsed time for both operations
	printf("Fill time: %.0fus\n", roundf(fillTime * 1000));
	printf("Compute time: %.0fus\n", roundf(computeTime * 1000));

	//If createOutput is 1, the grids will be printed out
	if (createOutput) {
		//Print out state of the filled and result grids
		printGrid(grid, "Filled");
		printf("\n");
		printGrid(result, "Result");
	}

	//Frees the memory used by the grids
	cudaFree(grid);
	cudaFree(result);

	return 0;
}

//Fills in the grid based on a thread's position in the grid
__global__ void fillGrid(int* grid) {
	//Retrieve the thread's position in the grid
	//x is the column, y is the row
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	//If the grid point is along the edges, fill in 0
	//Otherwise, fill in with value dependent on x and y values
	*(grid + THREAD_DIM * y + x) = x == 0 || y == 0 || x == THREAD_DIM - 1 || y == THREAD_DIM - 1 ? 0 : x * y;
}

//Computes the result grid by adding all neighbors
__global__ void computeGrid(int* grid, int* result) {
	//Retrieve the thread's position in the grid
	//x is the column, y is the row
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	//If along the edges of the grid, don't add neighbors
	if (x == 0 || y == 0 || x == THREAD_DIM - 1 || y == THREAD_DIM - 1) {
		return;
	}

	//Put the sum of the neighbors in the result
	*(result + THREAD_DIM * y + x) = *(grid + THREAD_DIM * (y - 1) + x) + *(grid + THREAD_DIM * (y + 1) + x) + *(grid + THREAD_DIM * y + x - 1) + *(grid + THREAD_DIM * y + x + 1);
}

//Prints out the state of the internal grid (can also print entire grid)
void printGrid(int* grid, const char* name) {
	printf("<<< %s >>>\n\n", name);

	for (int y = 1; y < THREAD_DIM - 1; y++) {
		for (int x = 1; x < THREAD_DIM - 1; x++) {
			printf("%-10d", *(grid + THREAD_DIM * y + x));
		}

		printf("\n");
	}
}