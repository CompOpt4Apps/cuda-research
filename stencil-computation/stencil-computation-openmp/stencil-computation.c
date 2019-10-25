#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define SIZE 12
#define MEM_SIZE (sizeof(int) * SIZE * SIZE)
#define TIME_STEPS 1000000

void fillGrid(int* grid);
void computeGrid(int* grid, int* result);
void swapGrids(int** read, int** write);
void printGrid(int* grid, int skip, char* name);

int main() {
	//Allocates what would be "host" memory
	int* hostRead = malloc(MEM_SIZE);
	int* hostWrite = malloc(MEM_SIZE);

	//Fill in the grid
	fillGrid(hostRead);

	//Allocates what would be "device" memory, copies input to deviceRead
	int* deviceRead = malloc(MEM_SIZE);
	int* deviceWrite = malloc(MEM_SIZE);
	memcpy(deviceRead, hostRead, MEM_SIZE);

	//Computes the grid, swapping read and write grids each time
	for (int i = 0; i < TIME_STEPS; i++) {
		computeGrid(deviceRead, deviceWrite);
		swapGrids(&deviceRead, &deviceWrite);
	}

	//Copies over the result (now in deviceRead) from device to host
	memcpy(hostWrite, deviceRead, MEM_SIZE);
	printGrid(hostWrite, 1, "Result");

	//Frees the memory used by the grids
	free(hostRead);
	free(hostWrite);
	free(deviceRead);
	free(deviceWrite);

	return 0;
}

//Fills in the grid based on a thread's position in the grid
void fillGrid(int* grid) {
	//Fills in the leftmost and rightmost columns
	#pragma omp parallel for
	for (int y = 0; y < SIZE; y++) {
		grid[SIZE * y] = grid[SIZE * y + SIZE - 1] = 0;
	}

	//Fills in the top and bottom rows
	#pragma omp parallel for
	for (int x = 0; x < SIZE; x++) {
		grid[x] = grid[SIZE * (SIZE - 1) + x] = 0;
	}

	//Fills in each spot of the grid with a 1
	#pragma omp parallel for collapse(2)
	for (int y = 1; y < SIZE - 1; y++) {
		for (int x = 1; x < SIZE - 1; x++) {
			grid[SIZE * y + x] = 1;
		}
	}
}

//Computes the result grid by adding all neighbors of a cell and storing the sum in the cell
void computeGrid(int* read, int* write) {
	#pragma omp parallel for collapse(2)
	for (int y = 1; y < SIZE - 1; y++) {
		for (int x = 1; x < SIZE - 1; x++) {
			write[SIZE * y + x] = read[SIZE * (y - 1) + x] + read[SIZE * (y + 1) + x] + read[SIZE * y + x - 1] + read[SIZE * y + x + 1];
		}
	}
}

//Swaps the pointers of two grids
void swapGrids(int** read, int** write) {
	int* temp = *read;
	*read = *write;
	*write = temp;
}

//Prints out the state of the internal grid (can also print entire grid)
void printGrid(int* grid, int skip, char* name) {
	printf("<<< %s >>>\n\n", name);

	for (int y = skip; y < SIZE - skip; y++) {
		for (int x = skip; x < SIZE - skip; x++) {
			printf("%-15d", grid[SIZE * y + x]);
		}

		printf("\n");
	}
}