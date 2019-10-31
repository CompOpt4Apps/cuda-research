#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

#define SIZE 30000				//Length and width of inner grid in cells
#define DIM (SIZE + 2)			//Length and width of the entire grid
#define MEM_SIZE (sizeof(float) * DIM * DIM)
#define TIME_STEPS 10
#define NUM_THREADS 14

void fillGrid(float* grid);
void computeGrid(float* read, float* write);
void swapGrids(float** read, float** write);
void printGrid(float* grid, const char* name);

int main(void) {
	//Disables dynamic teams to force the max number of threads to always be used
	omp_set_dynamic(0);

	//Sets the max number of threads to use for all parallel operations
	omp_set_num_threads(NUM_THREADS);

	//Allocate memory for the read and write grid
	float* read = malloc(MEM_SIZE);
	float* write = malloc(MEM_SIZE);
	assert(read != NULL && write != NULL);

	//Fill in the read grid
	fillGrid(read);

	//Compute the write grid TIME_STEPS times
	//Finally, swap the grids 1 last time to get result into write grid
	for (int i = 0; i < TIME_STEPS; i++) {
		computeGrid(read, write);
		swapGrids(&read, &write);
	}
	swapGrids(&read, &write);


	//Print out state of write grid
	//printGrid(write, "Result");

	//Frees the memory used by the grids
	free(read);
	free(write);

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

	//Fills in each spot of the inner grid with a 1.1
	for (int y = 1; y < DIM - 1; y++) {
		for (int x = 1; x < DIM - 1; x++) {
			grid[DIM * y + x] = 1.1;
		}
	}
}

//Computes the write grid by adding all neighboring cells
void computeGrid(float* read, float* write) {
	//The result of a cell is the sum of its neighbors
	#pragma omp parallel for collapse(2)
	for (int y = 1; y < DIM - 1; y++) {
		for (int x = 1; x < DIM - 1; x++) {
			write[DIM * y + x] = read[DIM * (y - 1) + x] + read[DIM * (y + 1) + x] + read[DIM * y + x - 1] + read[DIM * y + x + 1];
		}
	}
}

//Swaps the pointers of two grids
void swapGrids(float** read, float** write) {
	float* temp = *read;
	*read = *write;
	*write = temp;
}

//prints out the state of the internal grid (can also print entire grid)
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