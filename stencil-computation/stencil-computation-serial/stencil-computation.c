#include <stdio.h>
#include <stdlib.h>

#define SIZE 18000		//The size of the entire grid, including the outer edges

//Prototypes
void fillGrid(int* grid);
void computeGrid(int* grid, int* result);
void printGrid(int* grid, char* name);

int main() {
	//Allocate memory for the grid (1D array) and the result grid
	int* grid = malloc(SIZE * SIZE * sizeof(int));
	int* result = malloc(SIZE * SIZE * sizeof(int));

	//Fill in the grid, keeping time
	fillGrid(grid);

	//Compute the grid, keeping time
	computeGrid(grid, result);

	//Print out state of the filled and result grids
	printGrid(grid, "Filled");
	printf("\n");
	printGrid(result, "Result");		

	//Frees the memory used by the grids
	free(grid);
	free(result);

	return 0;
}

//Fills in the grid based on a thread's position in the grid
void fillGrid(int* grid) {
	//Fills in the top and bottom rows
	for (int x = 0; x < SIZE; x++) {
		*(grid + x) = 0;
		*(grid + SIZE * (SIZE - 1) + x) = 0;
	}

	//Fills in the leftmost and rightmost columns
	for (int y = 0; y < SIZE; y++) {
		*(grid + SIZE * y) = 0;
		*(grid + SIZE * y + SIZE - 1) = 0;
	}

	//Fills in each spot of the grid with a cells product of its x and y position
	for (int y = 1; y < SIZE - 1; y++) {
		for (int x = 1; x < SIZE - 1; x++) {
			*(grid + SIZE * y + x) = x * y;
		}
	}
}

//Computes the result grid by adding all neighbors
void computeGrid(int* grid, int* result) {
	for (int y = 1; y < SIZE - 1; y++) {
		for (int x = 1; x < SIZE - 1; x++) {
			//Gets each of the neighbor's values, add them together
			*(result + SIZE * y + x) = *(grid + SIZE * (y - 1) + x) + *(grid + SIZE * (y + 1) + x) + *(grid + SIZE * y + x - 1) + *(grid + SIZE * y + x + 1);
		}
	}
}

//Prints out the state of the internal grid (can also print entire grid)
void printGrid(int* grid, char* name) {
	printf("<<< %s >>>\n\n", name);

	for (int y = 1; y < SIZE - 1; y++) {
		for (int x = 1; x < SIZE - 1; x++) {
			printf("%-10d", *(grid + SIZE * y + x));
		}

		printf("\n");
	}
}