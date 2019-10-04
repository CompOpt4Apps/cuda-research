#include <stdio.h>
#include "cuda_runtime.h"

__global__ void greeting() {
	printf("Hello world!\n");
}

int main() {
	greeting<<<1, 1>>>();
	return 0;
}