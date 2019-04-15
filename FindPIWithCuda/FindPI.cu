// System includes
#include <stdio.h>
#include <math.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#include "Common.h"

#define MAX_THREADS_PER_BLOCK 1024

__global__ void preparePoints(const TPointData *input, TPointData *output, int count) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < count) {
		auto supportPoint = input[i];
		auto nextSupportPoint = input[(i + 1) % count];
		double nextPointX = (supportPoint.X + nextSupportPoint.X) / 2;
		double nextPointY = (supportPoint.Y + nextSupportPoint.Y) / 2;
		double radiusToNextPoint = sqrt(nextPointX * nextPointX + nextPointY * nextPointY);
		TPointData nextPoint;
		nextPoint.X = nextPointX / radiusToNextPoint;
		nextPoint.Y = nextPointY / radiusToNextPoint;
		output[i * 2] = supportPoint;
		output[(i * 2) + 1] = nextPoint;
	}
}

__global__ void calcLength(const TPointData *input, int count, double* accumulator) {
	__shared__ double local_lengths[MAX_THREADS_PER_BLOCK];
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < count)
	{
		auto prevPoint = input[i];
		auto point = input[(i + 1) % count];
		double diffX = point.X - prevPoint.X;
		double diffY = point.Y - prevPoint.Y;
		double length = sqrt(diffX * diffX + diffY * diffY);
		local_lengths[threadIdx.x] = length;
	}

	__syncthreads();

	if (threadIdx.x == 0) {
		double local_sum = 0;
		for (int j = i, int local_index = 0; j < count && local_index < blockDim.x; j++, local_index++) {
			local_sum += local_lengths[local_index];
		}
		accumulator[blockIdx.x] = local_sum;
	}
}

/**
 * Program main
 */
int main(int argc, char **argv) {
    printf("[Finding PI Using CUDA] - Starting...\n");

	int i_count = 4;
	size_t i_size = i_count * sizeof(TPointData);
	TPointData *h_V = (TPointData *)malloc(i_size);

	h_V[0].X = 1;
	h_V[0].Y = 0;
	h_V[1].X = 0;
	h_V[1].Y = -1;
	h_V[2].X = -1;
	h_V[2].Y = 0;
	h_V[3].X = 0;
	h_V[3].Y = 1;

	// Allocate the device input vector
	TPointData *d_iV = NULL;
	checkCudaErrors(cudaMalloc((void **)&d_iV, i_size));

	// Copy the host input vectors in host memory to the device input vectors in
	// device memory
	printf("Copy input data from the host memory to the CUDA device\n");
	checkCudaErrors(cudaMemcpy(d_iV, h_V, i_size, cudaMemcpyHostToDevice));


	for (int i = 0; i < 22; i++) {

		TPointData * d_oV = NULL;
		checkCudaErrors(cudaMalloc((void **)&d_oV, i_size  * 2));

		int threadsPerBlock = i_count < MAX_THREADS_PER_BLOCK ? i_count : MAX_THREADS_PER_BLOCK;
		int blocksPerGrid = (i_count + threadsPerBlock - 1) / threadsPerBlock;
		printf("CUDA kernel launch with %d blocks of %d threads for %d items\n", blocksPerGrid, threadsPerBlock, i_count);

		preparePoints<<<blocksPerGrid, threadsPerBlock>>>(d_iV, d_oV, i_count);

		cudaDeviceSynchronize();

		checkCudaErrors(cudaGetLastError());

		checkCudaErrors(cudaFree(d_iV));

		d_iV = d_oV;

		i_count = i_count * 2;
		i_size = i_size * 2;

		threadsPerBlock = i_count < MAX_THREADS_PER_BLOCK ? i_count : MAX_THREADS_PER_BLOCK;
		blocksPerGrid = (i_count + threadsPerBlock - 1) / threadsPerBlock;
		printf("CUDA kernel launch with %d blocks of %d threads for %d items\n", blocksPerGrid, threadsPerBlock, i_count);


		double result = 0;
		double *h_accumelator = NULL;

		h_accumelator = (double *)malloc(sizeof(double) * blocksPerGrid);

		double *d_accumulator = NULL;
		checkCudaErrors(cudaMalloc((void **)&d_accumulator, sizeof(double) * blocksPerGrid));
		checkCudaErrors(cudaMemset(d_accumulator, 0, sizeof(double) * blocksPerGrid));

		calcLength<<<blocksPerGrid, threadsPerBlock>>>(d_iV, i_count, d_accumulator);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());

		checkCudaErrors(cudaMemcpy(h_accumelator, d_accumulator, sizeof(double) * blocksPerGrid, cudaMemcpyDeviceToHost));

		for (int j = 0; j < blocksPerGrid; j++) {
			result += h_accumelator[j];
		}

		checkCudaErrors(cudaFree(d_accumulator));
		free(h_accumelator);

		fprintf(stdout, "Expected Pi:	%1.16f\n", 3.1415926535897931);
		fprintf(stdout, "Calculated Pi:	%1.16f\n", result / 2);
	}

	// Free device global memory

	checkCudaErrors(cudaFree(d_iV));

	free(h_V);
}

