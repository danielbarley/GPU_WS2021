/**************************************************************************************************
 *
 *       Computer Engineering Group, Heidelberg University - GPU Computing Exercise 06
 *
 *                 Gruppe : TODO
 *
 *                   File : kernel.cu
 *
 *                Purpose : Reduction
 *
 **************************************************************************************************/

#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

//
// Reduction_Kernel
//
__global__ void
reduction_Kernel(int numElements, float* dataIn, float* dataOut)
{
	extern __shared__ float sharedData[];

	int threadId = threadIdx.x;
	int const blockSize = blockDim.x;
	int elementId = blockIdx.x * blockDim.x + threadIdx.x;

	if (elementId < numElements)
	{
		sharedData[threadId] = dataIn[elementId];
	}
	__syncthreads();

	for (int i = blockSize / 2; i > 0; i /= 2)
	{
		if (threadId < i)
		{
			sharedData[threadId] += sharedData[threadId];
		}
		__syncthreads();
	}

	if (threadId == 0)
	{
		dataOut[blockIdx.x] = sharedData[0];
	}
}

__global__ void
reduction_Kernel2(int numElements, float* dataIn, float* dataOut)
{
	extern __shared__ float sharedData[];

	int threadId = threadIdx.x;
	int const blockSize = blockDim.x;
	int elementId = blockIdx.x * blockSize * 2 + threadIdx.x;


	sharedData[threadId] = dataIn[elementId] + dataIn[elementId + blockSize];
	__syncthreads();

	if (blockSize >= 1024) {
		if (threadId < 512) { sharedData[threadId] += sharedData[threadId + 512]; } __syncthreads();
	}
	if (blockSize >= 512) {
		if (threadId < 256) { sharedData[threadId] += sharedData[threadId + 256]; } __syncthreads();
	}
	if (blockSize >= 256) {
		if (threadId < 128) { sharedData[threadId] += sharedData[threadId + 128]; } __syncthreads();
	}
	if (blockSize >= 128) {
		if (threadId < 64) { sharedData[threadId] += sharedData[threadId + 64]; } __syncthreads();
	}

	float sum = 0;
	if (threadId < 32) sum = sharedData[threadId];

	if (threadId < 32 && blockSize >= 64) 
	{
		sum += sharedData[threadId + 32];
		sharedData[threadId] = sum; __syncwarp();
	}
	if (threadId < 16 && blockSize >= 32) 
	{
		sum += sharedData[threadId + 16]; __syncwarp();
		sharedData[threadId] = sum; __syncwarp();
	}
	if (threadId < 8 && blockSize >= 16) 
	{
		sum += sharedData[threadId + 8]; __syncwarp();
		sharedData[threadId] = sum; __syncwarp();
	}
	if (threadId < 4 && blockSize >= 8) 
	{
		sum += sharedData[threadId + 4]; __syncwarp();
		sharedData[threadId] = sum; __syncwarp();
	}
	if (threadId < 2 && blockSize >= 4) 
	{
		sum += sharedData[threadId + 2]; __syncwarp();
		sharedData[threadId] = sum; __syncwarp();
	}
	if (threadId < 1 && blockSize >= 2) 
	{
		sum += sharedData[threadId + 1]; __syncwarp();
		sharedData[threadId] = sum; __syncwarp();
	}

	if (threadId == 0)
	{
		dataOut[blockIdx.x] = sharedData[0];
	}
}

void reduction_Kernel_Wrapper(dim3 gridSize, dim3 blockSize, int numElements, float* dataIn, float* dataOut) {
	dim3 newBlockSize(blockSize.x / 2);
	dim3 newGridSize(gridSize.x / 2);
	reduction_Kernel2<<< gridSize, newBlockSize, newBlockSize.x * sizeof(float)>>>(numElements, dataIn, dataIn);
	reduction_Kernel2<<< 1, newGridSize, newGridSize.x * sizeof(float)>>>(newGridSize.x, dataIn, dataOut);
}

//
// Reduction Kernel using CUDA Thrust
//

void thrust_reduction_Wrapper(int numElements, float* dataIn, float* dataOut) {
	thrust::device_ptr<float> in_ptr = thrust::device_pointer_cast(dataIn);
	thrust::device_ptr<float> out_ptr = thrust::device_pointer_cast(dataOut);
	
	*out_ptr = thrust::reduce(in_ptr, in_ptr + numElements, (float) 0., thrust::plus<float>());	
}
