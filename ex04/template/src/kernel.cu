/******************************************************************************
 *
 *Computer Engineering Group, Heidelberg University - GPU Computing Exercise 04
 *
 *                  Group : TBD
 *
 *                   File : kernel.cu
 *
 *                Purpose : Memory Operations Benchmark
 *
 ******************************************************************************/

//
// Test Kernel
//

__global__ void 
globalMem2SharedMem
(float* data, size_t n)
{
	extern __shared__ float shm[];
	// assume one dimensional blocks
	const size_t threads = gridDim.x * blockDim.x;
	const size_t threadsPerBlock = blockDim.x;
	const size_t elementsPerThread = (n + threads - 1) / threads;

	int startIndex = (threadIdx.x + blockIdx.x * blockDim.x) * elementsPerThread;
	for (int i = 0; i < elementsPerThread; i += threadsPerBlock)
	{
		const size_t index = startIndex + i;
		if (i < n)
			shm[index] = data[index];
	}
}

void globalMem2SharedMem_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize, float* data) {
	size_t n = shmSize / sizeof(float);
	globalMem2SharedMem<<< gridSize, blockSize, shmSize >>>(data, n);
}

// this version is slower because of bad caching behaviour
__global__ void 
globalMem2SharedMem_v2
(float* data, size_t n)
{
	extern __shared__ float shm[];
	// assume one dimensional blocks
	size_t threads = gridDim.x * blockDim.x;
	size_t elementsPerThread = (n + threads - 1) / threads;
	const size_t startIndex = (threadIdx.x + blockDim.x * blockIdx.x) * elementsPerThread;
		
	extern __shared__ float shm[]; // uses `shmSize`-Parameter for size automatically
	for(int i = 0; i < elementsPerThread; ++i){
		const size_t index = startIndex + i;
		if(startIndex < n){
			shm[index] = data[index];
		}
	}
}

void globalMem2SharedMem_Wrapper_v2(dim3 gridSize, dim3 blockSize, int shmSize, float* data) {
	size_t n = shmSize / sizeof(float);
	globalMem2SharedMem_v2<<< gridSize, blockSize, shmSize >>>(data, n);
}

__global__ void 
SharedMem2globalMem
(float* data, size_t n)
{
	extern __shared__ float shm[];
	// assume one dimensional blocks
	const size_t threads = gridDim.x * blockDim.x;
	const size_t threadsPerBlock = blockDim.x;
	const size_t elementsPerThread = (n + threads - 1) / threads;

	int startIndex = (threadIdx.x + blockIdx.x * blockDim.x) * elementsPerThread;
	for (int i = 0; i < elementsPerThread; i += threadsPerBlock)
	{
		const size_t index = startIndex + i;
		if (i < n)
			data[0] = shm[index];
	}
}
void SharedMem2globalMem_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize, float* data) {
	size_t n = shmSize / sizeof(float);
	SharedMem2globalMem<<< gridSize, blockSize, shmSize >>>(data, n);
}

__global__ void 
SharedMem2Registers
(float* data, size_t n)
{
	extern __shared__ float shm[];
	// assume one dimensional blocks
	const size_t threads = gridDim.x * blockDim.x;
	const size_t threadsPerBlock = blockDim.x;
	const size_t elementsPerThread = (n + threads - 1) / threads;

	int startIndex = (threadIdx.x + blockIdx.x * blockDim.x) * elementsPerThread;
	float reg;
	for (int i = 0; i < elementsPerThread; i += threadsPerBlock)
	{
		const size_t index = startIndex + i;
		if (i < n)
			reg = shm[index];
	}

	if (startIndex == 0)
		data[0] = reg;
}
void SharedMem2Registers_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize, float* data) {
	size_t n = shmSize / sizeof(float);
	SharedMem2Registers<<< gridSize, blockSize, shmSize >>>(data, n);
}

__global__ void 
Registers2SharedMem
(float* data, size_t n)
{
	extern __shared__ float shm[];
	// assume one dimensional blocks
	const size_t threads = gridDim.x * blockDim.x;
	const size_t threadsPerBlock = blockDim.x;
	const size_t elementsPerThread = (n + threads - 1) / threads;

	int startIndex = (threadIdx.x + blockIdx.x * blockDim.x) * elementsPerThread;
	float reg = 133.7;
	for (int i = 0; i < elementsPerThread; i += threadsPerBlock)
	{
		const size_t index = startIndex + i;
		if (i < n)
			shm[index] = reg;
	}

	if (startIndex == 0)
		data[0] = reg;
}
void Registers2SharedMem_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize, float* data) {
	size_t n = shmSize / sizeof(float);
	Registers2SharedMem<<< gridSize, blockSize, shmSize >>>(data, n);
}

__global__ void 
bankConflictsRead
(float* data, size_t n, size_t stride, long* d_clock)
{
	extern __shared__ float shm[];
	// assume one dimensional blocks
	size_t index = (threadIdx.x + blockIdx.x * blockDim.x) * stride;
	// "modulo"
	size_t q = index / n;
	size_t p = q * n;
	index -= p;
	float reg;
	int iterations = 100;
	long start = clock64();
	for (int i = 0; i < iterations; i++)
	{
		reg = shm[index];
	}

	(*d_clock) += clock64() / iterations - start / iterations;

	if (index == 0)
		data[0] = reg;
}

void bankConflictsRead_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize, float* data, size_t stride, long* d_clock) {
	size_t n = shmSize / sizeof(float);
	bankConflictsRead<<< gridSize, blockSize, shmSize >>>(data, n, stride, d_clock);
}
