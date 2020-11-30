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
	const size_t blocks = gridDim.x;
	const size_t threads = gridDim.x * blockDim.x;
	const size_t threadsPerBlock = blockDim.x;
	const size_t elementsPerThread = (n + threads - 1) / threads;
	//const size_t elementsPerBlock = (n + blocks - 1) / blocks;

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
	const size_t blocks = gridDim.x;
	const size_t threads = gridDim.x * blockDim.x;
	const size_t threadsPerBlock = blockDim.x;
	const size_t elementsPerThread = (n + threads - 1) / threads;
	//const size_t elementsPerBlock = (n + blocks - 1) / blocks;

	int startIndex = (threadIdx.x + blockIdx.x * blockDim.x) * elementsPerThread;
	for (int i = 0; i < elementsPerThread; i += threadsPerBlock)
	{
		const size_t index = startIndex + i;
		if (i < n)
			data[index] = shm[index];
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
	const size_t blocks = gridDim.x;
	const size_t threads = gridDim.x * blockDim.x;
	const size_t threadsPerBlock = blockDim.x;
	const size_t elementsPerThread = (n + threads - 1) / threads;
	//const size_t elementsPerBlock = (n + blocks - 1) / blocks;

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
	const size_t blocks = gridDim.x;
	const size_t threads = gridDim.x * blockDim.x;
	const size_t threadsPerBlock = blockDim.x;
	const size_t elementsPerThread = (n + threads - 1) / threads;
	//const size_t elementsPerBlock = (n + blocks - 1) / blocks;

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
//(/*TODO Parameters*/)
( )
{
	/*TODO Kernel Code*/
}

void bankConflictsRead_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize /* TODO Parameters*/) {
	//globalMem2SharedMem<<< gridSize, blockSize, shmSize >>>( /* TODO Parameters */);
}
