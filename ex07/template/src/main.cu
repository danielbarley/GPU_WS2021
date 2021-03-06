/******************************************************************************
 *
 *           XXXII Heidelberg Physics Graduate Days - GPU Computing
 *
 *                 Gruppe : TODO
 *
 *                   File : main.cu
 *
 *                Purpose : n-Body Computation
 *
 ******************************************************************************/

#include <cmath>
#include <ctime>
#include <iostream>
#include <cstdlib>
#include <chCommandLine.h>
#include <chTimer.hpp>
#include <cstdio>
#include <iomanip>

const static int DEFAULT_NUM_ELEMENTS   = 1024;
const static int DEFAULT_NUM_ITERATIONS =    5;
const static int DEFAULT_BLOCK_DIM      =  128;

#define TIMESTEP     1e-6; // s
#define GAMMA   6.673e-11; // (Nm^2)/(kg^2)

//
// Structures
//
// Use a SOA (Structure of Arrays)
//
struct Body_t {
	float4* posMass;  /* x = x */
	                  /* y = y */
	                  /* z = z */
	                  /* w = Mass */
	float3* velocity; /* x = v_x*/
	                  /* y = v_y */
	                  /* z= v_z */
	
	Body_t(): posMass(NULL), velocity(NULL) {}
	};

//
// Function Prototypes
//
void printHelp(char *);
void printElement(Body_t, int, int);

bool isEqualFloat3(float3 a, float3 b, float epsilon = 1e-2);
bool isEqualFloat4(float4 a, float4 b, float epsilon = 1e-2);

//
// Device Functions
//

//
// Calculate the Distance of two points
//
__device__ float
getDistance(float4 a, float4 b)
{
	float distance;
	distance = sqrtf(powf(b.x - a.x, 2) + powf(b.y - a.y, 2) + powf(b.z - a.z, 2));
	return distance;
}

//
// Calculate the forces between two bodies
//
__device__ void
bodyBodyInteraction(float4 bodyA, float4 bodyB, float3& force)
{
	float distance = getDistance(bodyA, bodyB);

	if (distance==0) 
		return;


	float tmpForce = -((bodyA.w*bodyB.w) / powf(distance, 2)) * GAMMA;

	force.x += ((bodyB.x - bodyA.x) / distance) * tmpForce;
	force.y += ((bodyB.y - bodyA.y) / distance) * tmpForce;
	force.z += ((bodyB.z - bodyA.z) / distance) * tmpForce;
}

//
// Calculate the new velocity of one particle
//
__device__ void
calculateSpeed(float mass, float3& currentSpeed, float3 force)
{
	currentSpeed.x += (force.x / mass) * TIMESTEP;
	currentSpeed.y += (force.y / mass) * TIMESTEP;
	currentSpeed.z += (force.z / mass) * TIMESTEP;
}

//
// n-Body Kernel for the speed calculation
//
__global__ void
simpleNbody_Kernel(int numElements, float4* bodyPos, float3* bodySpeed)
{
	int elementId = blockIdx.x * blockDim.x + threadIdx.x;

	float4 elementPosMass;
	float3 elementForce;
	float3 elementSpeed;
	
	if (elementId < numElements) {
		elementPosMass = bodyPos[elementId];
		elementSpeed = bodySpeed[elementId];
		elementForce = make_float3(0,0,0);

		for (int i = 0; i < numElements; i++) {
			if (i != elementId) {
				bodyBodyInteraction(elementPosMass, bodyPos[i], elementForce);
			}
		}

		calculateSpeed(elementPosMass.w, elementSpeed, elementForce);

		bodySpeed[elementId] = elementSpeed;
	}
}

__global__ void
streamNbody_Kernel(int numStoredElements, int numSwappedElements, float4* bodyPos, float3* bodySpeed)
{
	int elementId = blockIdx.x * blockDim.x + threadIdx.x;

	float4 elementPosMass;
	float3 elementForce;
	float3 elementSpeed;
	
	if (elementId < numStoredElements) {
		elementPosMass = bodyPos[elementId];
		elementSpeed = bodySpeed[elementId];
		elementForce = make_float3(0,0,0);

		for (int i = numStoredElements; i < numStoredElements + numSwappedElements; i++) {
			bodyBodyInteraction(elementPosMass, bodyPos[i], elementForce);
		}

		calculateSpeed(elementPosMass.w, elementSpeed, elementForce);

		bodySpeed[elementId] = elementSpeed;
	}
}

__global__ void
sharedNbody_Kernel(int numElements, float4* bodyPos, float3* bodySpeed)
{
	extern __shared__ float4 shPosMass[];

	int elementId = blockIdx.x*blockDim.x + threadIdx.x;
	if (elementId < numElements)
	{
		float4 elementPosMass = bodyPos[elementId]; 
		float3 elementSpeed = bodySpeed[elementId]; 
		float3 elementForce = make_float3(0,0,0);

		//#pragma unroll 8
		for (int j = 0; j < numElements; j += blockDim.x) {
			shPosMass[threadIdx.x] = bodyPos[j + threadIdx.x];
			__syncthreads();
			// #pragma unroll 32
			for (size_t k = 0; k < blockDim.x; k++) {
				bodyBodyInteraction(elementPosMass, shPosMass[k], elementForce);
			}
			__syncthreads();
		}
		calculateSpeed(elementPosMass.w, elementSpeed, elementForce);
		bodySpeed[elementId] = elementSpeed;
	}
}

//
// n-Body Kernel to update the position
// Neended to prevent write-after-read-hazards
//
__global__ void
updatePosition_Kernel(int numElements, float4* bodyPos, float3* bodySpeed)
{
	int elementId = blockIdx.x * blockDim.x + threadIdx.x;

	float4 elementPosMass;
	float3 elementSpeed;

	if (elementId < numElements) {
		elementPosMass = bodyPos[elementId];
		elementSpeed = bodySpeed[elementId];
		
		elementPosMass.x += elementSpeed.x * TIMESTEP;
		elementPosMass.y += elementSpeed.y * TIMESTEP;
		elementPosMass.z += elementSpeed.z * TIMESTEP;
		
		bodyPos[elementId] = elementPosMass;
	}
}

//
// Main
//
int
main(int argc, char * argv[])
{
	bool showHelp = chCommandLineGetBool("h", argc, argv);
	if (!showHelp) {
		showHelp = chCommandLineGetBool("help", argc, argv);
	}

	if (showHelp) {
		printHelp(argv[0]);
		exit(0);
	}

	std::cout << "***" << std::endl
			  << "*** Starting ..." << std::endl
			  << "***" << std::endl;

	ChTimer memCpyH2DTimer, memCpyD2HTimer;
	ChTimer kernelTimerNaive, kernelTimerOptimized;
	ChTimer kernelTimerLimited, kernelTimerLimitedStreams;

	//
	// Allocate Memory
	//
	int numElements = 0;
	chCommandLineGet<int>(&numElements, "s", argc, argv);
	chCommandLineGet<int>(&numElements, "size", argc, argv);
	numElements = numElements != 0 ?
			numElements : DEFAULT_NUM_ELEMENTS;
	//
	// Host Memory
	//
	bool pinnedMemory = true;
	if (!pinnedMemory) {
		pinnedMemory = chCommandLineGetBool("pinned-memory",argc,argv);
	}

	Body_t h_particles;
	if (!pinnedMemory) {
		// Pageable
		h_particles.posMass = static_cast<float4*>
				(malloc(static_cast<size_t>
				(numElements * sizeof(*(h_particles.posMass)))));
		h_particles.velocity = static_cast<float3*>
				(malloc(static_cast<size_t>
				(numElements * sizeof(*(h_particles.velocity)))));
	} else {
		// Pinned
		cudaMallocHost(&(h_particles.posMass), 
				static_cast<size_t>
				(numElements * sizeof(*(h_particles.posMass))));
		cudaMallocHost(&(h_particles.velocity), 
				static_cast<size_t>
				(numElements * sizeof(*(h_particles.velocity))));
	}

	// allocate memory for results
	Body_t h_particles_results_naive;
	h_particles_results_naive.posMass = static_cast<float4*>
				(malloc(static_cast<size_t>
				(numElements * sizeof(*(h_particles.posMass)))));
	h_particles_results_naive.velocity = static_cast<float3*>
				(malloc(static_cast<size_t>
				(numElements * sizeof(*(h_particles.velocity)))));
	Body_t h_particles_results_optimized;
	h_particles_results_optimized.posMass = static_cast<float4*>
				(malloc(static_cast<size_t>
				(numElements * sizeof(*(h_particles.posMass)))));
	h_particles_results_optimized.velocity = static_cast<float3*>
				(malloc(static_cast<size_t>
				(numElements * sizeof(*(h_particles.velocity)))));
	Body_t h_particles_results_stream;
	cudaMallocHost(&(h_particles_results_stream.posMass), 
				static_cast<size_t>
				(numElements * sizeof(*(h_particles.posMass))));
	cudaMallocHost(&(h_particles_results_stream.velocity), 
			static_cast<size_t>
			(numElements * sizeof(*(h_particles.velocity))));


	// Init Particles
//	srand(static_cast<unsigned>(time(0)));
	srand(0); // Always the same random numbers
	for (int i = 0; i < numElements; i++) {
		h_particles.posMass[i].x = 1e-8*static_cast<float>(rand()); // Modify the random values to
		h_particles.posMass[i].y = 1e-8*static_cast<float>(rand()); // increase the position changes
		h_particles.posMass[i].z = 1e-8*static_cast<float>(rand()); // and the velocity
		h_particles.posMass[i].w =  1e4*static_cast<float>(rand());
		h_particles.velocity[i].x = 0.0f;
		h_particles.velocity[i].y = 0.0f;
		h_particles.velocity[i].z = 0.0f;
	}
	
	printElement(h_particles, 0, 0);

	// Device Memory
	Body_t d_particles;
	cudaMalloc(&(d_particles.posMass), 
			static_cast<size_t>(numElements * sizeof(*(d_particles.posMass))));
	cudaMalloc(&(d_particles.velocity), 
			static_cast<size_t>(numElements * sizeof(*(d_particles.velocity))));

	if (h_particles.posMass == NULL || h_particles.velocity == NULL ||
		d_particles.posMass == NULL || d_particles.velocity == NULL) {
		std::cout << "\033[31m***" << std::endl
		          << "*** Error - Memory allocation failed" << std::endl
		          << "***\033[0m" << std::endl;

		exit(-1);
	}

	//
	// Copy Data to the Device
	//
	memCpyH2DTimer.start();

	cudaMemcpy(d_particles.posMass, h_particles.posMass, 
			static_cast<size_t>(numElements * sizeof(float4)), 
			cudaMemcpyHostToDevice);
	cudaMemcpy(d_particles.velocity, h_particles.velocity, 
			static_cast<size_t>(numElements * sizeof(float3)), 
			cudaMemcpyHostToDevice);

	memCpyH2DTimer.stop();

	//
	// Get Kernel Launch Parameters
	//
	int blockSize = 0,
	    gridSize = 0,
	    numIterations = 0;

	// Number of Iterations	
	chCommandLineGet<int>(&numIterations,"i", argc, argv);
	chCommandLineGet<int>(&numIterations,"num-iterations", argc, argv);
	numIterations = numIterations != 0 ? numIterations : DEFAULT_NUM_ITERATIONS;

	// Block Dimension / Threads per Block
	chCommandLineGet<int>(&blockSize,"t", argc, argv);
	chCommandLineGet<int>(&blockSize,"threads-per-block", argc, argv);
	blockSize = blockSize != 0 ? blockSize : DEFAULT_BLOCK_DIM;

	if (blockSize > 1024) {
		std::cout << "\033[31m***" << std::endl
		          << "*** Error - The number of threads per block is too big" << std::endl
		          << "***\033[0m" << std::endl;

		exit(-1);
	}

	gridSize = ceil(static_cast<float>(numElements) / static_cast<float>(blockSize));

	dim3 grid_dim = dim3(gridSize);
	dim3 block_dim = dim3(blockSize);

	std::cout << "***" << std::endl;
	std::cout << "*** Grid: " << gridSize << std::endl;
	std::cout << "*** Block: " << blockSize << std::endl;
	std::cout << "***" << std::endl;
	
	bool silent = chCommandLineGetBool("silent", argc, argv);
	
	kernelTimerNaive.start();

	for (int i = 0; i < numIterations; i ++) {
		simpleNbody_Kernel<<<grid_dim, block_dim>>>(numElements, d_particles.posMass, 
				d_particles.velocity);
		updatePosition_Kernel<<<grid_dim, block_dim>>>(numElements, d_particles.posMass,
				d_particles.velocity);
	}
	
	// Synchronize
	cudaDeviceSynchronize();

	// Check for Errors
	cudaError_t cudaError = cudaGetLastError();
	if ( cudaError != cudaSuccess ) {
		std::cout << "\033[31m***" << std::endl
		          << "***ERROR*** " << cudaError << " - " << cudaGetErrorString(cudaError)
		          << std::endl
		          << "***\033[0m" << std::endl;

		return -1;
	}

	kernelTimerNaive.stop();

	//
	// Copy Back Data
	//
	memCpyD2HTimer.start();
	
	cudaMemcpy(h_particles_results_naive.posMass, d_particles.posMass, 
			static_cast<size_t>(numElements * sizeof(*(h_particles_results_naive.posMass))), 
			cudaMemcpyDeviceToHost);
	cudaMemcpy(h_particles_results_naive.velocity, d_particles.velocity, 
			static_cast<size_t>(numElements * sizeof(*(h_particles_results_naive.velocity))), 
			cudaMemcpyDeviceToHost);

	memCpyD2HTimer.stop();

	// Copy Data again to the GPU for optimized computation

	cudaMemcpy(d_particles.posMass, h_particles.posMass, 
		static_cast<size_t>(numElements * sizeof(float4)), 
		cudaMemcpyHostToDevice);
	cudaMemcpy(d_particles.velocity, h_particles.velocity, 
		static_cast<size_t>(numElements * sizeof(float3)), 
		cudaMemcpyHostToDevice);

	int shmem_size = blockSize * (sizeof(float4));

	kernelTimerOptimized.start();

	for (int i = 0; i < numIterations; i ++) {
		sharedNbody_Kernel<<<grid_dim, block_dim, shmem_size>>>(numElements, d_particles.posMass, 
				d_particles.velocity);
		updatePosition_Kernel<<<grid_dim, block_dim>>>(numElements, d_particles.posMass,
				d_particles.velocity);
	}
	
	// Synchronize
	cudaDeviceSynchronize();

	// Check for Errors
	cudaError = cudaGetLastError();
	if ( cudaError != cudaSuccess ) {
		std::cout << "\033[31m***" << std::endl
					<< "***ERROR*** " << cudaError << " - " << cudaGetErrorString(cudaError)
					<< std::endl
					<< "***\033[0m" << std::endl;

		return -1;
	}

	kernelTimerOptimized.stop();

	//
	// Copy Back Data from optimized computaion
	//
	
	cudaMemcpy(h_particles_results_optimized.posMass, d_particles.posMass, 
			static_cast<size_t>(numElements * sizeof(*(h_particles_results_optimized.posMass))), 
			cudaMemcpyDeviceToHost);
	cudaMemcpy(h_particles_results_optimized.velocity, d_particles.velocity, 
			static_cast<size_t>(numElements * sizeof(*(h_particles_results_optimized.velocity))), 
			cudaMemcpyDeviceToHost);

	bool test = chCommandLineGetBool("test", argc, argv);

	// check if both kernel give the same results
	if (test)
	{
		for (int i = 0; i < numElements; ++i)
		{
			if (! isEqualFloat4(h_particles_results_naive.posMass[i], h_particles_results_optimized.posMass[i]))
			{
				std::cout << "Wrong results for particle " << i << " in the posMass array!" << std::endl;
				return -1;
			}
		}
	}

	cudaFree(d_particles.posMass);
	cudaFree(d_particles.velocity);

	//
	// Exercise 3 tells to limit the device memory to 4 MB. Assuming each body consumes about 32B
	// results to roughly 128k particles. 
	// 

	int deviceLimit = 128000;

	int persistentData = deviceLimit / 2;
	int swappedData = deviceLimit - persistentData;


	Body_t d_particles_limited_default;

	cudaMalloc(&(d_particles_limited_default.posMass), 
		static_cast<size_t>(deviceLimit * sizeof(*(d_particles_limited_default.posMass))));
	cudaMalloc(&(d_particles_limited_default.velocity), 
		static_cast<size_t>(deviceLimit * sizeof(*(d_particles_limited_default.velocity))));

	int persistentData_tmp = persistentData;
	int swappedData_tmp = swappedData;

	kernelTimerLimited.start();

	for (int i = 0; i < numIterations; i ++) {
		{
			for (int j = 0; j < numElements; j += persistentData)
			{
				// copy the persistent particles
				cudaMemcpy(d_particles_limited_default.posMass, h_particles.posMass + j, 
						static_cast<size_t>(persistentData_tmp * sizeof(*(d_particles_limited_default.posMass))),
						cudaMemcpyHostToDevice);
				cudaMemcpy(d_particles_limited_default.velocity, h_particles.velocity + j, 
					static_cast<size_t>(persistentData_tmp * sizeof(*(d_particles_limited_default.velocity))),
					cudaMemcpyHostToDevice);

				for (int k = 0; k < numElements; k += swappedData)
				{
					// copy the swapped particles, only the position has to be stored
					cudaMemcpy(d_particles_limited_default.posMass + persistentData_tmp, h_particles.posMass + k, 
						static_cast<size_t>(swappedData_tmp * sizeof(*(d_particles_limited_default.posMass))),
						cudaMemcpyHostToDevice);
					streamNbody_Kernel<<<grid_dim, block_dim>>>(persistentData_tmp, swappedData_tmp, d_particles_limited_default.posMass, 
						d_particles_limited_default.velocity);
				}
				updatePosition_Kernel<<<grid_dim, block_dim>>>(persistentData_tmp, d_particles_limited_default.posMass,
					d_particles_limited_default.velocity);

				// copy the updated persistent particles back
				cudaMemcpy(h_particles_results_stream.posMass + j, d_particles_limited_default.posMass, 
					static_cast<size_t>(persistentData_tmp * sizeof(*(h_particles_results_stream.posMass))),
					cudaMemcpyDeviceToHost);
				cudaMemcpy(h_particles_results_stream.velocity + j, d_particles_limited_default.velocity, 
					static_cast<size_t>(persistentData_tmp * sizeof(*(h_particles_results_stream.velocity))),
					cudaMemcpyDeviceToHost);
			}
		}
	}

	cudaDeviceSynchronize();

	// Check for Errors
	cudaError = cudaGetLastError();
	if ( cudaError != cudaSuccess ) {
		std::cout << "\033[31m***" << std::endl
					<< "***ERROR*** " << cudaError << " - " << cudaGetErrorString(cudaError)
					<< std::endl
					<< "***\033[0m" << std::endl;

		return -1;
	}

	kernelTimerLimited.stop();

	std::cout << "after limited" << std::endl;

	// check if both kernel give the same results
	if (test)
	{
		for (int i = 0; i < numElements; ++i)
		{
			if (! isEqualFloat4(h_particles_results_naive.posMass[i], h_particles_results_stream.posMass[i]))
			{
				std::cout << "Wrong results for particle " << i << " in the posMass array!" << std::endl;
				return -1;
			}
		}
	}

	const int streams = 2;

	if (numElements % streams != 0)
	{
		std::cout << "numElements is not divisble by amount of streams" << std::endl;
		return -1;
	}

	int elementsPerStream = numElements / streams;

	int deviceLimitPerStream = deviceLimit / streams;

	if (deviceLimitPerStream % 2 != 0)
	{
		std::cout << "device array is not divisble by amount of streams" << std::endl;
		return -1;
	}

	persistentData = deviceLimitPerStream / 2;
	swappedData = deviceLimitPerStream - persistentData;

	// limit grid dim to not launch too many threads
	gridSize = ceil(static_cast<float>(persistentData) / static_cast<float>(blockSize));
	grid_dim = dim3(gridSize);

	cudaStream_t cudaStreams [streams];
	Body_t d_particles_limited [streams];

	for (int i = 0; i < streams; ++i)
	{
		cudaStreamCreate(&(cudaStreams[i]));
		cudaMalloc(&(d_particles_limited[i].posMass), 
			static_cast<size_t>(deviceLimitPerStream * sizeof(*(d_particles_limited[i].posMass))));
		cudaMalloc(&(d_particles_limited[i].velocity), 
			static_cast<size_t>(deviceLimitPerStream * sizeof(*(d_particles_limited[i].velocity))));
	}

	persistentData_tmp = persistentData;
	swappedData_tmp = swappedData;

	kernelTimerLimitedStreams.start();

	for (int i = 0; i < numIterations; i ++) {
		{
			for (int j = 0; j < elementsPerStream; j += persistentData)
			{
				// copy the persistent particles
				cudaMemcpyAsync(d_particles_limited[0].posMass, h_particles.posMass + j, 
						static_cast<size_t>(persistentData_tmp * sizeof(*(d_particles_limited[0].posMass))),
						cudaMemcpyHostToDevice, cudaStreams[0]);
				cudaMemcpyAsync(d_particles_limited[0].velocity, h_particles.velocity + j, 
					static_cast<size_t>(persistentData_tmp * sizeof(*(d_particles_limited[0].velocity))),
					cudaMemcpyHostToDevice, cudaStreams[0]);

				cudaMemcpyAsync(d_particles_limited[1].posMass, h_particles.posMass + elementsPerStream + j, 
						static_cast<size_t>(persistentData_tmp * sizeof(*(d_particles_limited[1].posMass))),
						cudaMemcpyHostToDevice, cudaStreams[1]);
				cudaMemcpyAsync(d_particles_limited[1].velocity, h_particles.velocity + elementsPerStream + j, 
					static_cast<size_t>(persistentData_tmp * sizeof(*(d_particles_limited[1].velocity))),
					cudaMemcpyHostToDevice, cudaStreams[1]);
				for (int k = 0; k < numElements; k += swappedData)
				{
					// copy the swapped particles, only the position has to be stored
					cudaMemcpyAsync(d_particles_limited[0].posMass + persistentData_tmp, h_particles.posMass + k, 
						static_cast<size_t>(swappedData_tmp * sizeof(*(d_particles_limited[0].posMass))),
						cudaMemcpyHostToDevice, cudaStreams[0]);
					streamNbody_Kernel<<<grid_dim, block_dim, 0, cudaStreams[0]>>>(persistentData_tmp, swappedData_tmp, d_particles_limited[0].posMass, 
						d_particles_limited[0].velocity);

					cudaMemcpyAsync(d_particles_limited[1].posMass + persistentData_tmp, h_particles.posMass + k, 
						static_cast<size_t>(swappedData_tmp * sizeof(*(d_particles_limited[1].posMass))),
						cudaMemcpyHostToDevice, cudaStreams[1]);
					streamNbody_Kernel<<<grid_dim, block_dim, 0, cudaStreams[1]>>>(persistentData_tmp, swappedData_tmp, d_particles_limited[1].posMass, 
						d_particles_limited[1].velocity);
				}
				updatePosition_Kernel<<<grid_dim, block_dim, 0, cudaStreams[0]>>>(persistentData_tmp, d_particles_limited[0].posMass,
					d_particles_limited[0].velocity);

				updatePosition_Kernel<<<grid_dim, block_dim, 0, cudaStreams[1]>>>(persistentData_tmp, d_particles_limited[1].posMass,
					d_particles_limited[1].velocity);
				// copy the updated persistent particles back
				cudaMemcpyAsync(h_particles_results_stream.posMass + j, d_particles_limited[0].posMass, 
					static_cast<size_t>(persistentData_tmp * sizeof(*(h_particles_results_stream.posMass))),
					cudaMemcpyDeviceToHost, cudaStreams[0]);
				cudaMemcpyAsync(h_particles_results_stream.velocity + j, d_particles_limited[0].velocity, 
					static_cast<size_t>(persistentData_tmp * sizeof(*(h_particles_results_stream.velocity))),
					cudaMemcpyDeviceToHost, cudaStreams[0]);

				cudaMemcpyAsync(h_particles_results_stream.posMass + elementsPerStream + j, d_particles_limited[1].posMass, 
					static_cast<size_t>(persistentData_tmp * sizeof(*(h_particles_results_stream.posMass))),
					cudaMemcpyDeviceToHost, cudaStreams[1]);
				cudaMemcpyAsync(h_particles_results_stream.velocity + elementsPerStream + j, d_particles_limited[1].velocity, 
					static_cast<size_t>(persistentData_tmp * sizeof(*(h_particles_results_stream.velocity))),
					cudaMemcpyDeviceToHost, cudaStreams[1]);
			}
		}
	}

	// Synchronize
	cudaDeviceSynchronize();

	// Check for Errors
	cudaError = cudaGetLastError();
	if ( cudaError != cudaSuccess ) {
		std::cout << "\033[31m***" << std::endl
					<< "***ERROR*** " << cudaError << " - " << cudaGetErrorString(cudaError)
					<< std::endl
					<< "***\033[0m" << std::endl;

		return -1;
	}

	kernelTimerLimitedStreams.stop();

	// exercise 3 finished

	// check if both kernel give the same results
	if (test)
	{
		for (int i = 0; i < numElements; ++i)
		{
			if (! isEqualFloat4(h_particles_results_naive.posMass[i], h_particles_results_stream.posMass[i]))
			{
				std::cout << "Wrong results for particle " << i << " in the posMass array!" << std::endl;
				return -1;
			}
		}
	}


	// Free Memory
	if (!pinnedMemory) {
		free(h_particles.posMass);
		free(h_particles.velocity);
	} else {
		cudaFreeHost(h_particles.posMass);
		cudaFreeHost(h_particles.velocity);
	}

	free(h_particles_results_naive.posMass);
	free(h_particles_results_naive.velocity);
	free(h_particles_results_optimized.posMass);
	free(h_particles_results_optimized.velocity);

	cudaFree(d_particles.posMass);
	cudaFree(d_particles.velocity);
	
	// Print Meassurement Results
    std::cout << "***" << std::endl
              << "*** Results:" << std::endl
              << "***    Num Elements: " << numElements << std::endl
              << "***    Num Iterations: " << numIterations << std::endl
              << "***    Threads per block: " << blockSize << std::endl
              << "***    Time to Copy to Device: " << 1e3 * memCpyH2DTimer.getTime()
                << " ms" << std::endl
              << "***    Copy Bandwidth: " 
                << 1e-9 * memCpyH2DTimer.getBandwidth(numElements * sizeof(h_particles))
                << " GB/s" << std::endl
              << "***    Time to Copy from Device: " << 1e3 * memCpyD2HTimer.getTime()
                << " ms" << std::endl
              << "***    Copy Bandwidth: " 
                << 1e-9 * memCpyD2HTimer.getBandwidth(numElements * sizeof(h_particles))
                << " GB/s" << std::endl
              << "***    Time for n-Body Computation naive implementation: " << 1e3 * kernelTimerNaive.getTime()
				<< " ms" << std::endl
			  << "***    Time for n-Body Computation optimized implementation: " << 1e3 * kernelTimerOptimized.getTime()
				<< " ms" << std::endl
			  << "***    Time for n-Body Computation streamed implementation on single stream: " << 1e3 * kernelTimerLimited.getTime()
                << " ms" << std::endl
			  << "***    Time for n-Body Computation streamed implementation on two streams: " << 1e3 * kernelTimerLimitedStreams.getTime()
                << " ms" << std::endl
              << "***" << std::endl;

	return 0;
}

void
printHelp(char * argv)
{
    std::cout << "Help:" << std::endl
              << "  Usage: " << std::endl
              << "  " << argv << " [-p] [-s <num-elements>] [-t <threads_per_block>]"
                  << std::endl
              << "" << std::endl
              << "  -p|--pinned-memory" << std::endl
              << "    Use pinned Memory instead of pageable memory" << std::endl
              << "" << std::endl
              << "  -s <num-elements>|--size <num-elements>" << std::endl
              << "    Number of elements (particles)" << std::endl
              << "" << std::endl
              << "  -i <num-iterations>|--num-iterations <num-iterations>" << std::endl
              << "    Number of iterations" << std::endl
              << "" << std::endl
              << "  -t <threads_per_block>|--threads-per-block <threads_per_block>" 
                  << std::endl
              << "    The number of threads per block" << std::endl
              << "" << std::endl
              << "  --silent" 
                  << std::endl
              << "    Suppress print output during iterations (useful for benchmarking)" << std::endl
              << "" << std::endl;
}

//
// Print one element
//
void
printElement(Body_t particles, int elementId, int iteration)
{
    float4 posMass = particles.posMass[elementId];
    float3 velocity = particles.velocity[elementId];

    std::cout << "***" << std::endl
              << "*** Printing Element " << elementId << " in iteration " << iteration << std::endl
              << "***" << std::endl
              << "*** Position: <" 
                  << std::setw(11) << std::setprecision(9) << posMass.x << "|"
                  << std::setw(11) << std::setprecision(9) << posMass.y << "|"
                  << std::setw(11) << std::setprecision(9) << posMass.z << "> [m]" << std::endl
              << "*** velocity: <" 
                  << std::setw(11) << std::setprecision(9) << velocity.x << "|"
                  << std::setw(11) << std::setprecision(9) << velocity.y << "|"
                  << std::setw(11) << std::setprecision(9) << velocity.z << "> [m/s]" << std::endl
              << "*** Mass: " 
                  << std::setw(11) << std::setprecision(9) << posMass.w << " kg"<< std::endl
              << "***" << std::endl;
}

//
// Compare float3 values on equality
//
bool
isEqualFloat3(float3 a, float3 b, float epsilon)
{
	if (abs(a.x - b.x) > epsilon)
		return false;
	if (abs(a.y - b.y) > epsilon)
		return false;
	if (abs(a.z - b.z) > epsilon)
		return false;
	return true;
}

//
// Compare float4 values on equality
//
bool
isEqualFloat4(float4 a, float4 b, float epsilon)
{
	if (abs(a.x - b.x) > epsilon)
		return false;
	if (abs(a.y - b.y) > epsilon)
		return false;
	if (abs(a.z - b.z) > epsilon)
		return false;
	if (abs(a.w - b.w) > epsilon)
		return false;
	return true;
}
