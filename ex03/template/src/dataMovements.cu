#include <stdio.h>
#include <chTimer.h>
#include <iostream>

int main(int argc, char *argv[]){

    const int size = std::stoi(argv[1]);

    float *hostPtr;
    float *pinPtr;
    float *devPtr;
    float *devPtr2;

    chTimerTimestamp start, stop;

    printf("size: %d\n", int(size));
    double data = (double) (N * sizeof(int)) / 1e9;

    hostPtr = (float*)  malloc(size);
    cudaMalloc( (void**) &devPtr, size);
    cudaMalloc( (void**) &devPtr2, size);
    cudaMallocHost( (void**) &pinPtr, size);

    chTimerGetTime(&start);
    cudaMemcpy(devPtr, hostPtr, size, cudaMemcpyHostToDevice);
    chTimerGetTime(&stop);
    double memH2D = 1e6*chTimerElapsedTime(&start, &stop);
    printf("NormalH2D: %f us\n", (double)(data/memH2D));

    chTimerGetTime(&start);
    cudaMemcpy(devPtr, hostPtr, size, cudaMemcpyDeviceToHost);
    chTimerGetTime(&stop);
    double memD2H = 1e6*chTimerElapsedTime(&start, &stop);
    printf("NormalD2H: %f us\n", (double)(data/memD2H));

    chTimerGetTime(&start);
    cudaMemcpy(devPtr, pinPtr, size, cudaMemcpyHostToDevice);
    chTimerGetTime(&stop);
    double pinnedH2D = 1e6*chTimerElapsedTime(&start, &stop);
    printf("PinnedH2D: %f us\n", (double)(data/pinnedH2D));

    chTimerGetTime(&start);
    cudaMemcpy(pinPtr, devPtr, size, cudaMemcpyDeviceToHost);
    chTimerGetTime(&stop);
    double pinnedD2H = 1e6*chTimerElapsedTime(&start, &stop);
    printf("PinnedD2H: %f us\n", (double)(data/pinnedD2H));

    chTimerGetTime(&start);
    cudaMemcpy(devPtr,devPtr2,N * sizeof(int),cudaMemcpyDeviceToDevice);
    chTimerGetTime(&stop);
    double memD2D = 1e6*chTimerElapsedTime(&start, &stop);
    printf("PinnedD2H: %f us\n", (doube)(data/pinnedD2H));

}
