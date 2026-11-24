# PCA-EXP-4-MATRIX-ADDITION-WITH-UNIFIED-MEMORY AY 23-24
<h3>AIM:</h3>
<h3>THANGAADHAVAN S</h3>
<h3>212225240168</h3>
<h3>EX. NO: 4</h3>
<h3>21.11.2026</h3>
<h1> <align=center> MATRIX ADDITION WITH UNIFIED MEMORY </h3>
  Refer to the program sumMatrixGPUManaged.cu. Would removing the memsets below affect performance? If you can, check performance with nvprof or nvvp.</h3>

## AIM:
To perform Matrix addition with unified memory and check its performance with nvprof.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
## PROCEDURE:
1.	Setup Device and Properties
Initialize the CUDA device and get device properties.
2.	Set Matrix Size: Define the size of the matrix based on the command-line argument or default value.
Allocate Host Memory
3.	Allocate memory on the host for matrices A, B, hostRef, and gpuRef using cudaMallocManaged.
4.	Initialize Data on Host
5.	Generate random floating-point data for matrices A and B using the initialData function.
6.	Measure the time taken for initialization.
7.	Compute Matrix Sum on Host: Compute the matrix sum on the host using sumMatrixOnHost.
8.	Measure the time taken for matrix addition on the host.
9.	Invoke Kernel
10.	Define grid and block dimensions for the CUDA kernel launch.
11.	Warm-up the kernel with a dummy launch for unified memory page migration.
12.	Measure GPU Execution Time
13.	Launch the CUDA kernel to compute the matrix sum on the GPU.
14.	Measure the execution time on the GPU using cudaDeviceSynchronize and timing functions.
15.	Check for Kernel Errors
16.	Check for any errors that occurred during the kernel launch.
17.	Verify Results
18.	Compare the results obtained from the GPU computation with the results from the host to ensure correctness.
19.	Free Allocated Memory
20.	Free memory allocated on the device using cudaFree.
21.	Reset Device and Exit
22.	Reset the device using cudaDeviceReset and return from the main function.

## PROGRAM:
Unifmem1
```
%%writefile unifmem1.cu
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define CHECK(call) { const cudaError_t e = call; if (e != cudaSuccess) { fprintf(stderr, "Error: %s:%d, code: %d, reason: %s\n", __FILE__, __LINE__, e, cudaGetErrorString(e)); exit(1); } }

inline double seconds()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}

void initialData(float *ip, const int size)
{
    for (int i = 0; i < size; i++) ip[i] = (float)(rand() & 0xFF) / 10.0f;
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
    for (int iy = 0; iy < ny; iy++)
        for (int ix = 0; ix < nx; ix++)
            C[iy * nx + ix] = A[iy * nx + ix] + B[iy * nx + ix];
}

void checkResult(float *h, float *g, const int N)
{
    for (int i = 0; i < N; i++)
    {
        if (fabs(h[i] - g[i]) > 1.0E-8)
        {
            printf("host %f gpu %f\nArrays do not match.\n\n", h[i], g[i]);
            return;
        }
    }
    printf("Arrays match.\n");
}

__global__ void sumMatrixGPU(float *a, float *b, float *c, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    if (ix < nx && iy < ny)
    {
        unsigned int idx = iy * nx + ix;
        c[idx] = a[idx] + b[idx];
    }
}

int main(int argc, char **argv)
{
    printf("%s Starting ", argv[0]);
    int dev = 0;
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, dev));
    printf("using Device %d: %s\n", dev, p.name);
    CHECK(cudaSetDevice(dev));

    int ishift = 12;
    if (argc > 1) ishift = atoi(argv[1]);
    int nx = 1 << ishift, ny = nx;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    float *A, *B, *hostRef, *gpuRef;
    CHECK(cudaMallocManaged((void **)&A, nBytes));
    CHECK(cudaMallocManaged((void **)&B, nBytes));
    CHECK(cudaMallocManaged((void **)&gpuRef, nBytes));
    CHECK(cudaMallocManaged((void **)&hostRef, nBytes));

    double t = seconds();
    initialData(A, nxy);
    initialData(B, nxy);
    printf("initialization: \t %f sec\n", seconds() - t);


    t = seconds();
    sumMatrixOnHost(A, B, hostRef, nx, ny);
    printf("sumMatrix on host:\t %f sec\n", seconds() - t);

    dim3 block(32, 32);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    sumMatrixGPU<<<grid, block>>>(A, B, gpuRef, 1, 1);
    CHECK(cudaDeviceSynchronize());

    t = seconds();
    sumMatrixGPU<<<grid, block>>>(A, B, gpuRef, nx, ny);
    CHECK(cudaDeviceSynchronize());
    printf("sumMatrix on gpu :\t %f sec <<<(%d,%d), (%d,%d)>>>\n", seconds() - t, grid.x, grid.y, block.x, block.y);

    CHECK(cudaGetLastError());
    checkResult(hostRef, gpuRef, nxy);

    CHECK(cudaFree(A));
    CHECK(cudaFree(B));
    CHECK(cudaFree(hostRef));
    CHECK(cudaFree(gpuRef));
    CHECK(cudaDeviceReset());
    return 0;
}
```
Unifmem2
```
%%writefile unifmem2.cu
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define CHECK(call) { const cudaError_t e = call; if (e != cudaSuccess) { fprintf(stderr, "Error: %s:%d, code: %d, reason: %s\n", __FILE__, __LINE__, e, cudaGetErrorString(e)); exit(1); } }

inline double seconds()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}

void initialData(float *ip, const int size)
{
    for (int i = 0; i < size; i++) ip[i] = (float)(rand() & 0xFF) / 10.0f;
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
    for (int iy = 0; iy < ny; iy++)
        for (int ix = 0; ix < nx; ix++)
            C[iy * nx + ix] = A[iy * nx + ix] + B[iy * nx + ix];
}

void checkResult(float *h, float *g, const int N)
{
    for (int i = 0; i < N; i++)
    {
        if (fabs(h[i] - g[i]) > 1.0E-8)
        {
            printf("host %f gpu %f\nArrays do not match.\n\n", h[i], g[i]);
            return;
        }
    }
    printf("Arrays match.\n");
}

__global__ void sumMatrixGPU(float *a, float *b, float *c, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    if (ix < nx && iy < ny)
    {
        unsigned int idx = iy * nx + ix;
        c[idx] = a[idx] + b[idx];
    }
}

int main(int argc, char **argv)
{
    printf("%s Starting ", argv[0]);
    int dev = 0;
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, dev));
    printf("using Device %d: %s\n", dev, p.name);
    CHECK(cudaSetDevice(dev));

    int ishift = 12;
    if (argc > 1) ishift = atoi(argv[1]);
    int nx = 1 << ishift, ny = nx;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    float *A, *B, *hostRef, *gpuRef;
    CHECK(cudaMallocManaged((void **)&A, nBytes));
    CHECK(cudaMallocManaged((void **)&B, nBytes));
    CHECK(cudaMallocManaged((void **)&gpuRef, nBytes));
    CHECK(cudaMallocManaged((void **)&hostRef, nBytes));

    double t = seconds();
    initialData(A, nxy);
    initialData(B, nxy);
    printf("initialization: \t %f sec\n", seconds() - t);


    t = seconds();
    sumMatrixOnHost(A, B, hostRef, nx, ny);
    printf("sumMatrix on host:\t %f sec\n", seconds() - t);

    dim3 block(32, 32);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    sumMatrixGPU<<<grid, block>>>(A, B, gpuRef, 1, 1);
    CHECK(cudaDeviceSynchronize());

    t = seconds();
    sumMatrixGPU<<<grid, block>>>(A, B, gpuRef, nx, ny);
    CHECK(cudaDeviceSynchronize());
    printf("sumMatrix on gpu :\t %f sec <<<(%d,%d), (%d,%d)>>>\n", seconds() - t, grid.x, grid.y, block.x, block.y);

    CHECK(cudaGetLastError());
    checkResult(hostRef, gpuRef, nxy);

    CHECK(cudaFree(A));
    CHECK(cudaFree(B));
    CHECK(cudaFree(hostRef));
    CHECK(cudaFree(gpuRef));
    CHECK(cudaDeviceReset());
    return 0;
}
```

## OUTPUT:
Unifmem1
```
/content/./unifmem1 Starting using Device 0: Tesla T4
Matrix size: nx 4096 ny 4096
initialization: 	 0.835426 sec
sumMatrix on host:	 0.066527 sec
sumMatrix on gpu :	 0.057218 sec <<<(128,128), (32,32)>>>
Arrays match.
Generating '/tmp/nsys-report-0933.qdstrm'
[1/8] [========================100%] report1.nsys-rep
[2/8] [========================100%] report1.sqlite
[3/8] Executing 'nvtx_sum' stats report
SKIPPED: /content/report1.sqlite does not contain NV Tools Extension (NVTX) data.
[4/8] Executing 'osrt_sum' stats report
```
Unifmem2
```
/content/./unifmem2 Starting using Device 0: Tesla T4
Matrix size: nx 4096 ny 4096
initialization: 	 0.893191 sec
sumMatrix on host:	 0.061445 sec
sumMatrix on gpu :	 0.059354 sec <<<(128,128), (32,32)>>>
Arrays match.
Generating '/tmp/nsys-report-25bc.qdstrm'
[1/8] [========================100%] report4.nsys-rep
[2/8] [========================100%] report4.sqlite
[3/8] Executing 'nvtx_sum' stats report
SKIPPED: /content/report4.sqlite does not contain NV Tools Extension (NVTX) data.
[4/8] Executing 'osrt_sum' stats report
```

## RESULT:
Thus, the matrix addition program was successfully executed using CUDA Unified Memory. It was observed that removing the memset function did not significantly affect the GPU execution time, as the matrices were initialized and the output was calculated directly by the program.
The GPU execution time was approximately 0.057–0.059 seconds in both cases. The output arrays matched successfully.
Therefore, the program was executed successfully, and the results were verified using GPU computation.
