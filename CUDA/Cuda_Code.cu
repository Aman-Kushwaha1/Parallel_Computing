#include <iostream>
#include<cmath>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include"sm_60_atomic_functions.h"
#include<cstdlib>



__global__ void func_gpu(double* array_ptr, double* array_ptr2, int grid, int n, double* sum_ptr, int itr)
{

    //Code that will be processed on the GPU will be written here

    __shared__ double sum_t[1024]; //will be used to do sum reduction.
    //initializing with zero
    sum_t[threadIdx.y * 32 + threadIdx.x] = 0.0;

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    double p, q, r, s;

    if ((i > 0) && (j > 0) && (i < n - 1) && (j < n - 1)) {
        p = array_ptr2[((i - 1) * n + (j + 1))];
        q = array_ptr2[(i - 1) * n + j - 1];
        r = array_ptr2[(i + 1) * n + j + 1];
        s = array_ptr2[(i + 1) * n + j - 1];
    }
    else {
        p = 0;
        q = 0;
        r = 0;
        s = 0;
    }

    //Calculate the 2nd smallest value:
    double a_temp[] = { p,q,r,s };
    double second_smallest;

    for (int i1 = 0; i1 < 4; i1++) {
        for (int j1 = i1+1; j1 < 4; j1++) {
            if (a_temp[i1] > a_temp[j1]) {
                double temp = a_temp[i1];
                a_temp[i1] = a_temp[j1];
                a_temp[j1] = temp;
            }
        }
    }
    second_smallest = a_temp[1];

    //synchronize threads:
    __syncthreads();

    int element_index = i * n + j;    //For mapping to GPU Global memory

    //update to orignal array: (need this because we need to do 10 loops)
    if ((i > 0) && (j > 0) && (i < n - 1) && (j < n - 1)) {
        array_ptr[element_index] = second_smallest;
    }

    
    __syncthreads();

    if ((i > 0) && (j > 0) && (i < n - 1) && (j < n - 1) && itr == 9) {

        //putting values in shared memory
       sum_t[threadIdx.y * 32 + threadIdx.x] = second_smallest;
    }

    __syncthreads();

    ////////////////////////////////////////////////////////////////////////

    if (itr == 9) {
        //taking the border values for calcilation:


        if (((i == 0) && (j < n)) || ((j == 0) && (i < n)) || ((i == n - 1) && (j < n)) || ((j == n - 1) && (i < n))) {
            sum_t[threadIdx.y * 32 + threadIdx.x] = array_ptr[element_index];
        }

        __syncthreads();

        //Performing sum reduction on sum_t
        for (int m = 1; m < 1024; m = m * 2) {
            if ((threadIdx.y * 32 + threadIdx.x) % (2 * m) == 0) {
                sum_t[threadIdx.y * 32 + threadIdx.x] += sum_t[threadIdx.y * 32 + threadIdx.x + m];
            }
            __syncthreads();
        }
        __syncthreads();

        if (threadIdx.x == 0 && threadIdx.y == 0) {
            sum_ptr[blockIdx.y * grid + blockIdx.x] = sum_t[0];
        }
    }
    
}

__global__ void func_gpu_cpy(double* array_ptr, double* array_ptr2, int grid, int n, double* sum_ptr, int itr)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    int element_index = i * n + j;

    if ((i > 0) && (j > 0) && (i < n - 1) && (j < n - 1)) {
        array_ptr2[element_index] = array_ptr[element_index];
    }
}



__global__ void verification_gpu(double* sum_ptr, double* sum_gpu, int grid, double* verification_val, int n, double* array_1val)
{

    //Performing sum reduction on sum_ptr:
    __shared__ double local_sum[1024];

    local_sum[(threadIdx.x)] = 0.0;

    __syncthreads();

    if ((blockIdx.x * 1024 + threadIdx.x) < grid * grid) {
        local_sum[(threadIdx.x)] = sum_ptr[(blockIdx.x * 1024 + threadIdx.x)];
    }

    __syncthreads();

    for (int m = 1; m < 1024; m = m * 2) {
        if (threadIdx.x % (2 * m) == 0) {
            local_sum[(threadIdx.x)] += local_sum[(threadIdx.x) + m];
        }
        __syncthreads();
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        sum_ptr[blockIdx.x] = local_sum[0];
    }

}

__global__ void verification_gpu1(double* sum_ptr, double* sum_gpu, int grid, double* verification_val, int n, double* array_1val)
{

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        sum_gpu[0] = 0;
        
        for (int i = 0; i < gridDim.x; i++) {
            sum_gpu[0] += sum_ptr[i];
        }
        verification_val[0] = array_1val[37 * n + 47];
    }
}

int main()
{
    const int n = 1000; double sum_val = 0; double finalsum; double final_verification; double verify[16];

    double array1[n][n];
    double array2[n][n];

    double intermediate;

    //Initializing the array in the CPU:
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            intermediate = (1 + std::cos(2 * i) + std::sin(j));
            array1[i][j] = std::pow(intermediate, 2);
            array2[i][j] = std::pow(intermediate, 2);
        }
    }

    //No of Blocks and Grids 
    int grid = ceil(n / 32.0);

    int launch_blocks = ceil(grid * grid / 1024.0);
    //std::cout << "Launch Blocks: " << launch_blocks << std::endl;

    //Arrays and variables in the GPU:
    double* array_1; double* array_2;  double* sum; double* temp_sum; double* verification_value;

    //Assigning memory space in the GPU for the variables:
    if (cudaMalloc(&array_1, n * n * sizeof(double)) != cudaSuccess) {
        std::cout << "Could not allocate array_1" << std::endl;
    }

    if (cudaMalloc(&array_2, n * n * sizeof(double)) != cudaSuccess) {
        std::cout << "Could not allocate array_2" << std::endl;
    }

    if (cudaMalloc(&temp_sum, grid * grid * sizeof(double)) != cudaSuccess) {
        std::cout << "Could not allocate temp_sum" << std::endl;
    }

    if (cudaMalloc(&sum, sizeof(double)) != cudaSuccess) {
        std::cout << "Could not allocate sum" << std::endl;
    }

    if (cudaMalloc(&verification_value, sizeof(double)) != cudaSuccess) {
        std::cout << "Could not allocate verification_value" << std::endl;
    }



    //Copying the array to the GPU Memory:
    if (cudaMemcpy(array_1, &array1, n * n * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cout << "Could not copy from Host to Device" << std::endl;
    }
    if (cudaMemcpy(array_2, &array2, n * n * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cout << "Could not copy from Host to Device" << std::endl;
    }

    if (cudaMemcpy(sum, &sum_val, sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cout << "Could not copy from Host to Device" << std::endl;
    }

    //Calling the cuda kernal (Function):
    dim3 grid_size(grid, grid);
    dim3 block_size(32, 32);

    //call this function 10 times:
    int num_times = 10;
    
    float time_elasped;
    cudaEvent_t start, stop;  //To calculate the time
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start, 0); 
    /*
    for (int i = 0; i < num_times; i++) {
        func_gpu << <grid_size, block_size >> > (array_1, grid, n, temp_sum, i);   //Launching a kernal is very cheap. The CPU won't wait for the kernal to finish and will will start processing next line.
        cudaDeviceSynchronize();
    }*/

    func_gpu << <grid_size, block_size >> > (array_1, array_2, grid, n, temp_sum, 0);
    func_gpu_cpy << <grid_size, block_size >> > (array_1, array_2, grid, n, temp_sum, 0);

    func_gpu << <grid_size, block_size >> > (array_1, array_2, grid, n, temp_sum, 1);
    func_gpu_cpy << <grid_size, block_size >> > (array_1, array_2, grid, n, temp_sum, 0);

    func_gpu << <grid_size, block_size >> > (array_1, array_2, grid, n, temp_sum, 2);
    func_gpu_cpy << <grid_size, block_size >> > (array_1, array_2, grid, n, temp_sum, 0);

    func_gpu << <grid_size, block_size >> > (array_1, array_2, grid, n, temp_sum, 3);
    func_gpu_cpy << <grid_size, block_size >> > (array_1, array_2, grid, n, temp_sum, 0);

    func_gpu << <grid_size, block_size >> > (array_1, array_2, grid, n, temp_sum, 4);
    func_gpu_cpy << <grid_size, block_size >> > (array_1, array_2, grid, n, temp_sum, 0);

    func_gpu << <grid_size, block_size >> > (array_1, array_2, grid, n, temp_sum, 5);
    func_gpu_cpy << <grid_size, block_size >> > (array_1, array_2, grid, n, temp_sum, 0);

    func_gpu << <grid_size, block_size >> > (array_1, array_2, grid, n, temp_sum, 6);
    func_gpu_cpy << <grid_size, block_size >> > (array_1, array_2, grid, n, temp_sum, 0);

    func_gpu << <grid_size, block_size >> > (array_1, array_2, grid, n, temp_sum, 7);
    func_gpu_cpy << <grid_size, block_size >> > (array_1, array_2, grid, n, temp_sum, 0);

    func_gpu << <grid_size, block_size >> > (array_1, array_2, grid, n, temp_sum, 8);
    func_gpu_cpy << <grid_size, block_size >> > (array_1, array_2, grid, n, temp_sum, 0);

    func_gpu << <grid_size, block_size >> > (array_1, array_2, grid, n, temp_sum, 9);
    func_gpu_cpy << <grid_size, block_size >> > (array_1, array_2, grid, n, temp_sum, 0);

    cudaDeviceSynchronize(); //not required

    //Determining the verification values:
    verification_gpu << <launch_blocks, 1024 >> > (temp_sum, sum, grid, verification_value, n, array_1);
    verification_gpu1 << <launch_blocks, 1024 >> > (temp_sum, sum, grid, verification_value, n, array_1);

    cudaDeviceSynchronize();
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elasped, start, stop); 


    //copy the verification value back to GPU
    if (cudaMemcpy(&finalsum, sum, sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {  //This blocks
        std::cout << "Could not copy from Device to Host" << std::endl;
    }

    if (cudaMemcpy(&final_verification, verification_value, sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {  //This blocks
        std::cout << "Could not copy from Device to Host" << std::endl;
    }

    if (cudaMemcpy(&array1, array_1, n*n*sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {  //This blocks
        std::cout << "Could not copy from Device to Host" << std::endl;
    }

    double vsum = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            vsum += array1[i][j];
            //std::cout << array1[i][j] << " ";
        }
        //std::cout << std::endl;
    }

    std::cout << "The final sum value is: " << finalsum << std::endl;
    std::cout << "The value of A(37,47) is: " << final_verification << std::endl;
    std::cout << "The total time taken for program to run is: " << time_elasped << " milliseconds" << std::endl;
    std::cout<<"verification sum is: " << vsum << std::endl;

    cudaFree(array_1);
    cudaFree(temp_sum);
    cudaFree(sum);
    cudaFree(verification_value);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}