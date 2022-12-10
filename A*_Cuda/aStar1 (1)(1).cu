#include<cuda_runtime.h>
#include<iostream>
#include<list>
#include<cmath>
#include "cuda.h"
#include<device_launch_parameters.h>
#include"sm_60_atomic_functions.h"
#include<cstdlib>


__global__ void initialize_m(int* m)      //__global__ function
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *m = 0;
    }
}


__global__ void empty_s(int* s_ptr)      //__global__ function
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *s_ptr = 0;
    }
}

__global__ void initialize_Q1(int* Q1_x, int* Q1_y, double* Q1_z, double* Q1_f, int* Q1_ptr, int* Q2_ptr, int* Q3_ptr, int* closed_ptr, int* closed_x, int* closed_y, double* closed_f)      // __device__ or __global__ function
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        Q1_x[0] = 0; Q1_y[0] = 0; Q1_z[0] = 0; 
        Q1_f[0] = 32.0; Q1_ptr[0] = 1; Q2_ptr[0] = 0; Q3_ptr[0] = 0;
        *closed_ptr = 1; closed_x[0]=0; closed_y[0]=0; closed_f[0]=32;
    }

}


__global__ void AstarSearch_step1(int* Q1_x, int* Q1_y, double* Q1_z, double* Q1_f, int* Q1_ptr,
    int* Q2_x, int* Q2_y, double* Q2_z, double* Q2_f, int* Q2_ptr,
    int* Q3_x, int* Q3_y, double* Q3_z, double* Q3_f, int* Q3_ptr,
    int* S_x, int* S_y, double* S_z, double* S_f, int* S_ptr, int* m)  //not taking start and end now

{

 int gridSize = 1200;
 
    if (Q1_ptr[0] != 0 || Q2_ptr[0] != 0 || Q3_ptr[0] != 0) {  // check if Q is empty

        //all threads will run this
        if (blockIdx.x == 0 && Q1_ptr[0] != 0) {
            //pick an element and expand it. To balance the processing of all the threads each thread will expand
            int x = Q1_x[blockIdx.y*1024 + threadIdx.x], y = Q1_y[blockIdx.y*1024 + threadIdx.x]; double z = Q1_z[blockIdx.y*1024 + threadIdx.x];

            //expand current point into 8 points            

            int px[8] = { x, x, x - 1, x + 1, x - 1, x - 1,x + 1, x + 1 };
            int py[8] = { y - 1, y + 1, y, y, y - 1,  y + 1, y - 1, y + 1 };
            double pz[8] = { z + 1 + (px[0] - 4) * (px[0] - 4) + (py[0] - 4) * (py[0] - 4), z + 1 + (px[1] - 4) * (px[1]- 4) + (py[1] - 4) * (py[1] - 4), z + 1 + (px[2] - 4) * (px[2] - 4) + (py[2] - 4) * (py[2] - 4),
                         z + 1 + (px[3] - 4) * (px[3] - 4) + (py[3] - 4) * (py[3] - 4), z + 1 + (px[4] - 4) * (px[4] - 4) + (py[4] - 4) * (py[4] - 4),  z + 1 + (px[5] - 4) * (px[5] - 4) + (py[5] - 4) * (py[5] - 4),
                         z + 1 + (px[6] - 4) * (px[6] - 4) + (py[6] - 4) * (py[6] - 4), z + 1 + (px[7] - 4) * (px[7] - 4) + (py[7] -4) * (py[7] - 4) };


            if (blockIdx.y*1024 + threadIdx.x < Q1_ptr[0]) {//check if that point is valid point or not
                //check if it is a target
                for (int i = 0; i < 8; i++) {
                    //put each of this point at s without race condition
                    if (px[i] == gridSize && py[i] == gridSize) {
                        m[0] = 1;
                    }
                    
                    if (px[i] >= 0 && py[i] >= 0 && py[i] <= gridSize && px[i] <= gridSize) {
                        register int idx = atomicAdd(S_ptr, 1);
                        S_x[idx] = px[i]; S_y[idx] = py[i]; S_f[idx] = pz[i]; S_z[i] = z + 1;   //figure out this
                    }
                }
            }
        }

        if (blockIdx.x == 1 && Q2_ptr[0] != 0) {
            //pick an element and expand it. To balance the processing of all the threads each thread will expand
            int x = Q2_x[blockIdx.y*1024 + threadIdx.x], y = Q2_y[blockIdx.y*1024 + threadIdx.x]; double z = Q2_z[blockIdx.y*1024 + threadIdx.x];

            //expand current point into 8 points           

            int px[8] = { x, x, x - 1, x + 1, x - 1, x - 1,x + 1, x + 1 };
            int py[8] = { y - 1, y + 1, y, y, y - 1,  y + 1, y - 1, y + 1 };
            double pz[8] = { z + 1 + (px[0] - 4) * (px[0] - 4) + (py[0] - 4) * (py[0] - 4), z + 1 + (px[1] - 4) * (px[1]- 4) + (py[1] - 4) * (py[1] - 4), z + 1 + (px[2] - 4) * (px[2] - 4) + (py[2] - 4) * (py[2] - 4),
                         z + 1 + (px[3] - 4) * (px[3] - 4) + (py[3] - 4) * (py[3] - 4), z + 1 + (px[4] - 4) * (px[4] - 4) + (py[4] - 4) * (py[4] - 4),  z + 1 + (px[5] - 4) * (px[5] - 4) + (py[5] - 4) * (py[5] - 4),
                         z + 1 + (px[6] - 4) * (px[6] - 4) + (py[6] - 4) * (py[6] - 4), z + 1 + (px[7] - 4) * (px[7] - 4) + (py[7] -4) * (py[7] - 4) };

            if (blockIdx.y*1024 + threadIdx.x < Q2_ptr[0]) {//check if that point is valid point or not
                //check if it is a target
                for (int i = 0; i < 8; i++) {
                    //put each of this point at s without race condition
                    if (px[i] ==  gridSize && py[i] ==  gridSize) {
                        m[0] =  1;
                    }

                    if (px[i] >= 0 && py[i] >= 0 && py[i] <=  gridSize && px[i] <=  gridSize) {
                        register int idx = atomicAdd(S_ptr, 1);
                        S_x[idx] = px[i]; S_y[idx] = py[i]; S_f[idx] = pz[i]; S_z[i] = z + 1;   //figure out this
                    }

                }

            }

        }
        if (blockIdx.x == 2 && Q3_ptr[0] != 0) {
            //pick an element and expand it. To balance the processing of all the threads each thread will expand
            int x = Q3_x[blockIdx.y*1024 + threadIdx.x], y = Q3_y[blockIdx.y*1024 + threadIdx.x]; double z = Q3_z[blockIdx.y*1024 + threadIdx.x];

            //expand current point into 8 points        

            int px[8] = { x, x, x - 1, x + 1, x - 1, x - 1,x + 1, x + 1 };
            int py[8] = { y - 1, y + 1, y, y, y - 1,  y + 1, y - 1, y + 1 };
            double pz[8] = { z + 1 + (px[0] - 4) * (px[0] - 4) + (py[0] - 4) * (py[0] - 4), z + 1 + (px[1] - 4) * (px[1]- 4) + (py[1] - 4) * (py[1] - 4), z + 1 + (px[2] - 4) * (px[2] - 4) + (py[2] - 4) * (py[2] - 4),
                         z + 1 + (px[3] - 4) * (px[3] - 4) + (py[3] - 4) * (py[3] - 4), z + 1 + (px[4] - 4) * (px[4] - 4) + (py[4] - 4) * (py[4] - 4),  z + 1 + (px[5] - 4) * (px[5] - 4) + (py[5] - 4) * (py[5] - 4),
                         z + 1 + (px[6] - 4) * (px[6] - 4) + (py[6] - 4) * (py[6] - 4), z + 1 + (px[7] - 4) * (px[7] - 4) + (py[7] -4) * (py[7] - 4) };

            if (blockIdx.y*1024 + threadIdx.x < Q3_ptr[0]) {//check if that point is valid point or not
                //check if it is a target
                for (int i = 0; i < 8; i++) {
                    //put each of this point at s without race condition
                    if (px[i] ==  gridSize && py[i] ==  gridSize) {
                        m[0] = 1;
                    }
                   
                    if (px[i] >= 0 && py[i] >= 0 && py[i] <=  gridSize && px[i] <=  gridSize) {
                        register int idx = atomicAdd(S_ptr, 1);
                        S_x[idx] = px[i]; S_y[idx] = py[i]; S_f[idx] = pz[i]; S_z[i] = z + 1;
                    }

                }

            }

        }

    }
}



__global__ void AstarSearch_step2(int* S_x, int* S_y, double* S_z, double* S_f, int* S_ptr, int* closed_x, int* closed_y, int* closed_ptr, double* closed_f)
{
    //here, z signifies only g value

    int idx=blockIdx.x*1024 + threadIdx.x;
    int sx=S_x[idx];
    int sy=S_y[idx];
    
    for (int i = 0; i < closed_ptr[0]; i++) {
        if (sx == closed_x[i] && sy == closed_y[i] && idx < S_ptr[0]) {
            S_f[idx] = -99;
        }
    }
}

__global__ void AstarSearch_step2_deDuplication(int* S_x, int* S_y, double* S_z, double* S_f, int index, int* S_ptr)
{

   int num_x=S_x[index];
   int num_y=S_y[index];
   
   int idx = blockIdx.x * 1024 + threadIdx.x;
   
   if(idx > index && idx < S_ptr[0] && num_x == S_x[idx] && num_y == S_y[idx] && S_f[idx]!=-99)
   {
        S_f[idx]=-99;
   }
}


__global__ void empty_q(int* Q1_ptr, int* Q2_ptr, int* Q3_ptr)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *Q1_ptr = 0; *Q2_ptr = 0; *Q3_ptr = 0;
    }
}

__global__ void AstarSearch_step3(int* S_x, int* S_y, double* S_z, double* S_f, int* S_ptr,
    int* closed_x, int* closed_y, double* closed_f, int* closed_ptr)  //Push everything from S in the closed list if not duplicate
{
    int idx = blockIdx.x * 1024 + threadIdx.x;
    int x = S_x[idx];
    int y = S_y[idx];
    double f = S_f[idx];

    if (idx < S_ptr[0] && f != -99) { //put that element in the closed list, f -99 means it's already in the closed list
        register int idx_new = atomicAdd(closed_ptr, 1);
        closed_x[idx_new] = x; closed_y[idx_new] = y; closed_f[idx_new] = f;
    }
}

__global__ void AstarSearch_step4(int* Q1_x, int* Q1_y, double* Q1_z, double* Q1_f, int* Q1_ptr,
    int* Q2_x, int* Q2_y, double* Q2_z, double* Q2_f, int* Q2_ptr, int* Q3_x, int* Q3_y, double* Q3_z, double* Q3_f, int* Q3_ptr,
    int* S_x, int* S_y, double* S_z, double* S_f, int* S_ptr)  //Push everything from S in the priority queues if not duplicate
{
    int idx = blockIdx.x * 1024 + threadIdx.x;
    int x = S_x[idx];
    int y = S_y[idx];
    double f = S_f[idx];
    double z = S_z[idx];

    if (idx < S_ptr[0] && f != -99) { //put that element in the closed list, f -99 means it's already in the closed list
        if (idx % 3 == 0) {
            register int idx_new = atomicAdd(Q1_ptr, 1);
            Q1_x[idx_new] = x; Q1_y[idx_new] = y; Q1_f[idx_new] = f; Q1_z[idx_new] = z;
        }
        if (idx % 3 == 1) {
            register int idx_new = atomicAdd(Q2_ptr, 1);
            Q2_x[idx_new] = x; Q2_y[idx_new] = y; Q2_f[idx_new] = f; Q2_z[idx_new] = z;
        }
        if (idx % 3 == 2) {
            register int idx_new = atomicAdd(Q3_ptr, 1);
            Q3_x[idx_new] = x; Q3_y[idx_new] = y; Q3_f[idx_new] = f; Q3_z[idx_new] = z;
        }
    }
}

int main()
{ 
    int check2; int S_ptr_count;

    //We have to make two lists. open and closed. Closed list will have only (x,y,f) value. whereas open list will have (x,y,f,g) values.

    //Making closed list:
    int* closed_x; int* closed_y; double* closed_f; int* closed_ptr; int* m; 

    if (cudaMalloc(&closed_x, 4*10240 * sizeof(int)) != cudaSuccess) {
        std::cout << "Could not allocate grid_gpu" << std::endl;
    }

    if (cudaMalloc(&closed_y, 4*10240 * sizeof(int)) != cudaSuccess) {
        std::cout << "Could not allocate grid_gpu" << std::endl;
    }

    if (cudaMalloc(&closed_f, 4*10240 * sizeof(double)) != cudaSuccess) {
        std::cout << "Could not allocate grid_gpu" << std::endl;
    }
    if (cudaMalloc(&closed_ptr, sizeof(int)) != cudaSuccess) {
        std::cout << "Could not allocate grid_gpu" << std::endl;
    }

    if (cudaMalloc(&m, sizeof(int)) != cudaSuccess) {
        std::cout << "Could not allocate memory" << std::endl;
    }
 
    
    //making open list Q as priority queue. Q will be split into k=3 parts here for parallization
    int* Q1_x; int* Q1_y; double* Q1_z; double* Q1_f; int* Q1_ptr;   //for Q1, k = 1
    int* Q2_x; int* Q2_y; double* Q2_z; double* Q2_f; int* Q2_ptr;   //for Q2, k = 2
    int* Q3_x; int* Q3_y; double* Q3_z; double* Q3_f; int* Q3_ptr;   //for Q3, k = 3

    //Q1
    if (cudaMalloc(&Q1_x, 4*10240 * sizeof(int)) != cudaSuccess) {
        std::cout << "Could not allocate grid_gpu" << std::endl;
    }

    if (cudaMalloc(&Q1_y, 4*10240 * sizeof(int)) != cudaSuccess) {
        std::cout << "Could not allocate grid_gpu" << std::endl;
    }

    if (cudaMalloc(&Q1_z, 4*10240 * sizeof(double)) != cudaSuccess) {
        std::cout << "Could not allocate grid_gpu" << std::endl;
    }

    if (cudaMalloc(&Q1_f, 4*10240 * sizeof(double)) != cudaSuccess) {
        std::cout << "Could not allocate grid_gpu" << std::endl;
    }

    if (cudaMalloc(&Q1_ptr, sizeof(int)) != cudaSuccess) {
        std::cout << "Could not allocate grid_gpu" << std::endl;
    }


    //Q2
    if (cudaMalloc(&Q2_x, 4*10240 * sizeof(int)) != cudaSuccess) {
        std::cout << "Could not allocate grid_gpu" << std::endl;
    }

    if (cudaMalloc(&Q2_y, 4*10240 * sizeof(int)) != cudaSuccess) {
        std::cout << "Could not allocate grid_gpu" << std::endl;
    }

    if (cudaMalloc(&Q2_z, 4*10240 * sizeof(double)) != cudaSuccess) {
        std::cout << "Could not allocate grid_gpu" << std::endl;
    }

    if (cudaMalloc(&Q2_f, 4*10240 * sizeof(double)) != cudaSuccess) {
        std::cout << "Could not allocate grid_gpu" << std::endl;
    }

    if (cudaMalloc(&Q2_ptr, sizeof(int)) != cudaSuccess) {
        std::cout << "Could not allocate grid_gpu" << std::endl;
    }


    //Q3
    if (cudaMalloc(&Q3_x, 4*10240 * sizeof(int)) != cudaSuccess) {
        std::cout << "Could not allocate grid_gpu" << std::endl;
    }

    if (cudaMalloc(&Q3_y, 4*10240 * sizeof(int)) != cudaSuccess) {
        std::cout << "Could not allocate grid_gpu" << std::endl;
    }

    if (cudaMalloc(&Q3_z, 4*10240 * sizeof(double)) != cudaSuccess) {
        std::cout << "Could not allocate grid_gpu" << std::endl;
    }

    if (cudaMalloc(&Q3_f, 4*10240 * sizeof(double)) != cudaSuccess) {
        std::cout << "Could not allocate grid_gpu" << std::endl;
    }

    if (cudaMalloc(&Q3_ptr, sizeof(int)) != cudaSuccess) {
        std::cout << "Could not allocate grid_gpu" << std::endl;
    }

    //making open list S. S will be used in each loop to store intermediate values.
    int* S_x; int* S_y; double* S_z; double* S_f; int* S_ptr;

    if (cudaMalloc(&S_x, 5*10240 * sizeof(int)) != cudaSuccess) {
        std::cout << "Could not allocate grid_gpu" << std::endl;
    }

    if (cudaMalloc(&S_y, 5*10240 * sizeof(int)) != cudaSuccess) {
        std::cout << "Could not allocate grid_gpu" << std::endl;
    }

    if (cudaMalloc(&S_z, 5*10240 * sizeof(double)) != cudaSuccess) {
        std::cout << "Could not allocate grid_gpu" << std::endl;
    }

    if (cudaMalloc(&S_f, 5*10240 * sizeof(double)) != cudaSuccess) {
        std::cout << "Could not allocate grid_gpu" << std::endl;
    }

    if (cudaMalloc(&S_ptr, sizeof(int)) != cudaSuccess) {
        std::cout << "Could not allocate grid_gpu" << std::endl;
    }
    
    std::cout << "Memory allocation complete: " << std::endl;

    //Initialize all Qs and closed lists:
    initialize_Q1 << <1, 1 >> > (Q1_x, Q1_y, Q1_z, Q1_f, Q1_ptr, Q2_ptr, Q3_ptr, closed_ptr, closed_x, closed_y, closed_f);

    //Initialize m:
    initialize_m << <1, 1 >> > (m);

    if (cudaMemcpy(&check2, m, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {  //This blocks
        std::cout << "Could not copy from Device to Host" << std::endl;
    }

    float time_elasped;
    cudaEvent_t start, stop;  //To calculate the time
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start, 0); 
   
    dim3 grid_size(3, 40); 
 
    while (check2 ==0) {
        //empty S:
        empty_s << <1, 1 >> > (S_ptr);

        AstarSearch_step1 << <grid_size, 1024 >> > (Q1_x, Q1_y, Q1_z, Q1_f, Q1_ptr, Q2_x, Q2_y, Q2_z, Q2_f, Q2_ptr,  //8 TO 23
            Q3_x, Q3_y, Q3_z, Q3_f, Q3_ptr, S_x, S_y, S_z, S_f, S_ptr, m);
        

       AstarSearch_step2 << <50, 1024 >> > (S_x, S_y, S_z, S_f, S_ptr, closed_x, closed_y, closed_ptr, closed_f);  //25 TO 29 

        if (cudaMemcpy(&S_ptr_count, S_ptr, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {  //This blocks
                std::cout << "Could not copy from Device to Host" << std::endl;
                }
        
        for(int index=0;index<S_ptr_count;index++)
        {
            AstarSearch_step2_deDuplication<<<50,1024>>>(S_x, S_y, S_z, S_f, index, S_ptr);
        }

        AstarSearch_step3 << <50, 1024 >> > (S_x, S_y, S_z, S_f, S_ptr, closed_x, closed_y, closed_f, closed_ptr);  //30 TO 34 

        empty_q << <1, 1 >> > (Q1_ptr, Q2_ptr, Q3_ptr);

        AstarSearch_step4 << <50, 1024 >> > (Q1_x, Q1_y, Q1_z, Q1_f, Q1_ptr, Q2_x, Q2_y, Q2_z, Q2_f, Q2_ptr, Q3_x, Q3_y, Q3_z, Q3_f, Q3_ptr,
           S_x, S_y, S_z, S_f, S_ptr);  //30 TO 34

        if (cudaMemcpy(&check2, m, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {  //This blocks
        std::cout << "Could not copy from Device to Host" << std::endl;
        }
      
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elasped, start, stop); 
   
    std::cout << "Processing done" << std::endl;
 
    std::cout << "The total time taken for program to run is: " << time_elasped << " milliseconds" << std::endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaFree(closed_x);
    cudaFree(closed_y);
    cudaFree(closed_f);
    cudaFree(closed_ptr);
    
    cudaFree(Q1_x);
    cudaFree(Q1_y);
    cudaFree(Q1_z);
    cudaFree(Q1_f);
    cudaFree(Q1_ptr);
    
    cudaFree(Q2_x);
    cudaFree(Q2_y);
    cudaFree(Q2_z);
    cudaFree(Q2_f);
    cudaFree(Q2_ptr);
   
    cudaFree(Q3_x);
    cudaFree(Q3_y);
    cudaFree(Q3_z);
    cudaFree(Q3_f);
    cudaFree(Q3_ptr);
    
    cudaFree(S_x);
    cudaFree(S_y);
    cudaFree(S_z);
    cudaFree(S_f);
    cudaFree(S_ptr);
    
    cudaFree(m);
    
    return 0;
}
