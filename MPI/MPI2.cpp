// MPI.cpp : This file contains the 'main' function. Program execution begins and ends there.
//


#include <iostream>
#include<mpi.h>
#include<cmath>

const int num_row = 2000; const int num_column = 500;

double f(double x) {
    double y = x;
    for (int i = 1; i <= 10; i++) {
        y = y + sin(x * i) / pow(2, i);
    }
    return y;
}

int main(int argc, char* argv[])
{    
    int numprocess, rank, length;
    char name[100];
   
    double array1[num_row][num_column];
    double array2[num_row][num_column];
    double ghost[2][num_column];
    
    double buffer_msg[num_column];
    int start = 0; int end = 0;


    MPI_Status status;

    MPI_Init(&argc, &argv); //initialize the MPI execution environment, each process is called a rank
    

    //MPI_Comm_WORLD is default communicator and processes (rank) are inside this
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //total no of process or rank in comm world
    MPI_Comm_size(MPI_COMM_WORLD, &numprocess);

    //name of the processor:
    MPI_Get_processor_name(name, &length);

    //Doing calculations which have to be done in parallel. Thus this caluculations happens in different processors parallely depending upon rank

    if (numprocess == 1) {

        //iterative steps for non parallel operation. This is the step "To initialize A"
        for (int i = 0; i < num_row; i++) {
            for (int j = 0; j < num_column; j++) {
                array1[i][j] = (i * sin(i) + j * cos(j) + sqrt(i + j));
                array2[i][j] = (i * sin(i) + j * cos(j) + sqrt(i + j));   
            }
        }

        //MPI_Barrier(MPI_COMM_WORLD); doesn't makes sense here

        double start_time = MPI_Wtime();
        //iterative step
        for (int steps = 0; steps < 10; steps++) {
            for (int i = 1; i < num_row - 1; i++) {
                for (int j = 1; j < num_column - 1; j++) {
                    double t = ((f(array2[i - 1][j]) + f(array2[i + 1][j]) + f(array2[i][j - 1]) + f(array2[i][j + 1]) + f(array2[i][j])) / 5);
                    double m1 = std::min(100.0, t);
                    array1[i][j] = std::max(-100.0, m1);
                }
            }
            for (int i = 1; i < num_row - 1; i++) {
                for (int j = 1; j < num_column - 1; j++) {
                    array2[i][j] = array1[i][j];
                }
            }
        }
        

        //printing results for verification:
        double sum_verify = 0; double sum_square = 0;
        for (int i = 0; i < num_row; i++) {
            for (int j = 0; j < num_column; j++) {
                sum_verify += array1[i][j];
                sum_square += (array1[i][j]* array1[i][j]);

            }
        }
        std::cout << "The sum of all the entries of A is: " << sum_verify << std::endl;
        std::cout << "The sum squares of  all the entries of A is: " << sum_square << std::endl;

        double end_time = MPI_Wtime();
        std::cout << "The process took " << end_time - start_time << " seconds to run." << std::endl;
    }
    else {  //if numprocess>1. This means we have to distribute the work among the processes

        if (rank == 0) { //rank 0 recieves the calculated results from all the other process and stores them to array1
            for (int i = 0; i < num_row; i++) {
                MPI_Recv(&buffer_msg, num_column, MPI_DOUBLE, MPI_ANY_SOURCE, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
                for (int j = 0; j < num_column; j++) {
                    array1[i][j] = buffer_msg[j];
                }
            }
        }
        else {
            double cou = 0; int c_index = 0; 
            for (int i = 0; i < numprocess - 1; i++) {
                int temp = ceil((num_row - cou) / (numprocess - 1 - i));
                cou += temp;
                end = cou - 1;
                start = cou - temp;

                //Dividing the given array into small peices to divide the work among various processes for parallel processing
                if (rank == i + 1) {
                    for (int i = start; i <= end; i++) {
                        for (int j = 0; j < num_column; j++) {
                            buffer_msg[j] = (i * sin(i) + j * cos(j) + sqrt(i + j));
                            array1[i][j] = (i * sin(i) + j * cos(j) + sqrt(i + j));
                            array2[i][j] = (i * sin(i) + j * cos(j) + sqrt(i + j));
                        }
                        MPI_Send(&buffer_msg, num_column, MPI_DOUBLE, 0, i, MPI_COMM_WORLD);
                    }
                }
            }
        }

        double start_time = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);

        for (int numtimes = 0; numtimes < 10; numtimes++) {
            //sending ghost cell values to different processes:
            double cou = 0;
            for (int i = 0; i < numprocess - 1; i++) {

                int temp = ceil((num_row - cou) / (numprocess - 1 - i));
                cou += temp;
                end = cou - 1;
                start = cou - temp;

                if (rank == 0) {
                    if (start == 0) {
                        for (int n = 0; n < num_column; n++) {
                            ghost[0][n] = array1[start][n];
                            ghost[1][n] = array1[end + 1][n];
                        }
                    }
                    else if (end == num_row - 1) {
                        for (int n = 0; n < num_column; n++) {
                            ghost[0][n] = array1[start - 1][n];
                            ghost[1][n] = array1[end][n];
                        }
                    }
                    else {
                        for (int n = 0; n < num_column; n++) {
                            ghost[0][n] = array1[start - 1][n];
                            ghost[1][n] = array1[end + 1][n];
                        }
                    }
                    //std::cout << "send: " << rank << std::endl;
                    MPI_Send(&ghost, 2 * num_column, MPI_DOUBLE, i + 1, i + 1, MPI_COMM_WORLD);
                }

                //std::cout << "rank: " << rank << std::endl;
                if (rank == i + 1) {
                    MPI_Recv(&ghost, 2 * num_column, MPI_DOUBLE, 0, i + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }

            }

            //Doing the final calculation of "Iterative step" once the values in the ghost cells have been received

            if (rank == 0) {  //rank 0 recieves the final calculation
                for (int i = 1; i < num_row - 1; i++) {
                    MPI_Recv(&buffer_msg, num_column, MPI_DOUBLE, MPI_ANY_SOURCE, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    for (int j = 1; j < num_column - 1; j++) {
                        array1[i][j] = buffer_msg[j];
                    }
                }
            }
            else {  //other ranks do the calculation in parallel by splittting the array each needs to process in parallel
                double cou = 0;
                for (int i = 0; i < numprocess - 1; i++) {
                    int temp = ceil((num_row - cou) / (numprocess - 1 - i));
                    cou += temp;
                    end = cou - 1;
                    start = cou - temp;

                    if (rank == i + 1) {

                        if (start == 0) {

                            for (int m = start + 1; m <= end - 1; m++) {
                                for (int n = 1; n < num_column - 1; n++) {
                                    double t; t = ((f(array1[m - 1][n]) + f(array1[m + 1][n]) + f(array1[m][n - 1]) + f(array1[m][n + 1]) + f(array1[m][n])) / 5);
                                    double m1 = std::min(100.0, t);
                                    array2[m][n] = std::max(-100.0, m1);
                                }
                                MPI_Send(&array2[m], num_column, MPI_DOUBLE, 0, m, MPI_COMM_WORLD);
                            }
                            //calculation for last row:
                            for (int n = 1; n < num_column - 1; n++) {
                                double z; z = ((f(array1[end - 1][n]) + f(ghost[1][n]) + f(array1[end][n - 1]) + f(array1[end][n + 1]) + f(array1[end][n])) / 5);
                                double m1 = std::min(100.0, z);
                                array2[end][n] = std::max(-100.0, m1);
                            }
                            MPI_Send(&array2[end], num_column, MPI_DOUBLE, 0, end, MPI_COMM_WORLD);
                        }

                        else if (end == num_row - 1) {
                            //calculation for first row:

                            for (int n = 1; n < num_column - 1; n++) {
                                double t;
                                t = ((f(ghost[0][n]) + f(array1[start + 1][n]) + f(array1[start][n - 1]) + f(array1[start][n + 1]) + f(array1[start][n])) / 5);
                                double m1 = std::min(100.0, t);
                                array2[start][n] = std::max(-100.0, m1);

                            }
                            MPI_Send(&array2[start], num_column, MPI_DOUBLE, 0, start, MPI_COMM_WORLD);

                            for (int m = start + 1; m <= end - 1; m++) {
                                for (int n = 1; n < num_column - 1; n++) {
                                    double t = ((f(array1[m - 1][n]) + f(array1[m + 1][n]) + f(array1[m][n - 1]) + f(array1[m][n + 1]) + f(array1[m][n])) / 5);
                                    double m1 = std::min(100.0, t);
                                    array2[m][n] = std::max(-100.0, m1);
                                }
                                MPI_Send(&array2[m], num_column, MPI_DOUBLE, 0, m, MPI_COMM_WORLD);
                            }

                        }

                        else {
                            for (int m = start; m <= end; m++) {
                                if (m == start) {
                                    for (int n = 1; n < num_column - 1; n++) {
                                        ///std::cout << "one" << std::endl;
                                        double t;
                                        t = ((f(ghost[0][n]) + f(array1[m + 1][n]) + f(array1[m][n - 1]) + f(array1[m][n + 1]) + f(array1[m][n])) / 5);
                                        double m1 = std::min(100.0, t);
                                        array2[m][n] = std::max(-100.0, m1);

                                    }
                                    MPI_Send(&array2[m], num_column, MPI_DOUBLE, 0, start, MPI_COMM_WORLD);
                                }
                                else if (m == end) {
                                    for (int n = 1; n < num_column - 1; n++) {
                                        //std::cout << "two" << std::endl;
                                        double t;
                                        t = ((f(array1[m - 1][n]) + f(ghost[1][n]) + f(array1[m][n - 1]) + f(array1[m][n + 1]) + f(array1[m][n])) / 5);
                                        double m1 = std::min(100.0, t);
                                        array2[m][n] = std::max(-100.0, m1);

                                    }
                                    MPI_Send(&array2[m], num_column, MPI_DOUBLE, 0, end, MPI_COMM_WORLD);
                                }
                                else {
                                    for (int n = 1; n < num_column - 1; n++) {
                                        double t;
                                        t = ((f(array1[m - 1][n]) + f(array1[m + 1][n]) + f(array1[m][n - 1]) + f(array1[m][n + 1]) + f(array1[m][n])) / 5);
                                        double m1 = std::min(100.0, t);
                                        array2[m][n] = std::max(-100.0, m1);

                                    }
                                    MPI_Send(&array2[m], num_column, MPI_DOUBLE, 0, m, MPI_COMM_WORLD);
                                }
                            }
                        }
                    }
                }
                //copying array2 to array1
                if (rank != 0) {
                    for (int p = 0; p < num_row; p++) {
                        for (int q = 0; q < num_column; q++) {
                            array1[p][q] = array2[p][q];
                        }
                    }
                }
            }
        } //runs 10 times as stated in question. The 10 times run loop finishes here

            

            //printing results for verification:
            double sum_verify = 0.0; double sum_square = 0.0;
            if (rank == 0) {
                
                for (int i = 0; i < num_row; i++) {
                    for (int j = 0; j < num_column; j++) {
                        sum_verify += array1[i][j];
                        sum_square += (array1[i][j] * array1[i][j]);

                    }
                }
                double end_time = MPI_Wtime();
                std::cout << "\nThe sum of all the entries of A is: " << sum_verify << std::endl;
                std::cout << "The sum squares of  all the entries of A is: " << sum_square << std::endl;
                std::cout << "The process took " << end_time - start_time << " seconds to run." << std::endl;

            }
            
        }
        
        MPI_Finalize();  //MPI step is binded between MPI init and MPI final.
        return 0;
  
}


