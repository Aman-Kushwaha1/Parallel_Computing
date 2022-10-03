// MPI.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include<cmath>
#include<mpi.h>

const int num_row = 2000; const int num_column = 500;

int main(int argc, char* argv[])
{    
    int numprocess, rank, length;
    char name[100];
   
    double array1[num_row][num_column];
    int tagrow=9;
    double buffer_msg[num_column];

    MPI_Status status;

    MPI_Init(&argc, &argv); //initialize the MPI execution environment, each process is called a rank
    double start = MPI_Wtime();

    //MPI_Comm_WORLD is default communicator and processes (rank) are inside this
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //total no of process or rank in comm world
    MPI_Comm_size(MPI_COMM_WORLD, &numprocess);

    //name of the processor:
    MPI_Get_processor_name(name, &length);

    //Doing calculations which have to be done in parallel but without synchronization. Thus this caluculations happens in different processors parallely depending upon rank

    //step 1:

    if (rank == 0) {
        for (int i = 0; i < num_row; i++) {
            //std::cout << "0 " << std::endl;
            MPI_Recv(&buffer_msg, num_column, MPI_DOUBLE, MPI_ANY_SOURCE, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //first i is recieving from which process, second one is tag for the message from the process (rank) i.
            for (int j = 0; j < num_column; j++) {
                array1[i][j] = buffer_msg[j];
            }
        }
    }
    else {
        int temp = 0;  //for counter
        for (int i = 0; i < num_row; i++) {
            temp++;
            if (temp==rank) {
                //std::fill_n(buffer_msg, num_column, 1);
                for (int j = 0; j < num_column; j++) {
                    buffer_msg[j] = ( i * sin(i) + j * cos(j) + sqrt(i + j) );
                }
                MPI_Send(&buffer_msg, num_column, MPI_DOUBLE, 0, i, MPI_COMM_WORLD);
            }
            if (temp == numprocess-1) {
                temp = 0; //resetting the counter
            }
        }
    }


    
    
    //step 2:

    if (rank == 0) {
        for (int i = 1; i < numprocess; i++) {
            MPI_Send(&array1, num_column*num_row, MPI_DOUBLE, i, i, MPI_COMM_WORLD);
        }
    }
    else {
        for (int i = 1; i < numprocess; i++) {
            if (rank == i) {
                MPI_Recv(&array1, num_column* num_row, MPI_DOUBLE, 0, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    //now doing the calculations to iterative process of step 2
    double buffer_out1[num_column];
    if (rank == 0) {
        for (int i = 1; i < num_row-1; i++) {
            double buffer_out2[num_column];
            //std::cout << "0 " << std::endl;
            MPI_Recv(buffer_out2, num_column, MPI_DOUBLE, MPI_ANY_SOURCE, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //first i is recieving from which process, second one is tag for the message from the process (rank) i.
            for (int j = 1; j < num_column-1; j++) {
                array1[i][j] = buffer_out2[j];
            }
        }
    }
    else {
        int temp = 0;  //for counter
        for (int i = 1; i < num_row-1; i++) {
            temp++;
            if (temp == rank) {
                for (int j = 1; j < num_column-1; j++) {
                    double t; t = ((array1[i - 1][j] + array1[i + 1][j] + array1[i][j - 1] + array1[i][j + 1] + array1[i][j]) / 5);
                    double m = std::min(100.0, t);
                    buffer_out1[j] = std::max(-100.0,m );
                }
                MPI_Send(buffer_out1, num_column, MPI_DOUBLE, 0, i, MPI_COMM_WORLD);
            }
            if (temp == numprocess - 1) {
                temp = 0; //resetting the counter
            }
        }
    }

    //std::cout << "exiting rank: " << rank << std::endl;
    double end = MPI_Wtime();
    if (rank == 0) {
        std::cout << "The process took " << end - start << " seconds to run." << std::endl;
        /*
        std::cout << "Printing the final results: " << std::endl;
        for (int i = 0; i < num_row; i++) {
            for (int j = 0; j < num_column; j++) {
                std::cout << array1[i][j] << " ";
            }
            std::cout << "\n";
        }  */
    }

    MPI_Finalize();  //MPI step is binded between MPI init and MPI final. This part of the code will be run in the parallel
    return 0;

}
