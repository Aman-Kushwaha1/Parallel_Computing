#include<iostream>
#include<queue>
#include<omp.h>
#include<vector>
#include"f.h"

using std::cout;


int main(){


double start_time = omp_get_wtime();

double max_value= std::max(f(a), f(b));  //This is in shared space can be excessed by all the threads

//priority queue:
std::priority_queue<std::vector<double>> pq;

double temp_var = (f(a) + f(b) + s*(a-b))/2;
std::vector<double> t{temp_var, a, b};

pq.push(t);

std::vector <bool> stat_all;  //to keep track and find out when to exit the parallel program
for(int i=0; i<33; i++){
    stat_all.push_back(false);
}

#pragma omp parallel
{   
    double c, d; bool stat = false; double f_d; double f_c; bool check_loop = true; 
        double temp;
        //Algorithm to divide the work in parallel:

        while (pq.size() !=0  || check_loop ) {

        #pragma omp critical
        {
            if(pq.size() != 0){
                c = (pq.top())[1];  
                d = (pq.top())[2];  
                pq.pop();
                stat = true;
                stat_all[omp_get_thread_num()] = true;
            }
            else{
                stat = false;
                stat_all[omp_get_thread_num()] = false;
            }
        }

        f_c = f(c); f_d = f(d);

        if (stat){
        #pragma omp critical
        {
        if (f_d > max_value || f_c > max_value){
            max_value = std::max(f_d, f_c );
        }
        }

        temp = (f_c + f_d + s*(d-c))/2;

        if (temp > (max_value + epsilon) && (d-c) > 0.01){
        
        //update the priority queue values:
        double div = (c+d)/2;
        double temp1 = (f_c + f(div) + s*(div - c))/2;
        double temp2 = (f( div ) + f_d + s*(d - div))/2;
        std::vector<double> t1{temp1, c, div};
        std::vector<double> t2{temp2, div, d};
        #pragma omp critical
        {   pq.push(t1);
            pq.push(t2);
        }
        }
        }

        int counter = 0;
        for(int m=0; m<33; m++){
            if(stat_all[m] == true ){
                counter++;
                break;
            }
        }
        if (counter ==0) check_loop = false;

}

}
double end_time = omp_get_wtime();

cout<<"The maximum value is: "<<max_value<<std::endl;
cout<<"The time taken to run the code is: "<<end_time - start_time <<std::endl;
    
return 0;
}