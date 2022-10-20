#include<iostream>
#include<queue>
#include<omp.h>
#include<vector>
#include"f.h"
#include <chrono>

using std::cout;


int main(){


auto start_time = std::chrono::high_resolution_clock::now();

double max_value= std::max(f(a), f(b));  //This is in shared space can be excessed by all the threads
std::queue<std::pair<double, double>>  q;
q.push({a, b});
std::vector <bool> stat_all;
for(int i=0; i<33; i++){
    stat_all.push_back(false);
}


#pragma omp parallel
{   //cout<<"code runs "<< " total threads: "<<omp_get_num_threads() <<std::endl;


    double c, d; bool stat = false; double f_d; double f_c; bool check_loop = true;
        double temp;
        //Algorithm to divide the work in parallel:
        //double max_value; This is private to the specific thread or process

        while (q.size() !=0  || check_loop ) {

        #pragma omp critical
        {
            if(q.size() != 0){
                c = (q.front()).first;  
                d = (q.front()).second;  
                q.pop();
                stat = true;
                stat_all[omp_get_thread_num()] = true;
            }
            else{
                stat = false;
                stat_all[omp_get_thread_num()] = false;
            }
        }

        if (stat){
        #pragma omp critical
        {
        if (f(d) > max_value || f(c) > max_value){
            max_value = std::max(f(d), f(c));
        }
        }

        temp = (f(c) + f(d) + s*(d-c))/2;

        if (temp > (max_value + epsilon) && (d-c) > 0.01){
            //update the stack with pair values:
            #pragma omp critical
        {   //if(std::find)
            q.push({c, (c+d)/2});
            q.push({(c+d)/2, d});
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

        /*
        #pragma omp critical
        {
        cout<<"max_value: "<<max_value<< " c: "<<c<<" d: "<<d<< " queue size: "<< q.size() <<std::endl; 
        cout<<"Thread no: "<<omp_get_thread_num()<<std::endl;
        }  */
        
}

}
auto end_time = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> elapsed = end_time - start_time;
cout<<"The maximum value is: "<<max_value<<std::endl;
cout<<"The time taken to run the code is: "<<elapsed.count() <<std::endl;
    return 0;
}