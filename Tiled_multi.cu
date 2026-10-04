#include <iostream>
#include <chrono>  //Used for time, accurate to the nanosecond
#include <random>
#include <algorithm>
using namespace std;

#define tile_size 16

void Matrix_Generator(float * a, int n){

    random_device rd;
    mt19937 gen(rd());

    uniform_int_distribution<int> dist(-10,10);

    for(int i=0;i<(n*n);i++){

        a[i] = dist(gen);


    }

}

__global__ void Parallel_Multi(float * a, float * b, float * c, int n){

    int row = threadIdx.y + (blockIdx.y * blockDim.y);
    int col = threadIdx.x + (blockIdx.x * blockDim.x);

    if(row<n && col <n){

        float sum = 0.0f;

        for(int i = 0;i<n; i++){

            sum += a[i + (row*n)] * b[col + (i*n)];


        }

        c[col + (row*n)] = sum;


    }


}



__global__ void Tiled_Multi(float * a,float * b, float * c, int n){

    
    __shared__ float tile_a [tile_size][tile_size];
    __shared__ float tile_b [tile_size][tile_size];  //here, each tile is loaded into shared memory that is faster than global

    
    int row = threadIdx.y + (blockIdx.y * tile_size);
    int col = threadIdx.x + (blockIdx.x * tile_size);       //row and col for c so basically solution
    
    float sum = 0.0f;

    int num_of_tiles = (n + tile_size - 1)/tile_size;

    
    for(int i=0;i<num_of_tiles;i++){

        int a_col = (i*tile_size) + threadIdx.x;
        int b_row = (i*tile_size) + threadIdx.y;

        if(row < n && a_col<n){

            tile_a[threadIdx.y][threadIdx.x] = a[a_col + (row * n)];    //because in memory, the matrix is stored in 1d
        }

        else{

            tile_a[threadIdx.y][threadIdx.x] = 0.0f;
        }


        if (col<n && b_row<n){

            tile_b[threadIdx.y][threadIdx.x] = b[(b_row*n) + col];
        }

        else{

            tile_b[threadIdx.y][threadIdx.x] = 0.0f;
        }


        __syncthreads();

        for(int j =0; j< tile_size; j++){

            sum += tile_a[threadIdx.y][j] * tile_b[j][threadIdx.x]; //for tile a, row number is const and for tile b, column number is const here
        
        }

        __syncthreads();
    
    }

    if(row<n && col<n){

        c[col + (row*n)] = sum;

    }



}


void tiled(float * d_a, float * d_b, float * d_c, float * c, int n){

    auto size = sizeof(float)*(n*n);    
    dim3 Block (tile_size,tile_size);     

    int blocks = (n + tile_size - 1)/tile_size;
    dim3 Grid (blocks, blocks);

    
    float total_time = 0.0;
        
        cudaEvent_t start, stop; //Used to check how long the algorithm takes
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

    for(int run=0; run<6; run++){

        
        cudaEventRecord(start);

        Tiled_Multi<<<Grid,Block>>>(d_a,d_b,d_c,n);  

        cudaEventRecord(stop);                  
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess){
            cout << "Kernel launch error: " << cudaGetErrorString(err) << endl;
            return;
        }

        float time_taken;
        cudaEventElapsedTime(&time_taken,start,stop);
        

        if(run == 0){
            cout<<"This is a warmup run: "<< time_taken<<" ms\n";
        }
        else{
            total_time += time_taken;
            cout<<"Run "<<run<<": "<<time_taken<<" ms\n";
        }

    

    }


    double average_time = total_time / 5.0;
    
    cout<<"Average time: "<<average_time<<" ms\n\n";

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);



    for(int i=0; i<min(5,n); i++){

        for(int j=0; j<min(5,n); j++){

            cout<<c[i*n + j]<<" ";

        }

        cout<<"\n";

    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);


}

void basic_parallel(float * d_a, float * d_b, float * d_c, float * c, int n){

    dim3 Block(16,16);
    int blocks = (n + 16 - 1) / 16;
    dim3 Grid(blocks,blocks);
    
    auto size = sizeof(float)*(n*n);  
    float total_time = 0.0f;

    cudaEvent_t start, stop; //Used to check how long the algorithm takes
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for(int run =0; run<6; run++){

        cudaEventRecord(start);

        Parallel_Multi<<<Grid,Block>>>(d_a,d_b,d_c,n);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess){
            cout << "Kernel launch error: " << cudaGetErrorString(err) << endl;
            return;
        }

        float time_taken;

        cudaEventElapsedTime(&time_taken,start,stop);

        if(run == 0){
            cout<<"This is a warmup run: "<< time_taken<<" ms\n";
        }

        else{
            total_time += time_taken;
            cout<<"Run " << run <<": "<<time_taken<<" ms"<<"\n";

        }

    }
    
    float avg = total_time/5;

    cout<<"Average time: "<<avg<<" ms\n\n";

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);



    for(int i=0; i<min(5,n); i++){

        for(int j=0; j<min(5,n); j++){

            cout<<c[i*n + j]<<" ";

        }

        cout<<"\n";

    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);


}


void cpu_multi(float * a, float * b, float * c, int n){

    for(int i=0; i<n; i++){

        for(int j=0; j<n; j++){

            float sum = 0.0f;

            for(int k=0; k<n; k++){

                sum += a[i*n + k] * b[k*n + j];

            }

            c[i*n + j] = sum;

        }

    }

}



int main(){

    int n;
    cout<<"Enter the size of the square matrix: ";
    cin>>n;

    float * a = new float[n*n];
    
    float * c = new float[n*n];

    Matrix_Generator(a,n);
    

    float * d_a, *d_b, *d_c;
    auto size = sizeof(float)*(n*n);

    cudaMalloc(&d_a,size);
    cudaMalloc(&d_b,size);
    cudaMalloc(&d_c,size);

    cudaMemcpy(d_a,a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,a,size,cudaMemcpyHostToDevice);

    cout<<"\nTiled Multiplication:\n";
    tiled(d_a,d_b,d_c,c,n);

    cout<<"\nBasic Parallel Multiplication:\n";  
    basic_parallel(d_a,d_b,d_c,c,n);

    cout<<"\nCPU Multiplication:\n";

    cpu_multi(a,a,c,n);
    double total_cpu_time = 0.0;

    for(int run=0; run<5; run++){

        auto start = chrono::high_resolution_clock::now();

        cpu_multi(a,a,c,n);

        auto stop = chrono::high_resolution_clock::now();

        double time_taken = chrono::duration<double,milli>(stop-start).count();

        total_cpu_time += time_taken;

        cout<<"Run "<<run+1<<": "<<time_taken<<" ms\n";

    }
    double average_time = total_cpu_time/5;
    cout<<"Average cpu time: "<< average_time<<" ms\n";

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    delete[] a;
    delete[] c;



}
