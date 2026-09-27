#include <iostream>
#include <chrono>  //Used for time, accurate to the nanosecond
#include <cstdlib> //Contains rand and etc
using namespace std;

#define tile_size 16

void Matrix_Genrator(float * a, int n){

    auto seed = chrono::high_resolution_clock::now().time_since_epoch().count();   

    srand(seed);

    for(int i=0; i<(n*n); i++){

        a[i] = (rand() % 21) - 10;

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



int main(){

    int n;
    cout<<"Enter the size of the square matrix: ";
    cin>>n;

    float * a = new float[n*n];
    float * c = new float[n*n];

    Matrix_Genrator(a,n);

    float * d_a, *d_b, *d_c;
    auto size = sizeof(float)*(n*n);

    cudaMalloc(&d_a,size);
    cudaMalloc(&d_b,size);
    cudaMalloc(&d_c,size);

    cudaMemcpy(d_a,a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,a,size,cudaMemcpyHostToDevice);

    dim3 Block (tile_size,tile_size);     

    int blocks = (n + tile_size - 1)/tile_size;
    dim3 Grid (blocks, blocks);

    
    double total_time = 0.0;


    for(int run=0; run<6; run++){

        auto start = chrono::high_resolution_clock::now();         //Used to check how long the algorithm takes

        Tiled_Multi<<<Grid,Block>>>(d_a,d_b,d_c,n);                    
        cudaDeviceSynchronize();

        auto end = chrono::high_resolution_clock::now();

        double time_taken = chrono::duration<double>(end-start).count();
        if(run!=0){
        total_time += time_taken;
    }

        if(run == 0){
            cout<<"This is a warmup run ";
        }
        
        cout<<"Run "<<run+1<<": "<<time_taken<<" seconds\n";

    }


    double average_time = total_time / 5.0;
    
    cout<<"Average time: "<<average_time<<" seconds\n";

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);



    for(int i=0; i<5; i++){

        for(int j=0; j<5; j++){

            cout<<c[i*n + j]<<" ";

        }

        cout<<"\n";

    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    delete[] a;
    delete[] c;

}