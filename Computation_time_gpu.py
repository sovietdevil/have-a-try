import numpy as np
from matplotlib import pyplot as plt
import os
import time

#1. generate code for the CUDA matrix multiplication
for i in range(100, 5000, 100):
    code1 = f'''#include <iostream>
#include <chrono>
#include <fstream>

using namespace std;
using namespace chrono;

#define n {i}'''

    code2 = '''
float h_a[n][n]={0};
float h_b[n][n]={0};
float h_c[n][n]={0};

__global__ void mul(float *d_a, float *d_b, float *d_c, int N){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int j=blockIdx.y*blockDim.y+threadIdx.y;
    for(int p=0;p<N;p++){
        d_c[i*N + j] = d_c[i*N + j] + d_a[i*N + p]*d_b[p*N + j];
    }
}

int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    const float s= sizeof(float)*n*n;

    float *d_a;
    float *d_b;
    float *d_c;

    cudaMalloc((void**)&d_a,s);
    cudaMalloc((void**)&d_b,s);
    cudaMalloc((void**)&d_c,s);

    cudaMemcpy(d_a,h_a,s,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,s,cudaMemcpyHostToDevice);

    mul<<<dim3(1,1,1),dim3(n,n,1)>>>(d_a,d_b,d_c,n);

    cudaMemcpy(h_c,d_c,s,cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << milliseconds << endl;
    ofstream out;
    out.open("output_time.txt", ios::app);
    out << milliseconds << endl;
    out.close();
    return 0;
}
    '''
    code = code1 + code2
    with open(f"mat_mul{i}.cu", "w") as output_file:
        output_file.write(code)

    #then conduct program compling and execution
    os.system(f"nvcc -ccbin \"%CUDA_lcpath%\" -o mat_mul{i} mat_mul{i}.cu")

time.sleep(10)
for i in range(100, 5000, 100):
    os.system(f"mat_mul{i}.exe")

timing = []
with open("output_time.txt") as time_record:
    for i in time_record:
        timing.append(float(i[:-1]))
print(timing)
plt.plot(range(100, 5000, 100), timing)
plt.xlabel("size")
plt.ylabel("time/ms")
plt.title("Running time for GPU")
plt.show()
for i in range(100, 5000, 100):
    os.remove(f"mat_mul{i}.cu")
    os.remove(f"mat_mul{i}.lib")
    os.remove(f"mat_mul{i}.exp")
    os.remove(f"mat_mul{i}.exe")
os.remove(f"output_time.txt")