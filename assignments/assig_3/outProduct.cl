__kernel void outProduct(const int N, __global float *C, __global float *A, __global float* B){
    int k;
    // size_t id = get_global_id(0);
    int i = get_global_id(0);
    int j = get_global_id(1);
    float temp = 0.0f;
    // printf("%d\n", get_global_id(1));
    for (k=0;k<N;++k){
        temp += A[i*N + k] * B[k*N + j];
    }
    C[i*N + j] += temp;
}