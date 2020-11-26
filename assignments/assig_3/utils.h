// CPU untils
#include <iostream>

void mat_mul_cpu(int N, float*C, float*A, float*B){
    int row, col, i;
    for (row = 0; row < N;++row){
        for (col = 0; col < N;++col){
            C[row * N + col] = 0.0f;
            for (i = 0; i < N;++i){
                C[row * N + col] += A[row * N + i] * B[i * N + col];
            }
        }
    }
}

bool check_result(int N, float *A, float *B){
    int row, col;
    for (row = 0; row < N;++row){
        for (col = 0; col < N;++col){
            if (abs(A[row * N + col] - B[row * N + col]) > 0.01f)
            {
                return false;
            }
        }
    }
    return true;
}

void print_mat(int N, float* mat){
    for (int row = 0; row < N;++row){
        for (int col = 0; col < N;++col){
            std::cout << mat[row*N + col]<<' ';
        }
        std::cout << std::endl;
    }
}