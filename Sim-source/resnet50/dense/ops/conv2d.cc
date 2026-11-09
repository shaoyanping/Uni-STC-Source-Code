#include "conv2d.h"

#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#include <tmatrix/DataStructure/transform.h>
#include <tmatrix/Calculation/TileMatrix_x_m.h>

task_results sum128, sum64;

// 这段代码定义了一个自定义的二维卷积（convolution）函数
// my_conv2d，该函数执行卷积操作，通常用于图像处理和神经网络中 my_conv2d
// 函数接收一张图像（img）和卷积核权重（weight），以及各种与卷积相关的参数，如输入输出尺寸、通道数、卷积核大小、步长和填充。
// 函数内部通过嵌套循环对图像进行卷积操作，并将结果存储在新分配的内存中。

MatValue* im2col(float*img, int hi, int wi, int ci, int kernel, int stride, int pad) {
    int ho = (hi + 2 * pad - kernel) / stride + 1;
    int wo = (wi + 2 * pad - kernel) / stride + 1;
    int col_size = kernel * kernel * ci;
    int out_size = ho * wo * col_size;

    MatValue* col = (MatValue*)calloc(out_size, sizeof(MatValue));
    for (int y = 0; y < ho; ++y) {
        for (int x = 0; x < wo; ++x) {
            // Iterate over the kernel dimensions
            for (int ky = 0; ky < kernel; ++ky) {
                for (int kx = 0; kx < kernel; ++kx) {
                    // Iterate over the input channels
                    for (int c = 0; c < ci; ++c) {
                        // Calculate the input image coordinates
                        int in_y = y * stride + ky - pad;
                        int in_x = x * stride + kx - pad;

                        // Check if the coordinates are within bounds
                        if (in_y >= 0 && in_y < hi && in_x >= 0 && in_x < wi) {
                            // Calculate the input and output matrix indices
                            int in_index = (c * hi + in_y) * wi + in_x;
                            int out_index = ((y * wo + x) * kernel + ky) * kernel * ci + kx * ci + c;
                            col[out_index] = img[in_index];
                        }
                    }
                }
            }
        }
    }

    return col;
}

void dense_matrix_to_csr(MatValue*col, int m, int n, MatIndex**row_ptr, MatIndex**col_ptr, MatValue**val_ptr) {
    int nnz = 0;
    #pragma omp parallel for reduction(+:nnz)
    for (int i = 0; i < m * n; ++i) if (col[i] != 0) nnz++;
    printf("Sparsity: %.2f%%\n", nnz * 1.0 / m / n * 100);

    *row_ptr = (MatIndex*)calloc(m + 1, sizeof(MatIndex));
    *col_ptr = (MatIndex*)malloc(nnz * sizeof(MatIndex));
    *val_ptr = (MatValue*)malloc(nnz * sizeof(MatValue));

    (*row_ptr)[0] = 0;

    int nnz_idx = 0;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            int idx = i * n + j;
            if (col[idx] != 0)
            {
                (*col_ptr)[nnz_idx] = j;
                (*val_ptr)[nnz_idx] = col[idx];
                nnz_idx++;
            }
        }
        (*row_ptr)[i + 1] = nnz_idx;
    }
}


void dense_matrix_to_csr_pruning(float*col, int m, int n, MatIndex**row_ptr, MatIndex**col_ptr, MatValue**val_ptr, float sparsity) {
    int nnz = 0;
    #pragma omp parallel for reduction(+:nnz)
    for (int i = 0; i < m * n; ++i) if (col[i] != 0) nnz++;

    *row_ptr = (MatIndex*)calloc(m + 1, sizeof(MatIndex));
    *col_ptr = (MatIndex*)malloc(nnz * sizeof(MatIndex));
    *val_ptr = (MatValue*)malloc(nnz * sizeof(MatValue));

    (*row_ptr)[0] = 0;

    int nnz_idx = 0;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            int idx = i * n + j;
            if (col[idx] != 0 and (rand() % 100) < sparsity * 100)
            {
                (*col_ptr)[nnz_idx] = j;
                (*val_ptr)[nnz_idx] = col[idx];
                nnz_idx++;
            }
        }
        (*row_ptr)[i + 1] = nnz_idx;
    }
    *col_ptr = (MatIndex*)realloc(*col_ptr, nnz_idx * sizeof(MatIndex));
    *val_ptr = (MatValue*)realloc(*val_ptr, nnz_idx * sizeof(MatValue));

    printf("Sparsity: %.2f%%\n", nnz_idx * 1.0 / m / n * 100);
}


float* my_conv2d(float* img,
                 float* weight,
                 int hi,
                 int wi,
                 int& ho,
                 int& wo,
                 int ci,
                 int co,
                 int kernel,
                 int stride,
                 int pad,
                 bool is_free_img) {
    // 计算输出图像的高度和宽度
    ho = (hi + 2 * pad - kernel) / stride + 1;
    wo = (wi + 2 * pad - kernel) / stride + 1;
    // 分配输出图像的内存
    float* out = (float*)malloc(ho * wo * co * sizeof(float));
    MatValue* col = im2col(img, hi, wi, ci, kernel, stride, pad);
    
    MatIndex* row_ptr_i, *row_ptr_w;
    MatIndex* col_ptr_i, *col_ptr_w;
    MatValue* val_ptr_i, *val_ptr_w;

    dense_matrix_to_csr(col, ho * wo, kernel * kernel * ci, &row_ptr_i, &col_ptr_i, &val_ptr_i);
    TileMatrix* tcol = CSR_to_TileMatrix(ho * wo, kernel * kernel * ci, row_ptr_i[ho * wo], row_ptr_i, col_ptr_i, val_ptr_i, "", false, true);

    // TileMatrix* C = TileMatrix_SPMM_Sym_v2(tcol, co);
    // sum128 += TileMatrix_SPMM_Num_Sim<128>(tcol, C);
    // sum64 += TileMatrix_SPMM_Num_Sim<64>(tcol, C);
    sum128 += TileMatrix_SPMM_Num_Sim<128>(tcol, (co + 15) / 16);
    sum64 += TileMatrix_SPMM_Num_Sim<64>(tcol, (co + 15) / 16);

    free(row_ptr_i); 
    free(col_ptr_i); 
    free(val_ptr_i); 
    destroy_TileMatrix(tcol, true);
    // destroy_TileMatrix(C, true);
    
    sum128.ds_mac = sum128.mpnz * 1.0 / 128 / sum128.ds_cycle * 100.0;
    sum128.rm_mac = sum128.mpnz * 1.0 / 128 / sum128.rm_cycle * 100.0;
    sum128.smu_mac = sum128.mpnz * 1.0 / 128 / sum128.smu_cycle * 100.0;
    sum128.smu_speedup = std::max(sum128.ds_cycle, sum128.rm_cycle) * 1.0 / sum128.smu_cycle;
    sum64.ds_mac = sum64.mpnz * 1.0 / 64 / sum64.ds_cycle * 100.0;
    sum64.rm_mac = sum64.mpnz * 1.0 / 64 / sum64.rm_cycle * 100.0;
    sum64.smu_mac = sum64.mpnz * 1.0 / 64 / sum64.smu_cycle * 100.0;
    sum64.smu_speedup = std::max(sum64.ds_cycle, sum64.rm_cycle) * 1.0 / sum64.smu_cycle;

    // printf(
    //     "%lu,%.2lf,%lu,%.2lf,%lu,%.2lf,%.2lf,%.2lf,%.2lf,%.2lf,%lu,%.2lf,%lu,%.2lf,%lu,%.2lf,%.2lf,%.2lf,%.2lf,%.2lf,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu\n",
    //     sum128.ds_cycle, sum128.ds_mac, sum128.rm_cycle, sum128.rm_mac, sum128.smu_cycle, sum128.smu_mac,
    //     sum128.ds_energy, sum128.rm_energy, sum128.smu_energy, sum128.smu_speedup,
    //     sum64.ds_cycle, sum64.ds_mac, sum64.rm_cycle, sum64.rm_mac, sum64.smu_cycle, sum64.smu_mac,
    //     sum64.ds_energy, sum64.rm_energy, sum64.smu_energy, sum64.smu_speedup,
    //     sum128.ds_a_ntr, sum128.ds_b_ntr, sum128.rm_a_ntr, sum128.rm_b_ntr, sum128.smu_a_ntr, sum128.smu_b_ntr,
    //     sum64.ds_a_ntr, sum64.ds_b_ntr, sum64.rm_a_ntr, sum64.rm_b_ntr, sum64.smu_a_ntr, sum64.smu_b_ntr);

    printf(
        "%lu,%lu,%lu,%.2lf,%lu,%.2lf,%lu,%.2lf,%.2lf,%.2lf,%.2lf,%.2lf,%lu,%.2lf,%lu,%.2lf,%lu,%.2lf,%.2lf,%.2lf,%.2lf,%.2lf,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%.2lf,%.2lf,%.2lf,%.2lf,%.2lf,%.2lf,%.2lf,%.2lf\n",
        sum128.mp16,sum128.mpnz,
        sum128.ds_cycle, sum128.ds_mac, sum128.rm_cycle, sum128.rm_mac, sum128.smu_cycle, sum128.smu_mac, 
        sum128.ds_energy, sum128.rm_energy, sum128.smu_energy, sum128.smu_speedup, 
        sum64.ds_cycle, sum64.ds_mac, sum64.rm_cycle, sum64.rm_mac, sum64.smu_cycle, sum64.smu_mac,
        sum64.ds_energy, sum64.rm_energy, sum64.smu_energy, sum64.smu_speedup,
        sum128.ds_mt125, sum128.ds_mt25, sum128.ds_mt50, sum128.ds_mt75, sum128.ds_mt100, sum128.rm_mt125, sum128.rm_mt25, sum128.rm_mt50, sum128.rm_mt75, sum128.rm_mt100, sum128.smu_mt125, sum128.smu_mt25, sum128.smu_mt50, sum128.smu_mt75, sum128.smu_mt100,
        sum64.ds_mt125, sum64.ds_mt25, sum64.ds_mt50, sum64.ds_mt75, sum64.ds_mt100, sum64.rm_mt125, sum64.rm_mt25, sum64.rm_mt50, sum64.rm_mt75, sum64.rm_mt100, sum64.smu_mt125, sum64.smu_mt25, sum64.smu_mt50, sum64.smu_mt75, sum64.smu_mt100,
        sum128.ds_a_ntr, sum128.ds_b_ntr, sum128.ds_c_ntr, sum128.rm_a_ntr, sum128.rm_b_ntr, sum128.rm_c_ntr, sum128.smu_a_ntr, sum128.smu_b_ntr, sum128.smu_c_ntr, sum64.ds_a_ntr, sum64.ds_b_ntr, sum64.ds_c_ntr, sum64.rm_a_ntr, sum64.rm_b_ntr, sum64.rm_c_ntr, sum64.smu_a_ntr, sum64.smu_b_ntr, sum64.smu_c_ntr,
        sum128.ds_nt125, sum128.ds_nt25, sum128.ds_nt50, sum128.ds_nt75, sum128.ds_nt100, sum128.rm_nt125, sum128.rm_nt25, sum128.rm_nt50, sum128.rm_nt75, sum128.rm_nt100, sum128.smu_nt125, sum128.smu_nt25, sum128.smu_nt50, sum128.smu_nt75, sum128.smu_nt100,
        sum64.ds_nt125, sum64.ds_nt25, sum64.ds_nt50, sum64.ds_nt75, sum64.ds_nt100, sum64.rm_nt125, sum64.rm_nt25, sum64.rm_nt50, sum64.rm_nt75, sum64.rm_nt100, sum64.smu_nt125, sum64.smu_nt25, sum64.smu_nt50, sum64.smu_nt75, sum64.smu_nt100,
        sum128.rm_c_nut, sum128.smu_a_nut, sum128.smu_b_nut, sum128.smu_c_nut, 
        sum64.rm_c_nut, sum64.smu_a_nut, sum64.smu_b_nut, sum64.smu_c_nut
    );

    // 遍历输出通道
    for (int co_idx = 0; co_idx < co; co_idx++) {
        // 遍历输出图像的高度
        for (int ho_idx = 0; ho_idx < ho; ho_idx++) {
            const int in_h_origin = ho_idx * stride - pad;  // 计算输入图像的起始高度位置
            // 遍历输出图像的宽度
            for (int wo_idx = 0; wo_idx < wo; wo_idx++) {
                const int in_w_origin = wo_idx * stride - pad;  // 计算输入图像的起始宽度位置
                const int filter_h_start = std::max(0, -in_h_origin);  // 计算卷积核在高度方向的起始位置
                const int filter_w_start = std::max(0, -in_w_origin);  // 计算卷积核在宽度方向的起始位置
                const int filter_h_end =
                    std::min(kernel, hi - in_h_origin);  // 计算卷积核在高度方向的结束位置
                const int filter_w_end =
                    std::min(kernel, wi - in_w_origin);  // 计算卷积核在宽度方向的结束位置
                float acc = 0;                           // 初始化乘累加结果

                // 遍历卷积核的高度
                for (int kh_idx = filter_h_start; kh_idx < filter_h_end; kh_idx++) {
                    const int hi_index = in_h_origin + kh_idx;  // 计算当前卷积核高度的索引
                    // 遍历卷积核的宽度
                    for (int kw_idx = filter_w_start; kw_idx < filter_w_end; kw_idx++) {
                        const int wi_index = in_w_origin + kw_idx;  // 计算当前卷积核宽度的索引
                        // 遍历输入通道
                        for (int ci_ = 0; ci_ < ci; ci_++) {
                            auto in_data = img[hi_index * wi * ci + wi_index * ci + ci_];  // 获取输入图像的数据
                            auto weight_data = weight[co_idx * kernel * kernel * ci + kh_idx * kernel * ci +
                                                      kw_idx * ci + ci_];  // 获取权重数据
                            acc += in_data * weight_data;  // 执行卷积运算，本质就是乘累加运算
                        }
                    }
                }
                out[ho_idx * wo * co + wo_idx * co + co_idx] = acc;  // 存储卷积结果
            }
        }
    }

    if (is_free_img) {
      free(img);
    }
    free(weight);
    return out;  // 返回卷积后的输出图像
}
