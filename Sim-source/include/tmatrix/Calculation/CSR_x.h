#pragma once
#include <_.h>
#include <tmatrix/Utils/hash_bitmap.h>
#include <tmatrix/Utils/hash_key_val.h>
#include <tmatrix/Utils/omp_utils.h>

void CSR_GEMM(
    MatIndex m, MatIndex n, MatIndex k, MatIndex nnzA, MatIndex nnzB, MatIndex *nnzC,
    MatIndex *row_ptr_A, MatIndex *col_ptr_A, MatValue *val_ptr_A,
    MatIndex *row_ptr_B, MatIndex *col_ptr_B, MatValue *val_ptr_B,
    MatIndex **row_ptr_C, MatIndex **col_ptr_C, MatValue **val_ptr_C)
{
    Timer t1, t2, t3;
    MatIndex *_row_ptr_C = (MatIndex *)calloc(m + 1, sizeof(MatIndex));

    const unsigned int bhash_size = (k + 63) >> 6;
    BHash *all_table = (BHash *)malloc((bhash_size * omp_get_max_threads()) * sizeof(BHash));

    timer_start(&t1);
#pragma omp parallel for
    for (MatIndex Ai = 0; Ai < m; ++Ai)
    {
        BHash *table = all_table + omp_get_thread_num() * bhash_size;
        memset(table, 0, bhash_size << 3);
        for (MatIndex Aj = row_ptr_A[Ai]; Aj < row_ptr_A[Ai + 1]; ++Aj)
        {
            MatIndex cur_A_col = col_ptr_A[Aj];
            for (MatIndex Bj = row_ptr_B[cur_A_col]; Bj < row_ptr_B[cur_A_col + 1]; ++Bj)
            {
                insert_table(table, col_ptr_B[Bj]);
            }
        }
        _row_ptr_C[Ai + 1] = count_table(table, bhash_size);
    }

    omp_inclusive_scan(_row_ptr_C + 1, m);
    timer_end(&t1);
    free(all_table);

    MatIndex _nnz = _row_ptr_C[m];
    echo(info, "CSR GEMM \"Symbolic\"\ttime: %f ms, nnz: %d", timer_duration(&t1), _nnz);
    MatIndex *_col_ptr_C = (MatIndex *)malloc(_nnz * sizeof(MatIndex));
    MatValue *_val_ptr_C = (MatValue *)calloc(_nnz, sizeof(MatValue));

//     timer_start(&t2);
// #pragma omp parallel for schedule(static, 128)
//     for (MatIndex Ai = 0; Ai < m; ++Ai)
//     {
//         unsigned int size = _row_ptr_C[Ai + 1] - _row_ptr_C[Ai];
//         if (!size)
//             continue;
//         hash_key_val_t *row_hash_table = create_key_val_hash(&size);
//         for (MatIndex Aj = row_ptr_A[Ai]; Aj < row_ptr_A[Ai + 1]; ++Aj)
//         {
//             MatIndex cur_A_col = col_ptr_A[Aj];
//             MatValue cur_A_val = val_ptr_A[Aj];
//             for (MatIndex Bj = row_ptr_B[cur_A_col]; Bj < row_ptr_B[cur_A_col + 1]; ++Bj)
//             {
//                 insert_key_val_hash(row_hash_table, size, col_ptr_B[Bj], cur_A_val * val_ptr_B[Bj]);
//             }
//         }
//         iter_key_val_hash(row_hash_table, size, _col_ptr_C + _row_ptr_C[Ai], _val_ptr_C + _row_ptr_C[Ai]); // unordered output
//         quicksort_key_val(_col_ptr_C + _row_ptr_C[Ai], _val_ptr_C + _row_ptr_C[Ai], 0, _row_ptr_C[Ai + 1] - _row_ptr_C[Ai] - 1);
//         free_hash(row_hash_table);
//     }
//     timer_end(&t2);
//     echo(info, "CSR GEMM \"Numerical\"\ttime: %f ms", timer_duration(&t2));

    *nnzC = _nnz;
    *row_ptr_C = _row_ptr_C;
    *col_ptr_C = _col_ptr_C;
    *val_ptr_C = _val_ptr_C;
}