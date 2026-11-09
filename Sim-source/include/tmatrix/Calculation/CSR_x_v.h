#pragma once
#include <_.h>

void CSR_SpMV(MatIndex m, MatIndex n, MatIndex nnz, MatIndex *row_ptr, MatIndex *col_ptr, MatValue *val_ptr, MatValue *v, MatValue *res)
{
#pragma omp parallel for
    for (MatIndex i = 0; i < m; ++i)
    {
        MatValue sum = 0;
        for (MatIndex j = row_ptr[i]; j < row_ptr[i + 1]; ++j)
        {
            sum += val_ptr[j] * v[col_ptr[j]];
        }
        res[i] = sum;
    }
}