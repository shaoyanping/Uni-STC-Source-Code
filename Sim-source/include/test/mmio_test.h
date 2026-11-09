#include <msg.h>
#include <tmatrix/MMIO/mmio_highlevel.h>

void test_mmio(const char* filepath)
{
    if (!filepath)
    {
        echo(error, "filepath is not specified");
        return;
    }
    int m, n, nnz, is_symmetric;
    int *row_ptr, *col_ind;
    double *val;

    mmio_allinone(filepath, &m, &n, &nnz, &is_symmetric, &row_ptr, &col_ind, &val);
    echo(info, "m = %d, n = %d, nnz = %d", m, n, nnz);

    for (int i = 0; i < m + 1; ++i) printf("%d ", row_ptr[i]);
    puts("");
    for (int i = 0; i < nnz; ++i) printf("%d ", col_ind[i]);
    puts("");
    for (int i = 0; i < nnz; ++i) printf("%lf ", val[i]);
    puts("");
    
    free(row_ptr);
    free(col_ind);
    free(val);
}