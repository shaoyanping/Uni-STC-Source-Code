#include <_.h>
#include <msg.h>
#include <tmatrix/MMIO/mmio_highlevel.h>
#include <tmatrix/DataStructure/transform.h>

void test_BSR_to_MtxFile(const char*mtx_path)
{
    int is_symmetric;
    MatIndex m, n, nnz, bsr_m, bsr_n, bsr_nnz;
    MatIndex *row_ptr, *col_ind, *bsr_row_ptr, *bsr_col_ptr;
    MatValue *val, *bsr_val;

    mmio_allinone(mtx_path, &m, &n, &nnz, &is_symmetric, &row_ptr, &col_ind, &val);
    echo(info, "m = %d, n = %d, nnz = %d", m, n, nnz);

    CSR_to_BSR(m, n, nnz, row_ptr, col_ind, val, &bsr_m, &bsr_n, &bsr_nnz, &bsr_row_ptr, &bsr_col_ptr, &bsr_val);

    echo(info, "bsr_m = %d, bsr_n = %d, bsr_nnz = %d", bsr_m, bsr_n, bsr_nnz);

    for (int i = 0; i < bsr_m + 1; ++i) printf("%d ", bsr_row_ptr[i]);
    puts("");

    FILE*fp = fopen("bsr.mtx", "w");
    fprintf(fp, "%%%%MatrixMarket matrix coordinate real general\n");
    fprintf(fp, "%d %d %d\n", m, n, bsr_nnz << 8);
    for (int i = 0; i < bsr_m; ++i)
    {
        for (int j = bsr_row_ptr[i]; j < bsr_row_ptr[i + 1]; ++j)
        {
            for (int k = 0; k < 256; ++k)
            {
                fprintf(fp, "%d %d %lf\n", i * 16 + (k >> 4) + 1, bsr_col_ptr[j] * 16 + (k & 15) + 1, bsr_val[j << 8 | k]);
            }
        }
    }

    free(bsr_row_ptr);
    free(bsr_col_ptr);
    free(bsr_val);

    free(row_ptr);
    free(col_ind);
    free(val);
}