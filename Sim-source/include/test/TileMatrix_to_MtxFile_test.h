#pragma once
#include <msg.h>
#include <tmatrix/DataStructure/transform.h>
#include <tmatrix/Calculation/TileMatrix_x_v.h>
#include <tmatrix/MMIO/mmio_highlevel.h>

void test_TileMatrix_to_MtxFile(const char *from_mtx_path)
{
    MatIndex m, n, nnz, isSymmetric;
    MatIndex *row_ptr, *col_ptr;
    MatValue *val_ptr;

    mmio_allinone(from_mtx_path, &m, &n, &nnz, &isSymmetric, &row_ptr, &col_ptr, &val_ptr);
    TileMatrix *t = CSR_to_TileMatrix(m, n, nnz, row_ptr, col_ptr, val_ptr);

    {
        // puts("tile_row_ptr");
        // for (MatIndex i = 0; i < t->_m + 1; ++i)
        // {
        //     printf("%d ", t->tile_row_ptr[i]);
        // }
        // putchar('\n');
        // puts("tile_col_ptr");
        // for (MatIndex i = 0; i < t->_nnz; ++i)
        // {
        //     printf("%d ", t->tile_col_ptr[i]);
        // }
        // putchar('\n');
        // puts("tile_val_map");
        // for (MatIndex i = 0; i < t->_nnz + 1; ++i)
        // {
        //     printf("%d ", t->tile_val_map[i]);
        // }
        // putchar('\n');
        // puts("tile_16_bitmap_ptr");
        // for (MatIndex i = 0; i < t->_nnz; ++i)
        // {
        //     printf("0x%x ", t->tile_16_bitmap_ptr[i]);
        // }
        // putchar('\n');
        // puts("tile_row_ptr_16");
        // for (MatIndex i = 0; i < t->_nnz * 5; ++i)
        // {
        //     printf("%d ", t->tile_row_ptr_16[i]);
        // }
        // putchar('\n');
        // puts("tile_col_ptr_16");

        MatIndex layer_1_val_nnz = t->tile_val_map[t->_nnz];

        // for (MatIndex i = 0; i < layer_1_val_nnz; ++i)
        // {
        //     printf("%d ", t->tile_col_ptr_16[i]);
        // }
        // putchar('\n');
        // puts("tile_row_map_16");
        // for (MatIndex i = 0; i < layer_1_val_nnz + 1; ++i)
        // {
        //     printf("%d ", t->tile_row_map_16[i]);
        // }
        // putchar('\n');
        // puts("tile_col_map_16");
        // for (MatIndex i = 0; i < layer_1_val_nnz + 1; ++i)
        // {
        //     printf("%d ", t->tile_col_map_16[i]);
        // }
        // putchar('\n');
        // puts("tile_val_map_16");
        // for (MatIndex i = 0; i < layer_1_val_nnz + 1; ++i)
        // {
        //     printf("%d ", t->tile_val_map_16[i]);
        // }
        // putchar('\n');
        // puts("tile_4_bitmap_ptr");
        // for (MatIndex i = 0; i < layer_1_val_nnz; ++i)
        // {
        //     printf("0x%x ", t->tile_4_bitmap_ptr[i]);
        // }
        // putchar('\n');
        puts("tile_format_ptr");
        MatIndex cnt[8] = {0};
        for (MatIndex i = 0; i < layer_1_val_nnz; ++i)
        {
            // printf("%s ", TileSparseFormatName[t->tile_format_ptr[i]]);
            cnt[t->tile_format_ptr[i]]++;
        }

        for (MatIndex i = 0; i < 8; ++i)
            printf("%s: %d\n", TileSparseFormatName[i], cnt[i]);
        putchar('\n');

        // MatIndex layer_2_val_nnz = t->tile_val_map_16[layer_1_val_nnz];
        // MatIndex layer_2_row_nnz = t->tile_row_map_16[layer_1_val_nnz];
        // MatIndex layer_2_col_nnz = t->tile_col_map_16[layer_1_val_nnz];

        // printf("tile_row_ptr_4: %p\n", t->tile_row_ptr_4);
        // for (MatIndex i = 0; i < layer_2_row_nnz; ++i)
        // {
        //     if (t->tile_row_ptr_4[i])
        //         printf("[bold red]%d[/] ", t->tile_row_ptr_4[i]);
        //     else
        //         printf("%d ", t->tile_row_ptr_4[i]);
        // }
        // putchar('\n');
        // printf("tile_col_ptr_4: %p\n", t->tile_col_ptr_4);
        // for (MatIndex i = 0; i < layer_2_col_nnz; ++i)
        // {
        //     if (t->tile_col_ptr_4[i])
        //         printf("[bold red]%d[/] ", t->tile_col_ptr_4[i]);
        //     else
        //         printf("%d ", t->tile_col_ptr_4[i]);
        // }
        // putchar('\n');
        // printf("tile_val_ptr: %p\n", t->tile_val_ptr);
        // for (MatIndex i = 0; i < layer_2_val_nnz; ++i)
        // {
        //     if (t->tile_val_ptr[i] != 0)
        //         printf("[bold red]%.0f[/] ", t->tile_val_ptr[i]);
        //     else
        //         printf("%.0f ", t->tile_val_ptr[i]);
        // }
        // putchar('\n');
    }

    free(row_ptr);
    free(col_ptr);
    free(val_ptr);

    TileMatrix_to_MtxFile(t, "dist/test_TileMatrix.mtx");

    destroy_TileMatrix(t);
}