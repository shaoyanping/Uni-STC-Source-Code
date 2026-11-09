#pragma once
#include "./bitmap.h"
#include "./bit_x_bit.h"
#include <tmatrix/Utils/omp_utils.h>
#include <tmatrix/Utils/hash_bitmap.h>
#include <tmatrix/Utils/hash_key_val.h>
#include <tmatrix/DataStructure/TileMatrix.h>
#include "./smu_sim.h"

#include <vector>

MatIndex binary_find_index(MatIndex *col_ptr, MatIndex *val, MatIndex len, MatIndex col, MatIndex *idx)
{
    MatIndex left = 0, right = len - 1;
    while (left <= right)
    {
        MatIndex mid = (left + right) >> 1;
        if (col_ptr[mid] == col)
        {
            if (idx)
                *idx = mid;
            return val[mid];
        }
        else if (col_ptr[mid] < col)
            left = mid + 1;
        else
            right = mid - 1;
    }
    return -1;
}

void Tile_16_Bitmap_x_Bitmap(
    TileBitmap A_16_bitmap, TileBitmap *A_bitmap_ptr, // input A
    TileBitmap B_16_bitmap, TileBitmap *B_bitmap_ptr, // input B
    TileBitmap C_16_Bitmap, TileBitmap *C_bitmap_ptr  // output C
)
{
    TileBitmap idx[4] = {0, 1, 17, 273};

    while (A_16_bitmap)
    {
        TileBitmap Ak = TileBitmap_lowbit(A_16_bitmap);
        A_16_bitmap ^= Ak;

        TileIndex Actz = TileBitmap_ctz(Ak);
        TileBitmap cur_A_4_bitmap = *A_bitmap_ptr++;
        if (!cur_A_4_bitmap)
            continue;

        TileIndex row = Actz & 0xc, Acol = Actz & 3, Bj = TileBitmap_popcount(B_16_bitmap & (0xf * idx[Acol]));
        TileIndex B_row_map = (B_16_bitmap & (0xf << (Acol << 2))) >> (Acol << 2); // only using 4 bits

        while (B_row_map)
        {
            TileIndex Bk = TileBitmap_lowbit(B_row_map);
            B_row_map ^= Bk;

            TileIndex col = TileBitmap_ctz(Bk);

            TileBitmap cur_B_4_bitmap = B_bitmap_ptr[Bj++];
            if (!cur_B_4_bitmap)
                continue;

            C_bitmap_ptr[row | col] |= TileBitmap_x(cur_A_4_bitmap, cur_B_4_bitmap);
        }
    }
}

void update_sym_hash(TileBitmap a16, TileBitmap *a4, TileBitmap b16, TileBitmap*b4, hash_key_bitmap_t*table, unsigned int size, MatIndex Bcol)
{
    hash_key_bitmap_val_t _tmp;
    _tmp.val16 = TileBitmap_x(a16, b16);
    
    if (_tmp.val16)
    {
        TileBitmap x = _tmp.val16;
        memset(_tmp.val4, 0, 32);
        Tile_16_Bitmap_x_Bitmap(a16, a4, b16, b4, _tmp.val16, _tmp.val4);

        while (x)
        {
            TileBitmap k = TileBitmap_lowbit(x);
            TileIndex ctz = TileBitmap_ctz(k);

            if (_tmp.val4[ctz] == 0) _tmp.val16 ^= k;

            x ^= k;
        }

        if (_tmp.val16)
            insert_key_bitmap_hash(table, size, Bcol, _tmp);
    }
}

TileMatrix *TileMatrix_GEMM_Sym(TileMatrix *A, TileMatrix *B)
{
    TileMatrix *C = (TileMatrix *)malloc(sizeof(TileMatrix));
    C->meta_m = A->meta_m;
    C->meta_n = B->meta_n;
    C->_m = A->_m;
    C->_n = B->_n;

    MatIndex max_threads = omp_get_max_threads();
    C->tile_row_ptr = (MatIndex *)calloc((C->_m + 1), sizeof(MatIndex));
    unsigned int _size = A->_n;
    hash_key_bitmap_t **all_table = (hash_key_bitmap_t **)malloc(sizeof(hash_key_bitmap_t*) * max_threads);

    #pragma omp parallel for
    for (MatIndex i = 0; i < max_threads; ++i)
    {
        all_table[i] = (hash_key_bitmap_t*)malloc(sizeof(hash_key_bitmap_t));
        all_table[i]->key = (MatIndex *)malloc(sizeof(MatIndex) * _size);
        all_table[i]->val = (hash_key_bitmap_val_t*) malloc(sizeof(hash_key_bitmap_val_t) * _size);
    }

    hash_key_bitmap_t **seg_tmp_val = (hash_key_bitmap_t **)malloc(sizeof(hash_key_bitmap_t *) * A->_m);
    // mempool mem;
    // mempool_init(&mem, 1024ull * 1024 * 1024 * 20); // 20GB
    
    // echo(debug, "Step 1: building hash structure");
#pragma omp parallel for //* Step 1: 计算16 * 16的非零元数
    for (MatIndex Ai = 0; Ai < A->_m; ++Ai)
    {
        MatIndex tid = omp_get_thread_num();
        hash_key_bitmap_t *table = all_table[tid];

        memset(table->key, -1, sizeof(MatIndex) * _size);
        memset(table->val,  0, sizeof(hash_key_bitmap_val_t) * _size);

        for (MatIndex Aj = A->tile_row_ptr[Ai]; Aj < A->tile_row_ptr[Ai + 1]; ++Aj)
        {
            MatIndex cur_A_16_col = A->tile_col_ptr[Aj];
            TileBitmap cur_A_16_bitmap = A->tile_16_bitmap_ptr[Aj];

            for (MatIndex Bj = B->tile_row_ptr[cur_A_16_col]; Bj < B->tile_row_ptr[cur_A_16_col + 1]; ++Bj)
            {
                update_sym_hash(
                    cur_A_16_bitmap, A->tile_4_bitmap_ptr + A->tile_val_map[Aj], 
                    B->tile_16_bitmap_ptr[Bj], B->tile_4_bitmap_ptr + B->tile_val_map[Bj], 
                    table, _size, B->tile_col_ptr[Bj]
                );
            }
        }
        C->tile_row_ptr[Ai + 1] = gather_hash(table, _size);
        if (C->tile_row_ptr[Ai + 1]) {
            // quicksort_key_val(table->key, table->val, 0, C->tile_row_ptr[Ai + 1] - 1);
            seg_tmp_val[Ai] = (hash_key_bitmap_t*) malloc(sizeof(hash_key_bitmap_t));
            seg_tmp_val[Ai]->key = (MatIndex *)malloc(sizeof(MatIndex) * C->tile_row_ptr[Ai + 1]);
            seg_tmp_val[Ai]->val = (hash_key_bitmap_val_t*) malloc(sizeof(hash_key_bitmap_val_t) * C->tile_row_ptr[Ai + 1]);

            memcpy(seg_tmp_val[Ai]->key, table->key, sizeof(MatIndex) * C->tile_row_ptr[Ai + 1]);
            memcpy(seg_tmp_val[Ai]->val, table->val, sizeof(hash_key_bitmap_val_t) * C->tile_row_ptr[Ai + 1]);
        } else seg_tmp_val[Ai] = NULL;
    }
    omp_inclusive_scan(C->tile_row_ptr + 1, C->_m);
    free(all_table);

    C->_nnz = C->tile_row_ptr[C->_m];

    C->tile_col_ptr = (MatIndex *)malloc(C->_nnz * sizeof(MatIndex));
    C->tile_16_bitmap_ptr = (TileBitmap *)malloc(C->_nnz * sizeof(TileBitmap));
    C->tile_val_map = (MatIndex *)malloc((C->_nnz + 1) * sizeof(MatIndex));
    C->tile_off_map = (MatIndex *)malloc((C->_nnz + 1) * sizeof(MatIndex));

    C->tile_val_map[0] = C->tile_off_map[0] = 0;

    // echo(debug, "Step 2: Sequential Step 1");
#pragma omp parallel for //* Step 2: 构建16*16位置信息
    for (MatIndex Ai = 0; Ai < A->_m; ++Ai)
    {
        hash_key_bitmap_t *table = seg_tmp_val[Ai];
        if (!table) continue;

        unsigned int _len = C->tile_row_ptr[Ai + 1] - C->tile_row_ptr[Ai];
        memcpy(C->tile_col_ptr + C->tile_row_ptr[Ai], table->key, _len * sizeof(MatIndex));

        for (MatIndex Cj = C->tile_row_ptr[Ai], idx = 0; Cj < C->tile_row_ptr[Ai + 1]; ++Cj, ++idx)
        {
            C->tile_16_bitmap_ptr[Cj] = table->val[idx].val16;
            C->tile_val_map[Cj + 1] = TileBitmap_popcount(table->val[idx].val16);
        }
    }
    omp_inclusive_scan(C->tile_val_map + 1, C->_nnz);

    MatIndex level_1_nnz = C->tile_val_map[C->_nnz];
    C->tile_4_bitmap_ptr = (TileBitmap *)malloc(level_1_nnz * sizeof(TileBitmap));
    C->tile_4_offmap_ptr = (TileIndex *)malloc(level_1_nnz * sizeof(TileIndex));

    // echo(debug, "Step 3: Sequential Step 2, l1nnz: %d", level_1_nnz);
    #pragma omp parallel for //* Step 3: 构建两级索引
    for (MatIndex Ai = 0; Ai < A->_m; ++Ai)
    {
        hash_key_bitmap_t *table = seg_tmp_val[Ai];
        if (!table) continue;

        for (MatIndex Cj = C->tile_row_ptr[Ai], iCj = 0; Cj < C->tile_row_ptr[Ai + 1]; ++Cj, ++iCj)
        {
            MatIndex val_map_Cj = C->tile_val_map[Cj];
            TileBitmap *dst = C->tile_4_bitmap_ptr + val_map_Cj, *src = table->val[iCj].val4;
            
            #pragma unroll
            for (TileIndex i = 0, idx=0; i < 16; ++i)
            {
                if (src[i]) dst[idx++] = src[i];
            }

            TileBitmap cur_C_16_bitmap = C->tile_16_bitmap_ptr[Cj];
            TileIndex *cur_C_4_valmap_ptr = C->tile_4_offmap_ptr + val_map_Cj;

            cur_C_4_valmap_ptr[0] = 0;

            TileIndex nnz = TileBitmap_popcount(cur_C_16_bitmap);
            for (TileIndex i = 1; i < nnz; ++i)
            {
                cur_C_4_valmap_ptr[i] = TileBitmap_popcount(dst[i - 1]) + cur_C_4_valmap_ptr[i - 1];
            }
            C->tile_off_map[Cj + 1] = cur_C_4_valmap_ptr[nnz - 1] + TileBitmap_popcount(dst[nnz - 1]);
        }
        free_hash(table);
    }
    free(seg_tmp_val);
    // mempool_free(&mem);

    omp_inclusive_scan(C->tile_off_map + 1, C->_nnz);
    C->meta_nnz = C->tile_off_map[C->_nnz];
    // C->tile_val_ptr = (MatValue *)calloc(C->meta_nnz, sizeof(MatValue));
    // echo(debug, "level 3 nnz: %d", C->meta_nnz);
    return C;
}
