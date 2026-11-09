#pragma once

#include "../TileMatrix.h"
#include "tmatrix/Utils/omp_utils.h"
#include "tmatrix/Utils/hash_bitmap.h"
#include "tmatrix/Calculation/bitmap.h"
#include <msg.h>

void _bitmap_to_bitmap(TileBitmap bitmap, MatValue *val, const MatValue *dns_val)
{
    while (bitmap)
    {
        TileBitmap k = TileBitmap_lowbit(bitmap);
        TileIndex ctz = TileBitmap_ctz(k);
        *val++ = dns_val[((ctz & 0xfc) << 2) | (ctz & 0x3)];
        bitmap ^= k;
    }
}

/*
 * CSR to BSR
 * BSR_BlockSz_bit: 2^BSR_BlockSz_bit = blockSz
 */
void CSR_to_BSR(
    const int m, const int n, const int nnz, const MatIndex *csr_row_ptr, const MatIndex *csr_col_ptr, const MatValue *val,
    int *bsr_m, int *bsr_n, int *bsr_nnz, MatIndex **bsr_row_ptr, MatIndex **bsr_col_ptr, MatValue **bsr_val)
{
    const int blockSz = 1 << BSR_BlockSz_bit;
    *bsr_m = (m + blockSz - 1) >> BSR_BlockSz_bit;
    *bsr_n = (n + blockSz - 1) >> BSR_BlockSz_bit;

    *bsr_row_ptr = (MatIndex *)calloc((*bsr_m + 1), sizeof(MatIndex));

#pragma omp parallel for
    for (int i = 0; i < m; i += blockSz)
    {
        unsigned int _size = *bsr_m;
        BHash *nnz_set_local = create_table(&_size);

        for (int j = 0; j < blockSz; ++j)
        {
            if (i + j >= m)
                break;
            MatIndex row_start = csr_row_ptr[i + j];
            MatIndex row_end = csr_row_ptr[i + j + 1];

            for (int k = row_start; k < row_end; ++k)
            {
                MatIndex col = csr_col_ptr[k] >> BSR_BlockSz_bit;
                insert_table(nnz_set_local, col);
            }
        }

        (*bsr_row_ptr)[(i >> BSR_BlockSz_bit) + 1] = count_table(nnz_set_local, _size);
        free_table(nnz_set_local);
    }

    omp_inclusive_scan(*bsr_row_ptr + 1, *bsr_m);

    *bsr_nnz = (*bsr_row_ptr)[*bsr_m];
    *bsr_col_ptr = (MatIndex *)malloc(sizeof(MatIndex) * *bsr_nnz);
    *bsr_val = (MatValue *)malloc(((*bsr_nnz) << BSR_BlockSz_bit << BSR_BlockSz_bit) * sizeof(MatValue));

    // pack to BSR
#pragma omp parallel for
    for (int bi = 0; bi < *bsr_m; ++bi)
    {
        MatValue **_blocks = (MatValue **)calloc((*bsr_n), sizeof(MatValue *));
        const MatIndex bsr_row_start = (*bsr_row_ptr)[bi];
        MatIndex n_blks = 0;

        for (char r = 0; r < blockSz; ++r)
        {
            MatIndex i = (bi << BSR_BlockSz_bit) | r;
            if (i >= m)
                continue;
            for (MatIndex jj = csr_row_ptr[i]; jj < csr_row_ptr[i + 1]; ++jj)
            {
                MatIndex j = csr_col_ptr[jj];
                MatIndex bj = j >> BSR_BlockSz_bit;

                if (_blocks[bj] == 0)
                {
                    _blocks[bj] = *bsr_val + ((bsr_row_start + n_blks) << BSR_BlockSz_bit << BSR_BlockSz_bit);
                    (*bsr_col_ptr)[bsr_row_start + n_blks] = bj;
                    ++n_blks;
                }

                _blocks[bj][(r << BSR_BlockSz_bit) | (j & TileBlockSz_negetive_one)] = val[jj];
            }
        }

        free(_blocks);
    }
}

TileBitmap _16x16_dns_to_TileLevel16_bitmap(const MatValue *bsr_val, MatIndex *off_ptr)
{
    TileBitmap cnt = 0;
    TileBitmap _bitmap = 0;
#pragma unroll
    for (TileIndex bi = 0; bi < BSR_BlockSz; bi += TileBlockSz)
    {
#pragma unroll
        for (TileIndex i = 0; i < TileBlockSz; ++i)
        {
#pragma unroll
            for (TileIndex j = 0; j < BSR_BlockSz; ++j)
            {
                if (bsr_val[(bi + i) * BSR_BlockSz + j] != 0)
                {
                    _bitmap |= (1 << ((j >> TileBlockSz_bit) + bi));
                    ++cnt;
                }
            }
        }
    }
    *off_ptr = cnt;
    return _bitmap;
}

void _16x16_dns_to_TileLevel4_bitmap(
    const MatValue *bsr_val, TileBitmap *bitmap_ptr, TileIndex *val_map_ptr, MatValue *val_ptr_4, TileBitmap bitmap_16)
{
    const int blockSz = 1 << BSR_BlockSz_bit;
    const int step_bit = BSR_BlockSz_bit - TileBlockSz_bit;

    TileIndex nnz = TileBitmap_popcount(bitmap_16);

    val_map_ptr[0] = 0;
    for (TileIndex x = 0; x < nnz; ++x)
    {
        TileBitmap k = TileBitmap_lowbit(bitmap_16);
        TileIndex ctz = TileBitmap_ctz(k), bi = ctz & 0xc, bj = (ctz & 3) << step_bit; // (bi + ii) << BSR_BlockSz_bit | bj | jj

        TileBitmap _bitmap = 0;
#pragma unroll
        for (int ii = 0; ii < TileBlockSz; ++ii)
        {
#pragma unroll
            for (int jj = 0; jj < TileBlockSz; ++jj)
            {
                if (bsr_val[(bi + ii) << BSR_BlockSz_bit | bj | jj] != 0)
                {
                    _bitmap |= (1 << (jj | (ii << 2)));
                }
            }
        }
        bitmap_ptr[x] = _bitmap;
        if (x + 1 < nnz) val_map_ptr[x + 1] = val_map_ptr[x] + TileBitmap_popcount(_bitmap);

        _bitmap_to_bitmap(bitmap_ptr[x], val_ptr_4 + val_map_ptr[x], bsr_val + (bi << BSR_BlockSz_bit | bj));
        bitmap_16 ^= k;
    }
}

__attribute__((malloc))
TileMatrix *
BSR_to_TileMatrix(const int bsr_m, const int bsr_n, const int nnz, MatIndex *bsr_row_ptr, MatIndex *bsr_col_ptr, const MatValue *bsr_val)
{
    const int blockSz = 1 << BSR_BlockSz_bit;

    TileMatrix *res = (TileMatrix *)malloc(sizeof(TileMatrix));
    res->_m = bsr_m;
    res->_n = bsr_n;
    res->_nnz = nnz;

    res->tile_row_ptr = bsr_row_ptr;
    res->tile_col_ptr = bsr_col_ptr;

    res->tile_16_bitmap_ptr = (TileBitmap *)malloc(sizeof(TileBitmap) * nnz);
    res->tile_val_map = (MatIndex *)malloc((nnz + 1) * sizeof(MatIndex));
    res->tile_off_map = (MatIndex *)malloc((nnz + 1) * sizeof(MatIndex));
    res->tile_val_map[0] = res->tile_off_map[0] = 0;

    int cnt_nnz_16 = 0, cnt_nnz_4 = 0, cnt_row_4 = 0, cnt_col_4 = 0;

#pragma omp parallel for
    for (int i = 0; i < nnz; ++i)
    {
        const MatValue *cur_block = (MatValue *)(bsr_val + i * blockSz * blockSz);
        res->tile_16_bitmap_ptr[i] = _16x16_dns_to_TileLevel16_bitmap(cur_block, res->tile_off_map + i + 1);
        res->tile_val_map[i + 1] = TileBitmap_popcount(res->tile_16_bitmap_ptr[i]);
    }

    omp_inclusive_scan(res->tile_val_map + 1, nnz);
    omp_inclusive_scan(res->tile_off_map + 1, nnz);

    cnt_nnz_16 = res->tile_val_map[nnz];

    res->tile_4_bitmap_ptr = (TileBitmap *)malloc((cnt_nnz_16 + 1) * sizeof(TileBitmap));
    res->tile_4_offmap_ptr = (TileIndex *)malloc((cnt_nnz_16 + 1) * sizeof(TileIndex));
    res->tile_val_ptr = (MatValue *)malloc(sizeof(MatValue) * res->tile_off_map[nnz]);

#pragma omp parallel for
    for (int i = 0; i < nnz; ++i)
    {
        const MatIndex _offset = res->tile_val_map[i];
        const MatValue *cur_block = bsr_val + (i * blockSz * blockSz);

        TileIndex *cur_val_map = res->tile_4_offmap_ptr + _offset;
        TileBitmap *cur_bitmap_4 = res->tile_4_bitmap_ptr + _offset;
        MatValue   *cur_val_ptr  = res->tile_val_ptr + res->tile_off_map[i];

        _16x16_dns_to_TileLevel4_bitmap(cur_block, cur_bitmap_4, cur_val_map, cur_val_ptr, res->tile_16_bitmap_ptr[i]);
    }

    return res;
}

__attribute__((malloc))
TileMatrix *
CSR_to_TileMatrix(const int m, const int n, const int nnz, const MatIndex *csr_row_ptr, const MatIndex *csr_col_ptr, const MatValue *val)
{
    int _m, _n, _nnz;
    MatIndex *bsr_row_ptr, *bsr_col_ptr;
    MatValue *bsr_val_ptr; // tmp data: 16 * 16 per block
    CSR_to_BSR(m, n, nnz, csr_row_ptr, csr_col_ptr, val, &_m, &_n, &_nnz, &bsr_row_ptr, &bsr_col_ptr, &bsr_val_ptr);
    TileMatrix *seq_tile_matrix = BSR_to_TileMatrix(_m, _n, _nnz, bsr_row_ptr, bsr_col_ptr, bsr_val_ptr);

    seq_tile_matrix->meta_m = m;
    seq_tile_matrix->meta_n = n;
    seq_tile_matrix->meta_nnz = nnz;

    free(bsr_val_ptr);

    return seq_tile_matrix;
}