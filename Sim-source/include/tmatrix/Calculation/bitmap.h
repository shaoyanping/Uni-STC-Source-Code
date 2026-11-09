#pragma once

#include <tmatrix/Utils/omp_utils.h>

#define TileBitmap_col_x_realrow(col, row) (((col) * 0xf) & row)
#define TileBitmap_col_x_row(col, row) (((col) * 0xf) & ((row) * 0x1111))

TileBitmap TileBitmap_x(TileBitmap bitMap_1, TileBitmap bitMap_2)
{
    // bitMap_1: input bitmap 1
    // bitMap_2: input bitmap 2
    // return: bitmap 3

    // get every col of bitMap_1 and every row of bitMap_2
    TileBitmap res = 0;

#pragma unroll
    for (TileIndex _ = 0; _ < 4; ++_)
    {
        res |= TileBitmap_col_x_row(bitMap_1 & 0x1111, bitMap_2 & 0xf);

        bitMap_1 >>= 1;
        bitMap_2 >>= 4;
    }

    return res;
}

TileIndex TileBitmap_x_mpd(TileBitmap bitMap_1, TileBitmap bitMap_2)
{
    // bitMap_1: input bitmap 1
    // bitMap_2: input bitmap 2
    // return: bitmap 3

    // get every col of bitMap_1 and every row of bitMap_2
    TileIndex res = 0;

#pragma unroll
    for (TileIndex _ = 0; _ < 4; ++_)
    {
        res += TileBitmap_popcount(TileBitmap_col_x_row(bitMap_1 & 0x1111, bitMap_2 & 0xf));

        bitMap_1 >>= 1;
        bitMap_2 >>= 4;
    }

    return res;
}

void TileBitmap_to_row_ptr(TileBitmap bitmap, TileIndex* row_ptr)
{
    TileBitmap mask = 0xf;
    row_ptr[0] = 0;
#pragma unroll
    for (TileIndex i = 0; i < 4; ++i)
    {
        row_ptr[i + 1] = row_ptr[i] + TileBitmap_popcount(bitmap & mask);
        mask <<= 4;
    }
}

void TileBitmap_to_col_ptr(TileBitmap bitmap, TileIndex* col_ptr)
{
    while (bitmap)
    {
        TileBitmap k = TileBitmap_lowbit(bitmap);
        *col_ptr++ = TileBitmap_ctz(k) & 3;
        bitmap ^= k;
    }
}

void bitmap2csr(TileBitmap bitmap_16, TileBitmap *bitmap_4, u_int16_t*rowptr, u_int8_t*colptr, u_int16_t*nz)
{
    TileBitmap b16 = bitmap_16;
    u_int8_t nz16 = TileBitmap_popcount(bitmap_16);
    for (u_int8_t i = 0; i < nz16; ++i)
    {
        TileBitmap k = TileBitmap_lowbit(b16);
        b16 ^= k;
        u_int8_t Actz = TileBitmap_ctz(k);
        u_int8_t row16 = Actz & 0xc;
        TileBitmap cur_4 = bitmap_4[i];
        *nz += TileBitmap_popcount(cur_4);
        
        while (cur_4)
        {
            TileBitmap _k = TileBitmap_lowbit(cur_4);
            cur_4 ^= _k;
            u_int8_t ctz = TileBitmap_ctz(_k), row4 = row16 | (ctz >> 2);
            rowptr[row4 + 1] += 1;
        }
    }
    inclusive_scan(rowptr + 1, 16);

    u_int16_t _rows[16];
    memcpy(_rows, rowptr, 16 * sizeof(u_int16_t));

    for (u_int8_t i = 0; i < nz16; ++i)
    {
        TileBitmap k = TileBitmap_lowbit(bitmap_16);
        bitmap_16 ^= k;
        u_int8_t Actz = TileBitmap_ctz(k);
        u_int8_t row16 = Actz & 0xc, col16 = (Actz & 3) << 2;
        TileBitmap cur_4 = bitmap_4[i];
        
        while (cur_4)
        {
            TileBitmap _k = TileBitmap_lowbit(cur_4);
            cur_4 ^= _k;
            u_int8_t ctz = TileBitmap_ctz(_k), row4 = row16 | (ctz >> 2);
            colptr[_rows[row4]++] = col16 | (ctz & 3);
        }
    }
}

inline void TileLevel_4_bitmap_x_v(TileBitmap bitmap, const MatValue *val, const MatValue *v, MatValue *result)
{
    TileIndex nnz = TileBitmap_popcount(bitmap);

    for (TileIndex i = 0; i < nnz; ++i)
    {
        TileBitmap _k = TileBitmap_lowbit(bitmap);
        bitmap ^= _k;

        TileIndex cnt = TileBitmap_ctz(_k);
        result[cnt >> 2] += val[i] * v[cnt & 0x3];
    }
}
