#pragma once
#include <msg.h>
#include "./bitmap.h"
#include <tmatrix/DataStructure/TileMatrix.h>

TileBitmap scan_row(TileBitmap bitmap)
{
    TileBitmap line = 0;
    #pragma unroll
    for (TileIndex i = 0; i < 4; ++i)
    {
        line |= (bitmap & 0xf);
        bitmap >>= 4;
    }
    return line;
}

TileBitmap scan_col(TileBitmap bitmap)
{
    TileBitmap line = 0;
    #pragma unroll
    for (TileIndex i = 0; i < 4; ++i)
    {
        line |= (bitmap & 0x1111);
        bitmap >>= 1;
    }
    return line & 1 | (line & 0x0010) >> 3 | (line & 0x0100) >> 6 | (line & 0x1000) >> 9;
}

void TileLevel_4_coo_x_v(TileBitmap bitmap, const TileIndex nnz, const TileIndex* row_ptr, const TileIndex* col_ptr, const MatValue* val, const MatValue* v, MatValue* result)
{
    TileBitmap line_input = scan_row(bitmap);
    TileBitmap line_output = scan_col(bitmap), _raw = line_output;
    
    MatValue cache_input_v[4], cache_output_v[4];

#pragma unroll
    for (TileIndex i = 0; i < 4; ++i) {
        cache_input_v[i] = line_input & 1? v[i]: 0;
        line_input >>= 1;
        cache_output_v[i] = line_output & 1? result[i]: 0;
        line_output >>= 1;
    }

    for (TileIndex i = 0; i < nnz; ++i)
    {
        cache_output_v[row_ptr[i]] += val[i] * cache_input_v[col_ptr[i]];
    }

#pragma unroll
    for (TileIndex i = 0; i < 4; ++i) {
        if (_raw & 1) result[i] = cache_output_v[i];
        _raw >>= 1;
    }
}

void TileLevel_4_csr_x_v(TileBitmap bitmap, const TileIndex nnz, const TileIndex* row_ptr, const TileIndex* col_ptr, const MatValue* val, const MatValue* v, MatValue* result)
{
    TileBitmap line_input = scan_row(bitmap);
    TileBitmap line_output = scan_col(bitmap), _raw = line_output;

    MatValue cache_input_v[4], cache_output_v[4];

    #pragma unroll
    for (TileIndex i = 0; i < 4; ++i) {
        cache_input_v[i] = line_input & 1? v[i]: 0;
        line_input >>= 1;
        cache_output_v[i] = line_output & 1? result[i]: 0;
        line_output >>= 1;
    }

    #pragma unroll
    for (TileIndex i = 0; i < 4; ++i)
    {
        const TileIndex row_end = row_ptr[i + 1];
        for (TileIndex j = row_ptr[i]; j < row_end; ++j)
        {
            cache_output_v[i] += val[j] * cache_input_v[col_ptr[j]];
        }
    }

    #pragma unroll
    for (TileIndex i = 0; i < 4; ++i) {
        if (_raw & 1) result[i] = cache_output_v[i];
        _raw >>= 1;
    }
}

void TileLevel_4_ell_x_v(TileBitmap bitmap, const TileIndex nnz, const TileIndex* row_ptr, const TileIndex* col_ptr, const MatValue* val, const MatValue* v, MatValue* result)
{
    TileBitmap line_input = scan_row(bitmap);
    TileBitmap line_output = scan_col(bitmap), _raw = line_output;

    MatValue cache_input_v[4], cache_output_v[4];

    #pragma unroll
    for (TileIndex i = 0; i < 4; ++i) {
        cache_input_v[i] = line_input & 1? v[i]: 0;
        line_input >>= 1;
        cache_output_v[i] = line_output & 1? result[i]: 0;
        line_output >>= 1;
    }

    TileIndex max_row = *row_ptr;
    #pragma unroll
    for (TileIndex i = 0; i < 4; ++i)
    {
        for (TileIndex j = 0; j < max_row; ++j)
        {
            if (col_ptr[i * max_row + j] != negetive_one)
                cache_output_v[i] += val[i * max_row + j] * cache_input_v[col_ptr[i * max_row + j]];
        }
    }

    #pragma unroll
    for (TileIndex i = 0; i < 4; ++i) {
        if (_raw & 1) result[i] = cache_output_v[i];
        _raw >>= 1;
    }
}

void TileLevel_4_hyb_x_v(TileBitmap bitmap, const TileIndex nnz, const TileIndex* row_ptr, const TileIndex* col_ptr, const MatValue* val, const MatValue* v, MatValue* result)
{
    TileBitmap line_input = scan_row(bitmap);
    TileBitmap line_output = scan_col(bitmap), _raw = line_output;

    MatValue cache_input_v[4], cache_output_v[4];

    // echo(debug, "0x%x 0x%x 0x%x", bitmap, line_input, line_output);

    #pragma unroll
    for (TileIndex i = 0; i < 4; ++i) {
        cache_input_v[i] = line_input & 1? v[i]: 0;
        line_input >>= 1;
        cache_output_v[i] = line_output & 1? result[i]: 0;
        line_output >>= 1;
    }

    TileIndex min_row = *row_ptr, coo_start = (min_row << 2);
    char rest = nnz - coo_start;

    // ELL PART
    #pragma unroll
    for (TileIndex i = 0; i < 4; ++i)
    {
        for (TileIndex j = 0; j < min_row; ++j)
        {
            if (col_ptr[i * min_row + j] != negetive_one) {
                cache_output_v[i] += val[i * min_row + j] * cache_input_v[col_ptr[i * min_row + j]];
                // echo(debug, "cache_output_v[%d] += val[%d] * cache_input_v[%d] = %.0f", i, i * min_row + j, col_ptr[i * min_row + j], cache_output_v[i]);
            }
        }
    }
    // COO PART
    for (TileIndex i = 0; i < rest; ++i)
    {
        cache_output_v[row_ptr[i + 1]] += val[coo_start + i] * cache_input_v[col_ptr[coo_start + i]];
    }

    #pragma unroll
    for (TileIndex i = 0; i < 4; ++i) {
        if (_raw & 1) result[i] = cache_output_v[i];
        _raw >>= 1;
    }
}

void TileLevel_4_dnsrow_x_v(TileBitmap bitmap, const TileIndex nnz, const TileIndex* row_ptr, const TileIndex* col_ptr, const MatValue* val, const MatValue* v, MatValue* result)
{
    TileBitmap line_output = scan_col(bitmap), _raw = line_output;

    MatValue cache_input_v[4], cache_output_v[4];

    #pragma unroll
    for (TileIndex i = 0; i < 4; ++i) {
        cache_input_v[i] = v[i];
        cache_output_v[i] = line_output & 1? result[i]: 0;
        line_output >>= 1;
    }

    TileIndex num_row = *row_ptr;

    for (TileIndex i = 0; i < num_row; ++i)
    {
        #pragma unroll
        for (TileIndex j = 0; j < 4; ++j) {
            cache_output_v[col_ptr[i]] += val[i << 2 | j] * cache_input_v[j];
        }
    }

    #pragma unroll
    for (TileIndex i = 0; i < 4; ++i) {
        if (_raw & 1) result[i] = cache_output_v[i];
        _raw >>= 1;
    }
}

void TileLevel_4_dnscol_x_v(TileBitmap bitmap, const TileIndex nnz, const TileIndex* row_ptr, const TileIndex* col_ptr, const MatValue* val, const MatValue* v, MatValue* result)
{
    TileBitmap line_input = scan_row(bitmap);

    MatValue cache_input_v[4], cache_output_v[4];

    #pragma unroll
    for (TileIndex i = 0; i < 4; ++i) {
        cache_input_v[i] = line_input & 1? v[i]: 0;
        line_input >>= 1;
        cache_output_v[i] = result[i];
    }

    TileIndex num_col = *row_ptr;
    
    for (TileIndex i = 0; i < num_col; ++i)
    {
        #pragma unroll
        for (TileIndex j = 0; j < 4; ++j) {
            cache_output_v[j] += val[i << 2 | j] * cache_input_v[col_ptr[i]];
        }
    }

    #pragma unroll
    for (TileIndex i = 0; i < 4; ++i) {
        result[i] = cache_output_v[i];
    }
}

void TileLevel_4_dns_x_v(TileBitmap bitmap, const TileIndex nnz, const TileIndex* row_ptr, const TileIndex* col_ptr, const MatValue* val, const MatValue* v, MatValue* result)
{
    MatValue cache_input_v[4], cache_output_v[4];

    #pragma unroll
    for (TileIndex i = 0; i < 4; ++i) {
        cache_input_v[i] = v[i];
        cache_output_v[i] = result[i];
    }

    #pragma unroll
    for (TileIndex i = 0; i < 4; ++i)
    {
        #pragma unroll
        for (TileIndex j = 0; j < 4; ++j) {
            cache_output_v[i] += val[i << 2 | j] * cache_input_v[j];
        }
    }

    #pragma unroll
    for (TileIndex i = 0; i < 4; ++i) {
        result[i] = cache_output_v[i];
    }
}

void TileLevel_4_bitmap_x_v(TileBitmap bitmap, const TileIndex nnz, const TileIndex* row_ptr, const TileIndex* col_ptr, const MatValue* val, const MatValue* v, MatValue* result)
{
    MatValue cache_input_v[4], cache_output_v[4];

    TileBitmap line_input = scan_row(bitmap);
    TileBitmap line_output = scan_col(bitmap), _raw = line_output;

    #pragma unroll
    for (TileIndex i = 0; i < 4; ++i) {
        cache_input_v[i] = line_input & 1? v[i]: 0;
        line_input >>= 1;
        cache_output_v[i] = line_output & 1? result[i]: 0;
        line_output >>= 1;
    }

    TileIndex _i = 0;

    while (bitmap)
    {
        TileBitmap _k = TileBitmap_lowbit(bitmap);
        TileIndex cnt = TileBitmap_ctz(_k);

        cache_output_v[cnt >> 2] += (*val++) * cache_input_v[cnt & 0x3];
        
        bitmap ^= _k;
    }

    // for (TileLevel_IndexType i = 0; i < nnz; ++i)
    // {
    //     TileLevel_Bitmap _k = TileBitmap_low_bit(bitmap);
    //     TileLevel_IndexType cnt = TileBitmap_ctz(_k);
    //     cache_output_v[cnt >> 2] += val[i] * cache_input_v[cnt & 0x3];
    //     bitmap ^= _k;
    // }

    
    #pragma unroll
    for (TileIndex i = 0; i < 4; ++i) {
        if (_raw & 1) result[i] = cache_output_v[i];
        _raw >>= 1;
    }
}

void TileLevel_4_nothing_x_v(TileBitmap bitmap, const TileIndex nnz, const TileIndex* row_ptr, const TileIndex* col_ptr, const MatValue* val, const MatValue* v, MatValue* result)
{

}

void (*TileLevel_4_format_x_v[8])(TileBitmap, const TileIndex, const TileIndex*, const TileIndex*, const MatValue*, const MatValue*, MatValue*) = {
    TileLevel_4_coo_x_v,
    TileLevel_4_csr_x_v,
    TileLevel_4_ell_x_v,
    TileLevel_4_hyb_x_v,
    TileLevel_4_dnsrow_x_v,
    TileLevel_4_dnscol_x_v,
    TileLevel_4_dns_x_v,
    TileLevel_4_bitmap_x_v
};

void SpMV(TileMatrix*tm, MatValue*v, MatValue*result)
{
    MatIndex _m = tm->_m, _n = tm->_n, _nnz = tm->_nnz;

    MatIndex *tile_row_ptr = tm->tile_row_ptr, *tile_col_ptr = tm->tile_col_ptr;

#pragma omp parallel for
    for (MatIndex i = 0; i < _m; ++i)
    {
        MatIndex tile_row_end = tile_row_ptr[i + 1];  // 3
        
        for (MatIndex j = tile_row_ptr[i]; j < tile_row_end; ++j)
        {
            const MatIndex _offset = tm->tile_val_map[j];
            MatIndex cur_j = tile_col_ptr[j];                                 // 0, 1, 3

            TileBitmap tile_16_bitmap = tm->tile_16_bitmap_ptr[j];    // 0xbcab, 0xbcab, 0xbcab

            TileIndex *tile_16_row_ptr = tm->tile_row_ptr_16 + j * 5;   // val0: row 1
            TileIndex *tile_16_col_ptr = tm->tile_col_ptr_16 + _offset; // val0: col 1, map: 0 10 20

            MatIndex* tile_16_row_map = tm->tile_row_map_16 + _offset;        // val0: val 1, map: 0 10 20
            MatIndex* tile_16_col_map = tm->tile_col_map_16 + _offset;        // val0: val 1, map: 0 10 20
            MatIndex* tile_16_val_map = tm->tile_val_map_16 + _offset;        // val0: val 1, map: 0 10 20

            TileIndex _cnt = 0;
            
            while (tile_16_bitmap)
            {
                TileBitmap _k = TileBitmap_lowbit(tile_16_bitmap);
                // echo(debug, "v start: %d, r start: %d, format: %s", cur_j << 4 | tile_16_col_ptr[_cnt] << 2, i << 4 | k & 0xc, TileSparseFormatName[tm->tile_format_ptr[_offset + _cnt]]);
                TileLevel_4_format_x_v[tm->tile_format_ptr[_offset + _cnt]](
                    tm->tile_4_bitmap_ptr[_offset + _cnt],       // TileLevel_Bitmap bitmap
                    tile_16_val_map[_cnt + 1] - tile_16_val_map[_cnt], // TileLevel_IndexType nnz
                    tm->tile_row_ptr_4 + tile_16_row_map[_cnt],
                    tm->tile_col_ptr_4 + tile_16_col_map[_cnt],
                    tm->tile_val_ptr + tile_16_val_map[_cnt],
                    v + (cur_j << 4 | tile_16_col_ptr[_cnt] << 2), result + (i << 4 | TileBitmap_ctz(_k) & 0xc)
                );
                ++_cnt;
                tile_16_bitmap ^= _k;
            }
        }
    }
}