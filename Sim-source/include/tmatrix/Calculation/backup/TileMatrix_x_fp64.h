#pragma once
#include "./TileMatrix_x_sym.h"
#include <tuple>

std::tuple<u_int16_t, u_int8_t> Tile_16_Mx_nv_64(
    TileBitmap A_16_bitmap, TileBitmap *A_bitmap_ptr,
    TileBitmap B_16_bitmap, TileBitmap *B_bitmap_ptr,
    TileBitmap C_16_Bitmap, TileBitmap *C_bitmap_ptr
)
{
    return std::make_tuple(0, 64);
}

std::tuple<u_int16_t, u_int8_t> Tile_16_Mx_ds_64(
    TileBitmap A_16_bitmap, TileBitmap *A_bitmap_ptr,
    TileBitmap B_16_bitmap, TileBitmap *B_bitmap_ptr,
    TileBitmap C_16_Bitmap, TileBitmap *C_bitmap_ptr
)
{
    // u_int16_t midx = 0;
    u_int8_t cycle = 0;
    TileBitmap amask[4] = {0x1111, 0x2222, 0x4444, 0x8888};
    TileBitmap bmask[4] = {0xf, 0xf0, 0xf00, 0xf000};
    for (TileIndex i = 0; i < 4; ++i)
    {
        TileBitmap A_col = A_16_bitmap & amask[i], B_row = B_16_bitmap & bmask[i];
        if (A_col && B_row)
        {
            for (TileIndex j = 0; j < 4; ++j)
            {
                TileBitmap _a = A_col, _b = B_row;
                u_int16_t A_col_len = 0;
                u_int16_t B_row_len = 0;

                while (_a)
                {
                    TileBitmap k = TileBitmap_lowbit(_a);
                    _a ^= k;

                    TileIndex ctz = TileBitmap_ctz(k);
                    TileBitmap a4b = A_bitmap_ptr[TileBitmap_popcount(A_16_bitmap & ((1 << ctz) - 1))];
                    A_col_len += TileBitmap_popcount(a4b & amask[j]);
                }

                while (_b)
                {
                    TileBitmap k = TileBitmap_lowbit(_b);
                    _b ^= k;

                    TileIndex ctz = TileBitmap_ctz(k);
                    TileBitmap b4b = B_bitmap_ptr[TileBitmap_popcount(B_16_bitmap & ((1 << ctz) - 1))];
                    B_row_len += TileBitmap_popcount(b4b & bmask[j]);
                }

                if (A_col_len && B_row_len){
                    A_col_len = (A_col_len + 7) >> 3;
                    B_row_len = (B_row_len + 7) >> 3;
                    cycle += std::max(A_col_len * B_row_len, 1);
                }
            }
        }
    }
    // return std::make_tuple(midx, cycle);
    return std::make_tuple(0, cycle);
}

std::tuple<u_int16_t, u_int8_t> Tile_16_Mx_rm_64(
    TileBitmap A_16_bitmap, TileBitmap *A_bitmap_ptr,
    TileBitmap B_16_bitmap, TileBitmap *B_bitmap_ptr,
    TileBitmap C_16_Bitmap, TileBitmap *C_bitmap_ptr
)
{
    // u_int16_t midx = 0;
    u_int8_t cycle = 0, nz16A = TileBitmap_popcount(A_16_bitmap);
    u_int16_t Arows[17] = {0}, Brows[17] = {0}, nzA = 0, nzB = 0;
    u_int8_t Acols[256] = {0}, Bcols[256] = {0};

    bitmap2csr(A_16_bitmap, A_bitmap_ptr, Arows, Acols, &nzA);
    bitmap2csr(B_16_bitmap, B_bitmap_ptr, Brows, Bcols, &nzB);

    for (u_int8_t i = 0; i < 4; ++i)
    {
        u_int8_t rowlen[4], ib = i << 2;

        for (u_int8_t j = 0; j < 4; ++j) rowlen[j] = Arows[ib + j + 1] - Arows[ib + j];

        u_int8_t max_row_len = std::max(std::max(rowlen[0], rowlen[1]), std::max(rowlen[2], rowlen[3]));

        for (u_int8_t window_base = 0; window_base < max_row_len; window_base += 2)
        {
            int max_col_len = 0;
            for (u_int8_t j = 0; j < 4; ++j)
            {
                if (window_base < rowlen[j]) {
                    u_int8_t col = Acols[Arows[i << 2 | j] + window_base];
                    // echo(debug, "col: %d, row_len: %d", col, Brows[col + 1] - Brows[col]);
                    max_col_len = std::max(max_col_len, Brows[col + 1] - Brows[col]);
                }
            }

            for (u_int8_t j = 0; j < 4; ++j)
            {
                if (window_base + 1 < rowlen[j]) {
                    u_int8_t col = Acols[Arows[i << 2 | j] + window_base + 1];
                    // echo(debug, "col: %d, row_len: %d", col, Brows[col + 1] - Brows[col]);
                    max_col_len = std::max(max_col_len, Brows[col + 1] - Brows[col]);
                }
            }

            cycle += (max_col_len + 7) >> 3;
            // echo(rule, "cycle: %d", cycle);
        }
    }    
    // return std::make_tuple(midx, cycle);
    return std::make_tuple(0, cycle);
}

void Tile_16_M_x_M_64(
    TileBitmap A_16_bitmap, TileBitmap *A_bitmap_ptr, 
    TileBitmap B_16_bitmap, TileBitmap *B_bitmap_ptr, 
    TileBitmap C_16_Bitmap, TileBitmap *C_bitmap_ptr, 
    std::tuple<u_int16_t, u_int8_t>&_nv, std::tuple<u_int16_t, u_int8_t>&_ds, std::tuple<u_int16_t, u_int8_t>&_rm, std::tuple<u_int16_t, u_int8_t>&_smu
)
{
    _nv = Tile_16_Mx_nv_64(A_16_bitmap, A_bitmap_ptr, B_16_bitmap, B_bitmap_ptr, C_16_Bitmap, C_bitmap_ptr);
    _ds = Tile_16_Mx_ds_64(A_16_bitmap, A_bitmap_ptr, B_16_bitmap, B_bitmap_ptr, C_16_Bitmap, C_bitmap_ptr);
    _rm = Tile_16_Mx_rm_64(A_16_bitmap, A_bitmap_ptr, B_16_bitmap, B_bitmap_ptr, C_16_Bitmap, C_bitmap_ptr);
    _smu = Tile_16_Mx_smu<64>(A_16_bitmap, A_bitmap_ptr, B_16_bitmap, B_bitmap_ptr, C_16_Bitmap, C_bitmap_ptr);
}

task_results TileMatrix_GEMM_Num_64(TileMatrix *A, TileMatrix *B, TileMatrix *C)
{
    task_results res;
    u_int64_t cycle_nv = 0, cycle_ds = 0, cycle_rm = 0, midx_smu = 0, cycle_smu = 0, mp16 = 0, mp4 = 0;

#pragma omp parallel for reduction(+ : cycle_nv, cycle_ds, cycle_rm, midx_smu, cycle_smu, mp16, mp4)
    for (MatIndex Ai = 0; Ai < A->_m; ++Ai)
    {
        unsigned short nv_id = 0, ds_id = 0, rm_id = 0;

        const MatIndex cur_C_row_len = C->tile_row_ptr[Ai + 1] - C->tile_row_ptr[Ai];
        MatIndex *cur_C_col_ptr = C->tile_col_ptr + C->tile_row_ptr[Ai];
        u_int64_t _nvc = 0, _dsc = 0, _rmc = 0, _smuc = 0, _smum = 0, _mp16 = 0, _mp4 = 0;

        for (MatIndex Aj = A->tile_row_ptr[Ai]; Aj < A->tile_row_ptr[Ai + 1]; ++Aj)
        {
            MatIndex cur_A_16_col = A->tile_col_ptr[Aj];
            TileBitmap cur_A_16_bitmap = A->tile_16_bitmap_ptr[Aj];
            TileBitmap *cur_A_4_bitmap_ptr = A->tile_4_bitmap_ptr + A->tile_val_map[Aj];

            for (MatIndex Bj = B->tile_row_ptr[cur_A_16_col]; Bj < B->tile_row_ptr[cur_A_16_col + 1]; ++Bj)
            {
                TileBitmap cur_B_16_bitmap = B->tile_16_bitmap_ptr[Bj], x = TileBitmap_x(cur_A_16_bitmap, cur_B_16_bitmap);
                if (!x)
                    continue;
                MatIndex Cj, cur_C_Ai_Bj_valmap = binary_find_index(cur_C_col_ptr, C->tile_val_map + C->tile_row_ptr[Ai], cur_C_row_len, B->tile_col_ptr[Bj], &Cj);
                if (cur_C_Ai_Bj_valmap == -1)
                    continue;

                Cj += C->tile_row_ptr[Ai];

                TileBitmap cur_C_16_bitmap = C->tile_16_bitmap_ptr[Cj];
                TileBitmap *cur_C_4_bitmap_ptr = C->tile_4_bitmap_ptr + cur_C_Ai_Bj_valmap;
                TileBitmap *cur_B_4_bitmap_ptr = B->tile_4_bitmap_ptr + B->tile_val_map[Bj];
                ++_mp16;
                _mp4 += TileBitmap_x_mpd(cur_A_16_bitmap, cur_B_16_bitmap);

                std::tuple<u_int16_t, u_int8_t> _nv, _ds, _rm, _smu; // 中间积数量 / Cycle数

                Tile_16_M_x_M_64(
                    cur_A_16_bitmap, cur_A_4_bitmap_ptr,
                    cur_B_16_bitmap, cur_B_4_bitmap_ptr,
                    cur_C_16_bitmap, cur_C_4_bitmap_ptr,
                    _nv, _ds, _rm, _smu);
                
                _nvc += std::get<1>(_nv);
                _dsc += std::get<1>(_ds);
                _rmc += std::get<1>(_rm);
                _smum += std::get<0>(_smu);
                _smuc += std::get<0>(_smu) && !std::get<1>(_smu)? 256: std::get<1>(_smu);
            }
        }
        cycle_nv += _nvc;
        cycle_ds += _dsc;
        cycle_rm += _rmc;
        midx_smu += _smum;
        cycle_smu += _smuc;
        mp16 += _mp16;
        mp4 += _mp4;
    }

    echo(markdown, "## 中间积: %lu", midx_smu);
    echo(markdown, "|Tensor Core|Cycle|Mac Rate (%d @ FP64)|\\n|---|---|:---:|\\n|NV-STC|%lu|%.2lf%%|\\n|DS-STC|%lu|%.2lf%%|\\n|RM-STC|%lu|%.2lf%%|\\n|**SMU**|***%lu***|%.2lf%%|",
         64, cycle_nv, midx_smu / 64.0 / cycle_nv * 100.0,
         cycle_ds, midx_smu / 64.0 / cycle_ds * 100.0,
         cycle_rm, midx_smu / 64.0 / cycle_rm * 100.0,
         cycle_smu, midx_smu / 64.0 / cycle_smu * 100.0);
    
    res.ds_cycle = cycle_ds;
    res.rm_cycle = cycle_rm;
    res.smu_cycle = cycle_smu;
    res.ds_mac = midx_smu / 64.0 / cycle_ds * 100.0;
    res.rm_mac = midx_smu / 64.0 / cycle_rm * 100.0;
    res.smu_mac = midx_smu / 64.0 / cycle_smu * 100.0;
    res.smu_speedup = std::max(cycle_ds, cycle_rm) * 1.0 / cycle_smu;
    res.mpnz = midx_smu;
    res.mp16 = mp16;
    res.mp4 = mp4;

    return res;
}
