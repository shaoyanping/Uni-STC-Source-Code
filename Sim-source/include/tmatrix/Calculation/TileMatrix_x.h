#pragma once
#include "./smu_sim.h"
#include <tmatrix/Utils/omp_utils.h>
#include <tmatrix/Utils/hash_sim.h>
#include <tmatrix/DataStructure/TileMatrix.h>
#include <tuple>
#include <algorithm>
#include <unordered_map>

#ifndef SCHEDULER
#define SCHEDULER 8
#endif // !SCHEDULER


void Tile_16_Bitmap_x_Bitmap(
    TileBitmap A_16_bitmap, TileBitmap *A_bitmap_ptr, // input A
    TileBitmap B_16_bitmap, TileBitmap *B_bitmap_ptr, // input B
    TileBitmap C_16_Bitmap, TileBitmap *C_bitmap_ptr  // output C
)
{
    TileBitmap idx[4] = {0, 1, 17, 273}, nnzA = TileBitmap_popcount(A_16_bitmap);

    for (TileIndex Ai = 0; Ai < nnzA; ++Ai)
    {
        TileBitmap Ak = TileBitmap_lowbit(A_16_bitmap);
        A_16_bitmap ^= Ak;

        TileIndex Actz = TileBitmap_ctz(Ak);
        TileBitmap cur_A_4_bitmap = A_bitmap_ptr[Ai];
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

void update_sym_hash(TileBitmap a16, TileBitmap *a4, TileBitmap b16, TileBitmap *b4, hash_key_bitmap_t *table, unsigned int size, MatIndex Bcol)
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

            if (_tmp.val4[ctz] == 0)
                _tmp.val16 ^= k;

            x ^= k;
        }

        if (_tmp.val16)
            insert_key_bitmap_hash(table, size, Bcol, _tmp);
    }
}

TileMatrix *TileMatrix_GEMM_Sym_Sim(TileMatrix *A, TileMatrix *B, bool with_detail_C = false)
{
    TileMatrix *C = (TileMatrix *)malloc(sizeof(TileMatrix));
    C->meta_m = A->meta_m;
    C->meta_n = B->meta_n;
    C->_m = A->_m;
    C->_n = B->_n;

    MatIndex max_threads = omp_get_max_threads();
    C->tile_row_ptr = (MatIndex *)calloc((C->_m + 1), sizeof(MatIndex));
    unsigned int _size = C->_n;
    hash_key_bitmap_t **all_table = (hash_key_bitmap_t **)malloc(sizeof(hash_key_bitmap_t *) * max_threads);

#pragma omp parallel for
    for (MatIndex i = 0; i < max_threads; ++i)
    {
        all_table[i] = (hash_key_bitmap_t *)malloc(sizeof(hash_key_bitmap_t));
        all_table[i]->key = (MatIndex *)malloc(sizeof(MatIndex) * _size);
        all_table[i]->val = (hash_key_bitmap_val_t *)malloc(sizeof(hash_key_bitmap_val_t) * _size);
    }

    hash_key_bitmap_t **seg_tmp_val = (hash_key_bitmap_t **)malloc(sizeof(hash_key_bitmap_t *) * A->_m);

    // echo(debug, "Step 1: building hash structure");
#pragma omp parallel for //* Step 1: 计算16 * 16的非零元数
    for (MatIndex Ai = 0; Ai < A->_m; ++Ai)
    {
        MatIndex tid = omp_get_thread_num();
        hash_key_bitmap_t *table = all_table[tid];

        memset(table->key, -1, sizeof(MatIndex) * _size);
        memset(table->val, 0, sizeof(hash_key_bitmap_val_t) * _size);

        for (MatIndex Aj = A->tile_row_ptr[Ai]; Aj < A->tile_row_ptr[Ai + 1]; ++Aj)
        {
            MatIndex cur_A_16_col = A->tile_col_ptr[Aj];
            TileBitmap cur_A_16_bitmap = A->tile_16_bitmap_ptr[Aj];

            for (MatIndex Bj = B->tile_row_ptr[cur_A_16_col]; Bj < B->tile_row_ptr[cur_A_16_col + 1]; ++Bj)
            {
                update_sym_hash(
                    cur_A_16_bitmap, A->tile_4_bitmap_ptr + A->tile_val_map[Aj],
                    B->tile_16_bitmap_ptr[Bj], B->tile_4_bitmap_ptr + B->tile_val_map[Bj],
                    table, _size, B->tile_col_ptr[Bj]);
            }
        }
        C->tile_row_ptr[Ai + 1] = gather_hash(table, _size);
        if (C->tile_row_ptr[Ai + 1])
        {
            // quicksort_key_val(table->key, table->val, 0, C->tile_row_ptr[Ai + 1] - 1);
            seg_tmp_val[Ai] = (hash_key_bitmap_t *)malloc(sizeof(hash_key_bitmap_t));
            seg_tmp_val[Ai]->key = (MatIndex *)malloc(sizeof(MatIndex) * C->tile_row_ptr[Ai + 1]);
            seg_tmp_val[Ai]->val = (hash_key_bitmap_val_t *)malloc(sizeof(hash_key_bitmap_val_t) * C->tile_row_ptr[Ai + 1]);

            memcpy(seg_tmp_val[Ai]->key, table->key, sizeof(MatIndex) * C->tile_row_ptr[Ai + 1]);
            memcpy(seg_tmp_val[Ai]->val, table->val, sizeof(hash_key_bitmap_val_t) * C->tile_row_ptr[Ai + 1]);
        }
        else
            seg_tmp_val[Ai] = NULL;
    }
    omp_inclusive_scan(C->tile_row_ptr + 1, C->_m);
    free(all_table);

    C->_nnz = C->tile_row_ptr[C->_m];

    C->tile_col_ptr = (MatIndex *)malloc(C->_nnz * sizeof(MatIndex));

    if (with_detail_C)
    {
        C->tile_16_bitmap_ptr = (TileBitmap *)malloc(C->_nnz * sizeof(TileBitmap));
        C->tile_val_map = (MatIndex *)malloc((C->_nnz + 1) * sizeof(MatIndex));
        C->tile_off_map = (MatIndex *)malloc((C->_nnz + 1) * sizeof(MatIndex));
        C->tile_val_map[0] = C->tile_off_map[0] = 0;
    }

    // echo(debug, "Step 2: Sequential Step 1");
#pragma omp parallel for //* Step 2: 构建16*16位置信息
    for (MatIndex Ai = 0; Ai < A->_m; ++Ai)
    {
        hash_key_bitmap_t *table = seg_tmp_val[Ai];
        if (!table)
            continue;

        unsigned int _len = C->tile_row_ptr[Ai + 1] - C->tile_row_ptr[Ai];
        memcpy(C->tile_col_ptr + C->tile_row_ptr[Ai], table->key, _len * sizeof(MatIndex));

        if (with_detail_C)
        {
            for (MatIndex Cj = C->tile_row_ptr[Ai], idx = 0; Cj < C->tile_row_ptr[Ai + 1]; ++Cj, ++idx)
            {
                C->tile_16_bitmap_ptr[Cj] = table->val[idx].val16;
                C->tile_val_map[Cj + 1] = TileBitmap_popcount(table->val[idx].val16);
            }
        }
    }
    if (with_detail_C)
    {
        omp_inclusive_scan(C->tile_val_map + 1, C->_nnz);

        MatIndex level_1_nnz = C->tile_val_map[C->_nnz];
        C->tile_4_bitmap_ptr = (TileBitmap *)malloc(level_1_nnz * sizeof(TileBitmap));
        C->tile_4_offmap_ptr = (TileIndex *)malloc(level_1_nnz * sizeof(TileIndex));

#pragma omp parallel for //* Step 3: 构建两级索引
        for (MatIndex Ai = 0; Ai < A->_m; ++Ai)
        {
            hash_key_bitmap_t *table = seg_tmp_val[Ai];
            if (!table)
                continue;

            for (MatIndex Cj = C->tile_row_ptr[Ai], iCj = 0; Cj < C->tile_row_ptr[Ai + 1]; ++Cj, ++iCj)
            {
                MatIndex val_map_Cj = C->tile_val_map[Cj];
                TileBitmap *dst = C->tile_4_bitmap_ptr + val_map_Cj, *src = table->val[iCj].val4;

#pragma unroll
                for (TileIndex i = 0, idx = 0; i < 16; ++i)
                {
                    if (src[i])
                        dst[idx++] = src[i];
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
        C->meta_nnz = C->tile_off_map[C->_nnz];
    }
    free(seg_tmp_val);
    // mempool_free(&mem);

    // omp_inclusive_scan(C->tile_off_map + 1, C->_nnz);
    // C->tile_val_ptr = (MatValue *)calloc(C->meta_nnz, sizeof(MatValue));
    // echo(debug, "level 3 nnz: %d", C->meta_nnz);
    return C;
}

TileMatrix *TileMatrix_GEMM_Sym_v2(TileMatrix *A, TileMatrix *B)
{
    TileMatrix *C = (TileMatrix *)malloc(sizeof(TileMatrix));
    C->meta_m = A->meta_m;
    C->meta_n = B->meta_n;
    C->_m = A->_m;
    C->_n = B->_n;

    MatIndex max_threads = omp_get_max_threads();
    C->tile_row_ptr = (MatIndex *)calloc((C->_m + 1), sizeof(MatIndex));
    unsigned int _size = nextPow2(C->_n);
    hash_key_bitmap_t **all_table = (hash_key_bitmap_t **)malloc(sizeof(hash_key_bitmap_t *) * max_threads);

#pragma omp parallel for
    for (MatIndex i = 0; i < max_threads; ++i)
    {
        all_table[i] = (hash_key_bitmap_t *)malloc(sizeof(hash_key_bitmap_t));
        all_table[i]->key = (MatIndex *)malloc(sizeof(MatIndex) * _size);
        all_table[i]->val = (hash_key_bitmap_val_t *)malloc(sizeof(hash_key_bitmap_val_t) * _size);
    }

    hash_key_bitmap_t **seg_tmp_val = (hash_key_bitmap_t **)malloc(sizeof(hash_key_bitmap_t *) * A->_m);

    echo(debug, "Step 1: building hash structure");
#pragma omp parallel for //* Step 1: 计算16 * 16的非零元数
    for (MatIndex Ai = 0; Ai < A->_m; ++Ai)
    {
        MatIndex tid = omp_get_thread_num();
        hash_key_bitmap_t *table = all_table[tid];

        memset(table->key, -1, sizeof(MatIndex) * _size);
        memset(table->val, 0, sizeof(hash_key_bitmap_val_t) * _size);

        for (MatIndex Aj = A->tile_row_ptr[Ai]; Aj < A->tile_row_ptr[Ai + 1]; ++Aj)
        {
            MatIndex cur_A_16_col = A->tile_col_ptr[Aj];
            TileBitmap cur_A_16_bitmap = A->tile_16_bitmap_ptr[Aj];

            for (MatIndex Bj = B->tile_row_ptr[cur_A_16_col]; Bj < B->tile_row_ptr[cur_A_16_col + 1]; ++Bj)
            {
                update_sym_hash(
                    cur_A_16_bitmap, A->tile_4_bitmap_ptr + A->tile_val_map[Aj],
                    B->tile_16_bitmap_ptr[Bj], B->tile_4_bitmap_ptr + B->tile_val_map[Bj],
                    table, _size, B->tile_col_ptr[Bj]);
            }
        }
        C->tile_row_ptr[Ai + 1] = gather_hash(table, _size);
        if (C->tile_row_ptr[Ai + 1])
        {
            quicksort_key_val(table->key, table->val, 0, C->tile_row_ptr[Ai + 1] - 1);

            seg_tmp_val[Ai] = (hash_key_bitmap_t *)malloc(sizeof(hash_key_bitmap_t));
            seg_tmp_val[Ai]->key = (MatIndex *)malloc(sizeof(MatIndex) * C->tile_row_ptr[Ai + 1]);
            seg_tmp_val[Ai]->val = (hash_key_bitmap_val_t *)malloc(sizeof(hash_key_bitmap_val_t) * C->tile_row_ptr[Ai + 1]);

            memcpy(seg_tmp_val[Ai]->key, table->key, sizeof(MatIndex) * C->tile_row_ptr[Ai + 1]);
            memcpy(seg_tmp_val[Ai]->val, table->val, sizeof(hash_key_bitmap_val_t) * C->tile_row_ptr[Ai + 1]);
        }
        else
            seg_tmp_val[Ai] = NULL;
    }
    omp_inclusive_scan(C->tile_row_ptr + 1, C->_m);
    free(all_table);

    C->_nnz = C->tile_row_ptr[C->_m];

    C->tile_col_ptr = (MatIndex *)malloc(C->_nnz * sizeof(MatIndex));
    C->tile_16_bitmap_ptr = (TileBitmap *)malloc(C->_nnz * sizeof(TileBitmap));
    C->tile_val_map = (MatIndex *)malloc((C->_nnz + 1) * sizeof(MatIndex));
    C->tile_off_map = (MatIndex *)malloc((C->_nnz + 1) * sizeof(MatIndex));

    C->tile_val_map[0] = C->tile_off_map[0] = 0;

    echo(debug, "Step 2: Sequential Step 1, _nnz = %d", C->_nnz);
#pragma omp parallel for //* Step 2: 构建16*16位置信息
    for (MatIndex Ai = 0; Ai < A->_m; ++Ai)
    {
        hash_key_bitmap_t *table = seg_tmp_val[Ai];
        if (!table)
            continue;

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

    echo(debug, "Step 3: Sequential Step 2, l1nnz: %d", level_1_nnz);
#pragma omp parallel for //* Step 3: 构建两级索引
    for (MatIndex Ai = 0; Ai < A->_m; ++Ai)
    {
        hash_key_bitmap_t *table = seg_tmp_val[Ai];
        if (!table)
            continue;

        for (MatIndex Cj = C->tile_row_ptr[Ai], iCj = 0; Cj < C->tile_row_ptr[Ai + 1]; ++Cj, ++iCj)
        {
            MatIndex val_map_Cj = C->tile_val_map[Cj];
            TileBitmap *dst = C->tile_4_bitmap_ptr + val_map_Cj, *src = table->val[iCj].val4;

#pragma unroll
            for (TileIndex i = 0, idx = 0; i < 16; ++i)
            {
                if (src[i])
                    dst[idx++] = src[i];
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

    omp_inclusive_scan(C->tile_off_map + 1, C->_nnz);
    C->meta_nnz = C->tile_off_map[C->_nnz];
    return C;
}

template <int mac_num>
task_results TileMatrix_GEMM_Num_Sim(TileMatrix *A, TileMatrix *B, TileMatrix *C)
{
    task_results res;

#pragma omp parallel for reduction(+ : res)
    for (MatIndex Ai = 0; Ai < A->_m; ++Ai)
    {
        const MatIndex cur_C_row_len = C->tile_row_ptr[Ai + 1] - C->tile_row_ptr[Ai];
        MatIndex *cur_C_col_ptr = C->tile_col_ptr + C->tile_row_ptr[Ai];
        task_results tmp_result;

        for (MatIndex Aj = A->tile_row_ptr[Ai]; Aj < A->tile_row_ptr[Ai + 1]; ++Aj)
        {
            MatIndex cur_A_16_col = A->tile_col_ptr[Aj];
            TileBitmap cur_A_16_bitmap = A->tile_16_bitmap_ptr[Aj];
            TileBitmap *cur_A_4_bitmap_ptr = A->tile_4_bitmap_ptr + A->tile_val_map[Aj];
            bool load_a = true;

            for (MatIndex Bj = B->tile_row_ptr[cur_A_16_col]; Bj < B->tile_row_ptr[cur_A_16_col + 1]; ++Bj)
            {
                TileBitmap cur_B_16_bitmap = B->tile_16_bitmap_ptr[Bj], x = TileBitmap_x(cur_A_16_bitmap, cur_B_16_bitmap);
                if (!x)
                    continue;
                MatIndex Cj = _binary_find_index(cur_C_col_ptr, cur_C_row_len, B->tile_col_ptr[Bj]);
                if (Cj == -1)
                    continue;

                TileBitmap *cur_B_4_bitmap_ptr = B->tile_4_bitmap_ptr + B->tile_val_map[Bj];
                ++tmp_result.mp16;
                tmp_result.mp4 += TileBitmap_x_mpd(cur_A_16_bitmap, cur_B_16_bitmap);

                simulate_result_16 _nv, _gamma, _trapezoid, _ds, _rm, _smu; // 中间积数量 / Cycle数

                Tile_16_M_x_M<mac_num>(
                    cur_A_16_bitmap, cur_A_4_bitmap_ptr,
                    cur_B_16_bitmap, cur_B_4_bitmap_ptr,
                    // cur_C_16_bitmap, cur_C_4_bitmap_ptr,
                    _nv, _gamma, _trapezoid, _ds, _rm, _smu,
                    load_a);
                tmp_result.add_simulate_result(_nv, 0);
                tmp_result.add_simulate_result(_ds, 1);
                tmp_result.add_simulate_result(_rm, 2);
                tmp_result.add_simulate_result(_smu, 3);
                tmp_result.add_simulate_result(_trapezoid, 4);
                tmp_result.add_simulate_result(_gamma, 5);
            }
        }
        res += tmp_result;
    }

    u_int64_t nv_cycle = 4096 / mac_num * res.mp16;
    res.ds_mac = res.mpnz * 1.0 / mac_num / res.ds_cycle * 100.0;
    res.rm_mac = res.mpnz * 1.0 / mac_num / res.rm_cycle * 100.0;
    res.smu_mac = res.mpnz * 1.0 / mac_num / res.smu_cycle * 100.0;
    res.trapezoid_mac = res.mpnz * 1.0 / mac_num / res.trapezoid_cycle * 100.0;
    res.gamma_mac = res.mpnz * 1.0 / mac_num / res.gamma_cycle * 100.0;

    double rm_net_rate = res.rm_c_ntr * 1.0 / res.rm_c_nut;
    double smu_net_rate = res.smu_c_ntr * 1.0 / res.smu_c_nut;
    rm_net_rate *= rm_net_rate;
    smu_net_rate *= smu_net_rate;

    echo(markdown, "## MAC: %d@FP64, 中间积: %lu", mac_num, res.mpnz);
    echo(markdown, "|STC|Cycle|Mac Rate|能耗 (mJ)|A NTR|B NTR|C NTR|A ENE|B ENE|C ENE|C NEP|DNS|ALL NTR|ALL NEP|\\n|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\\n|NV-STC|%lu|%.2lf%%|---|---|---|---|---|---|---|---|---|---|---|\\n|GAMMA|%lu|%.2lf%%|%.2lf|%lu|%lu|%lu|%.2lf|%.2lf|%.2lf|%.2lf|%.2lf|%lu|%.2lf|\\n|Trapezoid|%lu|%.2lf%%|%.2lf|%lu|%lu|%lu|%.2lf|%.2lf|%.2lf|%.2lf|%.2lf|%lu|%.2lf|\\n|DS-STC|%lu|%.2lf%%|%.2lf|%lu|%lu|%lu|%.2lf|%.2lf|%.2lf|%.2lf|%.2lf|%lu|%.2lf|\\n|RM-STC|%lu|%.2lf%%|%.2lf|%lu|%lu|%lu|%.2lf|%.2lf|%.2lf|%.2lf|%.2lf|%lu|%.2lf|\\n|**SMU**|***%lu***|***%.2lf%%***|***%.2lf***|***%lu***|***%lu***|***%lu***|***%.2lf***|***%.2lf***|***%.2lf***|***%.2lf***|***%.2lf***|***%lu***|***%.2lf***|",
        nv_cycle, res.mpnz * 1.0 / mac_num / nv_cycle * 100.0,

        res.gamma_cycle, res.gamma_mac, -1.0,
        0, 0, 0,
        -1.0, -1.0, -1.0,
        -1.0, -1.0,
        0, -1.0,

        res.trapezoid_cycle, res.trapezoid_mac, -1.0,
        res.trapezoid_a_ntr, res.trapezoid_b_ntr, res.trapezoid_c_ntr,
        res.trapezoid_a_ntr * 0.6875, res.trapezoid_b_ntr * 0.6875, res.trapezoid_c_ntr * 0.6875,
        -1.0, -1.0,
        res.trapezoid_a_ntr + res.trapezoid_b_ntr + res.trapezoid_c_ntr,
        res.trapezoid_a_ntr * 0.6875 + res.trapezoid_b_ntr * 0.6875 + res.trapezoid_c_ntr * 0.6875,

        res.ds_cycle, res.ds_mac, res.ds_energy, 
        res.ds_a_ntr, res.ds_b_ntr, res.ds_c_ntr, 
        res.ds_a_ntr * 0.0625, res.ds_b_ntr * 0.0625, res.ds_c_ntr * 2.0, 
        1.0, 16.0,
        res.ds_a_ntr + res.ds_b_ntr + res.ds_c_ntr,
        res.ds_a_ntr * 0.0625 + res.ds_b_ntr * 0.0625 + res.ds_c_ntr * 2.0,

        res.rm_cycle, res.rm_mac, res.rm_energy, 
        res.rm_a_ntr, res.rm_b_ntr, res.rm_c_ntr, 
        res.rm_a_ntr * 0.0625, res.rm_b_ntr * 0.25, res.rm_c_ntr * 0.5,
        res.ds_c_ntr * 1.0 / res.rm_c_ntr * 4.0, 1.0,
        res.rm_a_ntr + res.rm_b_ntr + res.rm_c_ntr,
        res.rm_a_ntr * 0.0625 + res.rm_b_ntr * 0.25 + res.rm_c_ntr * 0.5,

        res.smu_cycle, res.smu_mac, res.smu_energy, 
        res.smu_a_ntr, res.smu_b_ntr, res.smu_c_ntr, 
        res.smu_a_nut * 0.6875, res.smu_b_nut * 0.6875, res.smu_c_nut * 0.6875,
        res.ds_c_ntr * 1.0 / res.smu_c_ntr * 2.83, 2.0 / smu_net_rate,
        res.smu_a_ntr + res.smu_b_ntr + res.smu_c_ntr,
        res.smu_a_nut * 0.6875 + res.smu_b_nut * 0.6875 + res.smu_c_nut * 0.6875
    );

    echo(markdown, "|STC|Mac Rate < 12.5%%|Mac Rate < 25%%|Mac Rate < 50%%|Mac Rate < 75%%|Mac Rate < 100%%|\\n|---|:---:|:---:|:---:|:---:|:---:|\\n|NV-STC|%lu(%.2lf\%)|%lu(%.2lf\%)|%lu(%.2lf\%)|%lu(%.2lf\%)|%lu(%.2lf\%)|\\n|RM-STC|%lu(%.2lf\%)|%lu(%.2lf\%)|%lu(%.2lf\%)|%lu(%.2lf\%)|%lu(%.2lf\%)|\\n|DS-STC|%lu(%.2lf\%)|%lu(%.2lf\%)|%lu(%.2lf\%)|%lu(%.2lf\%)|%lu(%.2lf\%)|\\n|RM-STC|%lu(%.2lf\%)|%lu(%.2lf\%)|%lu(%.2lf\%)|%lu(%.2lf\%)|%lu(%.2lf\%)|\\n|**SMU**|***%lu(%.2lf\%)***|***%lu(%.2lf\%)***|***%lu(%.2lf\%)***|***%lu(%.2lf\%)***|***%lu(%.2lf\%)***|",
         res.nv_mt125, res.nv_mt125* 1.0 / nv_cycle * 100.0, res.nv_mt25, res.nv_mt25 * 1.0 / nv_cycle * 100.0, res.nv_mt50, res.nv_mt50 * 1.0 / nv_cycle * 100.0, res.nv_mt75, res.nv_mt75 * 1.0 / nv_cycle * 100.0, res.nv_mt100, res.nv_mt100 * 1.0 / nv_cycle * 100.0,
         res.ds_mt125, res.ds_mt125* 1.0 / res.ds_cycle * 100.0, res.ds_mt25, res.ds_mt25 * 1.0 / res.ds_cycle * 100.0, res.ds_mt50, res.ds_mt50 * 1.0 / res.ds_cycle * 100.0, res.ds_mt75, res.ds_mt75 * 1.0 / res.ds_cycle * 100.0, res.ds_mt100, res.ds_mt100 * 1.0 / res.ds_cycle * 100.0,
         res.rm_mt125, res.rm_mt125* 1.0 / res.rm_cycle * 100.0, res.rm_mt25, res.rm_mt25 * 1.0 / res.rm_cycle * 100.0, res.rm_mt50, res.rm_mt50 * 1.0 / res.rm_cycle * 100.0, res.rm_mt75, res.rm_mt75 * 1.0 / res.rm_cycle * 100.0, res.rm_mt100, res.rm_mt100 * 1.0 / res.rm_cycle * 100.0,
         res.smu_mt125, res.smu_mt125* 1.0 / res.smu_cycle * 100.0, res.smu_mt25, res.smu_mt25 * 1.0 / res.smu_cycle * 100.0, res.smu_mt50, res.smu_mt50 * 1.0 / res.smu_cycle * 100.0, res.smu_mt75, res.smu_mt75 * 1.0 / res.smu_cycle * 100.0, res.smu_mt100, res.smu_mt100 * 1.0 / res.smu_cycle * 100.0);

    echo(markdown, "|STC|Network Rate < 12.5%%|Network Rate < 25%%|Network Rate < 50%%|Network Rate < 75%%|Network Rate < 100%%|\\n|---|:---:|:---:|:---:|:---:|:---:|\\n|DS-STC|%lld|%lld|%lld|%lld|%lld|\\n|RM-STC|%lld|%lld|%lld|%lld|%lld|\\n|**SMU**|***%lld***|***%lld***|***%lld***|***%lld***|***%lld***|",
         res.ds_nt125, res.ds_nt25, res.ds_nt50, res.ds_nt75, res.ds_nt100,
         res.rm_nt125, res.rm_nt25, res.rm_nt50, res.rm_nt75, res.rm_nt100,
         res.smu_nt125, res.smu_nt25, res.smu_nt50, res.smu_nt75, res.smu_nt100);

    // res.rm_c_nut = 1.0 / rm_net_rate;
    // res.smu_c_nut = 2.0 / smu_net_rate;

    return res;
}
