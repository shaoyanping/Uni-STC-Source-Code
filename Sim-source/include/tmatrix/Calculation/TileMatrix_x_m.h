#pragma once
#include <tmatrix/Calculation/smu_sim.h>
#include <tmatrix/Utils/omp_utils.h>
#include <tmatrix/Utils/hash_sim.h>
#include <tmatrix/DataStructure/TileMatrix.h>
#include <tuple>
#include <algorithm>
#include <unordered_map>

#ifndef SCHEDULER
#define SCHEDULER 8
#endif // !SCHEDULER


template <int mac_num>
task_results TileMatrix_SPMM_Num_Sim(TileMatrix *A, MatIndex B_n)
{
    TileBitmap Bb[16];
    memset(Bb, -1, sizeof(Bb));
    task_results res;

#pragma omp parallel for reduction(+ : res)
    for (MatIndex Ai = 0; Ai < A->_m; ++Ai)
    {
        task_results tmp_result;

        for (MatIndex Aj = A->tile_row_ptr[Ai]; Aj < A->tile_row_ptr[Ai + 1]; ++Aj)
        {
            TileBitmap cur_A_16_bitmap = A->tile_16_bitmap_ptr[Aj];
            TileBitmap *cur_A_4_bitmap_ptr = A->tile_4_bitmap_ptr + A->tile_val_map[Aj];
            bool load_a = true;

            tmp_result.mp16 += B_n;
            tmp_result.mp4 += TileBitmap_x_mpd(cur_A_16_bitmap, 0xffff) * B_n;

            simulate_result_16 _nv, __gamma, _trapezoid, _ds, _rm, _smu; // 中间积数量 / Cycle数
            Tile_16_M_x_M<mac_num>(
                cur_A_16_bitmap, cur_A_4_bitmap_ptr,
                0xffff, Bb,
                _nv, __gamma, _trapezoid, _ds, _rm, _smu, load_a);
            tmp_result.add_simulate_result(_nv, 0);
            tmp_result.add_simulate_result(_ds, 1);
            tmp_result.add_simulate_result(_rm, 2);
            tmp_result.add_simulate_result(_smu, 3);
            tmp_result.add_simulate_result(_trapezoid, 4);

            Tile_16_M_x_M<mac_num>(
                cur_A_16_bitmap, cur_A_4_bitmap_ptr,
                0xffff, Bb,
                _nv,
                __gamma, _trapezoid, _ds, _rm, _smu, load_a);

            tmp_result.add_simulate_result(_nv, 0, B_n - 1);
            tmp_result.add_simulate_result(_ds, 1, B_n - 1);
            tmp_result.add_simulate_result(_rm, 2, B_n - 1);
            tmp_result.add_simulate_result(_smu, 3, B_n - 1);
            tmp_result.add_simulate_result(_trapezoid, 4, B_n - 1);
        }
        res += tmp_result;
    }

    u_int64_t nv_cycle = 4096 / mac_num * res.mp16;
    res.ds_mac = res.mpnz * 1.0 / mac_num / res.ds_cycle * 100.0;
    res.rm_mac = res.mpnz * 1.0 / mac_num / res.rm_cycle * 100.0;
    res.smu_mac = res.mpnz * 1.0 / mac_num / res.smu_cycle * 100.0;
    res.trapezoid_mac = res.mpnz * 1.0 / mac_num / res.trapezoid_cycle * 100.0;

    echo(markdown, "## MAC: %d@FP64, 中间积: %lu", mac_num, res.mpnz);
    double rm_net_rate = res.rm_c_ntr * 1.0 / res.rm_c_nut;
    double smu_net_rate = res.smu_c_ntr * 1.0 / res.smu_c_nut;
    rm_net_rate *= rm_net_rate;
    smu_net_rate *= smu_net_rate;

    echo(markdown, "## MAC: %d@FP64, 中间积: %lu", mac_num, res.mpnz);
    echo(markdown, "|STC|Cycle|Mac Rate|能耗 (mJ)|A NTR|B NTR|C NTR|A ENE|B ENE|C ENE|C NEP|DNS|ALL NTR|ALL NEP|\\n|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\\n|NV-STC|%lu|%.2lf%%|---|---|---|---|---|---|---|---|---|---|---|\\n|Trapezoid|%lu|%.2lf%%|%.2lf|%lu|%lu|%lu|%.2lf|%.2lf|%.2lf|%.2lf|%.2lf|%lu|%.2lf|\\n|DS-STC|%lu|%.2lf%%|%.2lf|%lu|%lu|%lu|%.2lf|%.2lf|%.2lf|%.2lf|%.2lf|%lu|%.2lf|\\n|RM-STC|%lu|%.2lf%%|%.2lf|%lu|%lu|%lu|%.2lf|%.2lf|%.2lf|%.2lf|%.2lf|%lu|%.2lf|\\n|**SMU**|***%lu***|***%.2lf%%***|***%.2lf***|***%lu***|***%lu***|***%lu***|***%.2lf***|***%.2lf***|***%.2lf***|***%.2lf***|***%.2lf***|***%lu***|***%.2lf***|",
        nv_cycle, res.mpnz * 1.0 / mac_num / nv_cycle * 100.0,

        res.trapezoid_cycle, res.trapezoid_mac, -1,
        res.trapezoid_a_ntr, res.trapezoid_b_ntr, res.trapezoid_c_ntr,
        res.trapezoid_a_ntr * 0.6875, res.trapezoid_b_ntr * 0.6875, res.trapezoid_c_ntr * 0.6875,
        -1, -1,
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

    echo(markdown, "|STC|Mac Rate < 12.5%%|Mac Rate < 25%%|Mac Rate < 50%%|Mac Rate < 75%%|Mac Rate < 100%%|\\n|---|:---:|:---:|:---:|:---:|:---:|\\n|DS-STC|%lu(%.2lf\%)|%lu(%.2lf\%)|%lu(%.2lf\%)|%lu(%.2lf\%)|%lu(%.2lf\%)|\\n|RM-STC|%lu(%.2lf\%)|%lu(%.2lf\%)|%lu(%.2lf\%)|%lu(%.2lf\%)|%lu(%.2lf\%)|\\n|**SMU**|***%lu(%.2lf\%)***|***%lu(%.2lf\%)***|***%lu(%.2lf\%)***|***%lu(%.2lf\%)***|***%lu(%.2lf\%)***|",
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
