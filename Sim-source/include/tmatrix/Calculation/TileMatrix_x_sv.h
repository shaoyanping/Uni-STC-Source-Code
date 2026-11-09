#pragma once
#include <msg.h>
#include "./bitmap.h"
#include <tmatrix/DataStructure/TileMatrix.h>
#include "./smu_sim.h"


template <int mac_num>
task_results TileMatrixSpMSpV(TileMatrix *tm, bool using_random_vector)
{
    task_results res;

#pragma omp parallel for reduction(+ : res)
    for (MatIndex i = 0; i < tm->_m; ++i)
    {
        MatIndex row_end = tm->tile_row_ptr[i + 1];
        task_results tmp_result;
        bool read_y = true;
        TileBitmap bv = 0xa5a5;

        tmp_result.mp16 = row_end - tm->tile_row_ptr[i];
        for (MatIndex j = tm->tile_row_ptr[i]; j < row_end; j += 2) // 俩任务拼一起
        {
            simulate_result_16 _gamma, _trapezoid, _ds, _rm, _smu; // 中间积数量 / Cycle数
            TileMatrixSpMSpV_16<mac_num>(
                tm->tile_16_bitmap_ptr[j],
                tm->tile_4_bitmap_ptr + tm->tile_val_map[j],
                j + 1 < row_end ? tm->tile_16_bitmap_ptr[j + 1] : 0,
                j + 1 < row_end ? tm->tile_4_bitmap_ptr + tm->tile_val_map[j + 1] : nullptr,
                _gamma, _trapezoid, _ds, _rm, _smu, read_y, bv);
            tmp_result.add_simulate_result(_ds, 1);
            tmp_result.add_simulate_result(_rm, 2);
            tmp_result.add_simulate_result(_smu, 3);
            tmp_result.add_simulate_result(_trapezoid, 4);
            tmp_result.add_simulate_result(_gamma, 5);
            tmp_result.mp4 += TileBitmap_x_mpd(tm->tile_16_bitmap_ptr[j], 0x1111);
            if (j + 1 < row_end) tmp_result.mp4 += TileBitmap_x_mpd(tm->tile_16_bitmap_ptr[j + 1], 0x1111);

            if (read_y) read_y = false;
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
