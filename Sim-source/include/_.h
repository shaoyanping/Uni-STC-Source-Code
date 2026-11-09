#pragma once

#include <omp.h>
#include <math.h>
#include <stdio.h>

#define TileLevel16_bit 4

typedef unsigned char TileIndex;
typedef unsigned int MatIndex;
typedef double MatValue;
typedef unsigned short ushort;
typedef ushort TileBitmap;

#define TileIndex_size 2  // bits (uint 2)
#define MatIndex_size 32  // bits (uint 32)
#define MatValue_size 64  // bits (double 64)

#define ushort_popcount(x) __builtin_popcount(x)
#define TileBitmap_lowbit(bitmap) ((bitmap) & (-(bitmap)))
#define TileBitmap_popcount(x) __builtin_popcount(x)
#define TileBitmap_ctz(bitmap) __builtin_ctz(bitmap)

#define BSR_BlockSz_bit 4
#define TileBlockSz_bit 2
#define BSR_BlockSz 16
#define TileBlockSz 4
#define TileBlockSz_negetive_one 0xf
#define negetive_one 0xff

// #define max(x, y) ((x) > (y) ? (x) : (y))
// #define min(x, y) ((x) > (y) ? (y) : (x))

#define multiply_5(x) ((x << 2) + x)

typedef enum {
    False,
    True
} Bool;

struct simulate_result_16
{
    u_int16_t cycle, middle_product, nnz;
    u_int16_t mt125, mt25, mt50, mt75, mt100;
    u_int16_t nt125, nt25, nt50, nt75, nt100;
    u_int16_t C_ntr, A_ntr, B_ntr;
    double energy_cost, mac_rate, A_nut, B_nut, C_nut;

    simulate_result_16()
    {
        reset();
    }

    void reset(){
        cycle = middle_product = nnz = 0;
        A_nut = A_ntr = B_nut = B_ntr = C_nut = C_ntr = 0;
        nt125 = nt25 = nt50 = nt75 = nt100 = 0;
        mt125 = mt25 = mt50 = mt75 = mt100 = 0;
        energy_cost = 0;
    }

    // += 
    simulate_result_16& operator+=(const simulate_result_16 &a)
    {
        cycle += a.cycle;
        middle_product += a.middle_product;
        C_nut += a.C_nut;
        C_ntr += a.C_ntr;
        A_nut += a.A_nut;
        A_ntr += a.A_ntr;
        B_nut += a.B_nut;
        B_ntr += a.B_ntr;

        nt125 += a.nt125;
        nt25 += a.nt25;
        nt50 += a.nt50;
        nt75 += a.nt75;
        nt100 += a.nt100;
        
        mt125 += a.mt125;
        mt25 += a.mt25;
        mt50 += a.mt50;
        mt75 += a.mt75;
        mt100 += a.mt100;
        energy_cost += a.energy_cost;
        return *this;
    }
};

struct task_results
{
    u_int64_t ds_cycle, rm_cycle, smu_cycle, trapezoid_cycle, gamma_cycle, mp16, mp4, mpnz;

    u_int64_t ds_nt125, ds_nt25, ds_nt50, ds_nt75, ds_nt100;
    u_int64_t rm_nt125, rm_nt25, rm_nt50, rm_nt75, rm_nt100;
    u_int64_t smu_nt125, smu_nt25, smu_nt50, smu_nt75, smu_nt100;

    u_int64_t nv_mt125, nv_mt25, nv_mt50, nv_mt75, nv_mt100;
    u_int64_t ds_mt125, ds_mt25, ds_mt50, ds_mt75, ds_mt100;
    u_int64_t rm_mt125, rm_mt25, rm_mt50, rm_mt75, rm_mt100;
    u_int64_t smu_mt125, smu_mt25, smu_mt50, smu_mt75, smu_mt100;

    u_int64_t ds_a_ntr, ds_b_ntr, ds_c_ntr;
    u_int64_t rm_a_ntr, rm_b_ntr, rm_c_ntr;
    u_int64_t smu_a_ntr, smu_b_ntr, smu_c_ntr;
    u_int64_t trapezoid_a_ntr, trapezoid_b_ntr, trapezoid_c_ntr;

    double ds_a_nut, ds_b_nut, ds_c_nut;
    double rm_a_nut, rm_b_nut, rm_c_nut;
    double smu_a_nut, smu_b_nut, smu_c_nut;

    double ds_mac, rm_mac, smu_mac, trapezoid_mac, gamma_mac;
    double ds_energy, rm_energy, smu_energy, trapezoid_energy, gamma_energy;
    double smu_speedup;

    task_results()
    {
        ds_cycle = rm_cycle = smu_cycle = trapezoid_cycle = gamma_cycle = 0;
        mp16 = mp4 = mpnz = 0;
        gamma_mac = trapezoid_mac = ds_mac = rm_mac = smu_mac = ds_energy = rm_energy = smu_energy = trapezoid_energy = gamma_energy = smu_speedup = 0;

        ds_nt125 = ds_nt25 = ds_nt50 = ds_nt75 = ds_nt100 = 0;
        rm_nt125 = rm_nt25 = rm_nt50 = rm_nt75 = rm_nt100 = 0;
        smu_nt125 = smu_nt25 = smu_nt50 = smu_nt75 = smu_nt100 = 0;

        nv_mt125 = nv_mt25 = nv_mt50 = nv_mt75 = nv_mt100 = 0;
        ds_mt125 = ds_mt25 = ds_mt50 = ds_mt75 = ds_mt100 = 0;
        rm_mt125 = rm_mt25 = rm_mt50 = rm_mt75 = rm_mt100 = 0;
        smu_mt125 = smu_mt25 = smu_mt50 = smu_mt75 = smu_mt100 = 0;
        ds_a_ntr = ds_a_nut = ds_b_ntr = ds_b_nut = ds_c_ntr = ds_c_nut = 0;
        rm_a_ntr = rm_a_nut = rm_b_ntr = rm_b_nut = rm_c_ntr = rm_c_nut = 0;
        smu_a_ntr = smu_a_nut = smu_b_ntr = smu_b_nut = smu_c_ntr = smu_c_nut = 0;
        trapezoid_a_ntr = trapezoid_b_ntr = trapezoid_c_ntr = 0;
    }

    task_results& operator+=(const task_results &a)
    {
        ds_cycle += a.ds_cycle;
        rm_cycle += a.rm_cycle;
        smu_cycle += a.smu_cycle;
        mp16 += a.mp16;
        mp4 += a.mp4;
        mpnz += a.mpnz;
        ds_energy += a.ds_energy;
        rm_energy += a.rm_energy;
        smu_energy += a.smu_energy;
        ds_a_ntr += a.ds_a_ntr, ds_a_nut += a.ds_a_nut, ds_b_ntr += a.ds_b_ntr, ds_b_nut += a.ds_b_nut, ds_c_ntr += a.ds_c_ntr, ds_c_nut += a.ds_c_nut;
        rm_a_ntr += a.rm_a_ntr, rm_a_nut += a.rm_a_nut, rm_b_ntr += a.rm_b_ntr, rm_b_nut += a.rm_b_nut, rm_c_ntr += a.rm_c_ntr, rm_c_nut += a.rm_c_nut;
        smu_a_ntr += a.smu_a_ntr, smu_a_nut += a.smu_a_nut, smu_b_ntr += a.smu_b_ntr, smu_b_nut += a.smu_b_nut, smu_c_ntr += a.smu_c_ntr, smu_c_nut += a.smu_c_nut;
        ds_nt125 += a.ds_nt125, ds_nt25 += a.ds_nt25, ds_nt50 += a.ds_nt50, ds_nt75 += a.ds_nt75, ds_nt100 += a.ds_nt100;
        rm_nt125 += a.rm_nt125, rm_nt25 += a.rm_nt25, rm_nt50 += a.rm_nt50, rm_nt75 += a.rm_nt75, rm_nt100 += a.rm_nt100;
        smu_nt125 += a.smu_nt125, smu_nt25 += a.smu_nt25, smu_nt50 += a.smu_nt50, smu_nt75 += a.smu_nt75, smu_nt100 += a.smu_nt100;
        nv_mt125 += a.nv_mt125, nv_mt25 += a.nv_mt25, nv_mt50 += a.nv_mt50, nv_mt75 += a.nv_mt75, nv_mt100 += a.nv_mt100;
        ds_mt125 += a.ds_mt125, ds_mt25 += a.ds_mt25, ds_mt50 += a.ds_mt50, ds_mt75 += a.ds_mt75, ds_mt100 += a.ds_mt100;
        rm_mt125 += a.rm_mt125, rm_mt25 += a.rm_mt25, rm_mt50 += a.rm_mt50, rm_mt75 += a.rm_mt75, rm_mt100 += a.rm_mt100;
        smu_mt125 += a.smu_mt125, smu_mt25 += a.smu_mt25, smu_mt50 += a.smu_mt50, smu_mt75 += a.smu_mt75, smu_mt100 += a.smu_mt100;

        trapezoid_a_ntr += a.trapezoid_a_ntr;
        trapezoid_b_ntr += a.trapezoid_b_ntr;
        trapezoid_c_ntr += a.trapezoid_c_ntr;
        trapezoid_cycle += a.trapezoid_cycle;
        trapezoid_energy += a.trapezoid_energy;

        gamma_cycle += a.gamma_cycle;
        gamma_energy += a.gamma_energy;
        return *this;
    }

    void add_simulate_result(const simulate_result_16 &a, int stc_id, int times = 1)
    {
        switch (stc_id)
        {
        case 0: // NV-STC
            nv_mt125 += a.mt125 * times, nv_mt25 += a.mt25 * times, nv_mt50 += a.mt50 * times, nv_mt75 += a.mt75 * times, nv_mt100 += a.mt100 * times;
            break;
        case 1: // DS-STC
            ds_cycle += a.cycle * times;
            ds_energy += a.energy_cost * times;
            
            ds_nt125 += a.nt125 * times, ds_nt25 += a.nt25 * times, ds_nt50 += a.nt50 * times, ds_nt75 += a.nt75 * times, ds_nt100 += a.nt100 * times;
            ds_mt125 += a.mt125 * times, ds_mt25 += a.mt25 * times, ds_mt50 += a.mt50 * times, ds_mt75 += a.mt75 * times, ds_mt100 += a.mt100 * times;
            ds_a_ntr += a.A_ntr * times, ds_a_nut += a.A_nut * times, ds_b_ntr += a.B_ntr * times, ds_b_nut += a.B_nut * times, ds_c_ntr += a.C_ntr * times, ds_c_nut += a.C_nut * times;
            break;
        case 2: // RM-STC
            rm_cycle += a.cycle * times;
            rm_energy += a.energy_cost * times;
            rm_nt125 += a.nt125 * times, rm_nt25 += a.nt25 * times, rm_nt50 += a.nt50 * times, rm_nt75 += a.nt75 * times, rm_nt100 += a.nt100 * times;
            rm_mt125 += a.mt125 * times, rm_mt25 += a.mt25 * times, rm_mt50 += a.mt50 * times, rm_mt75 += a.mt75 * times, rm_mt100 += a.mt100 * times;
            rm_a_ntr += a.A_ntr * times, rm_a_nut += a.A_nut * times, rm_b_ntr += a.B_ntr * times, rm_b_nut += a.B_nut * times, rm_c_ntr += a.C_ntr * times, rm_c_nut += a.C_nut * times;
            break;
        case 3: // SMU
            smu_cycle += a.cycle * times;
            smu_energy += a.energy_cost * times;
            smu_nt125 += a.nt125 * times, smu_nt25 += a.nt25 * times, smu_nt50 += a.nt50 * times, smu_nt75 += a.nt75 * times, smu_nt100 += a.nt100 * times;
            smu_mt125 += a.mt125 * times, smu_mt25 += a.mt25 * times, smu_mt50 += a.mt50 * times, smu_mt75 += a.mt75 * times, smu_mt100 += a.mt100 * times;
            smu_a_ntr += a.A_ntr * times, smu_a_nut += a.A_nut * times, smu_b_ntr += a.B_ntr * times, smu_b_nut += a.B_nut * times, smu_c_ntr += a.C_ntr * times, smu_c_nut += a.C_nut * times;
            mpnz += a.middle_product * times;
            break;
        case 4: // Trapezoid
            trapezoid_cycle += a.cycle * times;
            trapezoid_energy += a.energy_cost * times;
            trapezoid_a_ntr += a.A_ntr * times, trapezoid_b_ntr += a.B_ntr * times, trapezoid_c_ntr += a.C_ntr * times;
            break;
        case 5: // GAMMA
            gamma_cycle += a.cycle * times;
            gamma_energy += a.energy_cost * times;
        default:
            break;
        }
    }
};

Bool file_exists(const char* filename)
{
    FILE*f = fopen(filename, "r");
    if (f == NULL)
        return False;
    fclose(f);
    return True;
}

MatIndex _binary_find_index(MatIndex *col_ptr, MatIndex len, MatIndex col)
{
    int left = 0, right = len - 1;
    while (left <= right)
    {
        int mid = (left + right) >> 1;
        if (col_ptr[mid] == col)
        {
            return mid;
        }
        else if (col_ptr[mid] < col)
            left = mid + 1;
        else
            right = mid - 1;
    }
    return -1;
}

void print_task_result(task_results &t32, task_results &t64)
{
    printf(
        "%lu,%.2lf,%lu,%.2lf,%lu,%.2lf,%.2lf,%.2lf,%.2lf,%.2lf,%lu,%.2lf,%lu,%.2lf,%lu,%.2lf,%.2lf,%.2lf,%.2lf,%.2lf,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%.2lf,%.2lf,%.2lf,%.2lf,%.2lf,%.2lf,%.2lf,%.2lf,%llu,%llu\n",
        t32.ds_cycle, t32.ds_mac, t32.rm_cycle, t32.rm_mac, t32.smu_cycle, t32.smu_mac, 
        t32.ds_energy, t32.rm_energy, t32.smu_energy, t32.smu_speedup, 
        t64.ds_cycle, t64.ds_mac, t64.rm_cycle, t64.rm_mac, t64.smu_cycle, t64.smu_mac,
        t64.ds_energy, t64.rm_energy, t64.smu_energy, t64.smu_speedup,
        t32.ds_mt125, t32.ds_mt25, t32.ds_mt50, t32.ds_mt75, t32.ds_mt100, t32.rm_mt125, t32.rm_mt25, t32.rm_mt50, t32.rm_mt75, t32.rm_mt100, t32.smu_mt125, t32.smu_mt25, t32.smu_mt50, t32.smu_mt75, t32.smu_mt100,
        t64.ds_mt125, t64.ds_mt25, t64.ds_mt50, t64.ds_mt75, t64.ds_mt100, t64.rm_mt125, t64.rm_mt25, t64.rm_mt50, t64.rm_mt75, t64.rm_mt100, t64.smu_mt125, t64.smu_mt25, t64.smu_mt50, t64.smu_mt75, t64.smu_mt100,
        t32.ds_a_ntr, t32.ds_b_ntr, t32.ds_c_ntr, t32.rm_a_ntr, t32.rm_b_ntr, t32.rm_c_ntr, t32.smu_a_ntr, t32.smu_b_ntr, t32.smu_c_ntr, t64.ds_a_ntr, t64.ds_b_ntr, t64.ds_c_ntr, t64.rm_a_ntr, t64.rm_b_ntr, t64.rm_c_ntr, t64.smu_a_ntr, t64.smu_b_ntr, t64.smu_c_ntr,
        t32.ds_nt125, t32.ds_nt25, t32.ds_nt50, t32.ds_nt75, t32.ds_nt100, t32.rm_nt125, t32.rm_nt25, t32.rm_nt50, t32.rm_nt75, t32.rm_nt100, t32.smu_nt125, t32.smu_nt25, t32.smu_nt50, t32.smu_nt75, t32.smu_nt100,
        t64.ds_nt125, t64.ds_nt25, t64.ds_nt50, t64.ds_nt75, t64.ds_nt100, t64.rm_nt125, t64.rm_nt25, t64.rm_nt50, t64.rm_nt75, t64.rm_nt100, t64.smu_nt125, t64.smu_nt25, t64.smu_nt50, t64.smu_nt75, t64.smu_nt100,
        t32.rm_c_nut, t32.smu_a_nut, t32.smu_b_nut, t32.smu_c_nut, 
        t64.rm_c_nut, t64.smu_a_nut, t64.smu_b_nut, t64.smu_c_nut,
        t64.trapezoid_cycle, t64.gamma_cycle
    );
}