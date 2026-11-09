#pragma once
#include <tmatrix/Utils/timer.h>
#include <tmatrix/MMIO/mmio_highlevel.h>
#include <tmatrix/DataStructure/Factory.h>
#include <tmatrix/DataStructure/transform.h>
#include <tmatrix/Calculation/TileMatrix_x.h>

TileMatrix* load_matrix(const char*filepath)
{
    MatIndex m, n, nnz;
    MatIndex *row_ptr, *col_ptr;
    MatValue *val_ptr;
    Timer read_mtx, csr_to_seqtile;

    char tile_path[128];
    sprintf(tile_path, "%s.tile", filepath);
    bool exists = file_exists(tile_path);

    if (!exists) {
        int isSymmetric;
        echo(start_status, "Reading Matrix file: [bold green]%s[/]", filepath);
        timer_start(read_mtx);
        int status = mmio_allinone(filepath, &m, &n, &nnz, &isSymmetric, &row_ptr, &col_ptr, &val_ptr, true);
        timer_end(read_mtx);
        echo(stop_status, "");
        if (status)
        {
            echo(error, "Matrix is not square, please use square matrix");
            return nullptr;
        }
        echo(success, "m = %d, n = %d, nnz = %d; time: %f ms", m, n, nnz, timer_duration(read_mtx));
    }

    echo(start_status, "Converting CSR to TileMatrix");
    timer_start(csr_to_seqtile);
    TileMatrix *seqtile = CSR_to_TileMatrix(m, n, nnz, row_ptr, col_ptr, val_ptr, filepath, true, false, true);
    if (seqtile == nullptr)
    {
        echo(error, "Matrix is not symmetric, please use symmetric matrix");
        echo(stop_status, "");
        return nullptr;
    }

    symbolRateCalculation(seqtile);

    timer_end(csr_to_seqtile);
    echo(stop_status, "");
    echo(success, "Convert CSR to TileMatrix time: %f ms, m: %d, _nnz: %d, l1nnz: %d, nnz: %d", timer_duration(csr_to_seqtile), seqtile->meta_m, seqtile->_nnz, seqtile->tile_val_map[seqtile->_nnz], seqtile->meta_nnz);

    if (!exists) {
        free(row_ptr);
        free(col_ptr);
        free(val_ptr);
    }

    return seqtile;
}

int test_SpGEMM(const char *filepath, const char*filepath2 = nullptr)
{
    TileMatrix* seqtile_A = load_matrix(filepath), *seqtile_B;
    if (seqtile_A == nullptr) return 1;
    if (filepath2 != nullptr) {
        seqtile_B = load_matrix(filepath2);
        if (seqtile_B == nullptr) return 1;
    } else seqtile_B = seqtile_A;
    
    echo(start_status, "TMX Symbolic GEMM");
    TileMatrix *seqtile_c = TileMatrix_GEMM_Sym_Sim(seqtile_A, seqtile_B);
    // echo(info, "TMX Sym: _nnz: %d, l1nnz: %d, nnz: %d", seqtile_c->_nnz, seqtile_c->tile_val_map[seqtile_c->_nnz], seqtile_c->meta_nnz);
    echo(info, "TMX Sym: _nnz: %d", seqtile_c->_nnz);
    // symbolRateCalculation(seqtile_c);
    echo(start_status, "Running Simulation 128 MAC...");
    task_results t32 = TileMatrix_GEMM_Num_Sim<128>(seqtile_A, seqtile_B, seqtile_c), t64;
    if (seqtile_B == seqtile_A) {
        echo(start_status, "Running Simulation 64 MAC...");
        t64 = TileMatrix_GEMM_Num_Sim<64>(seqtile_A, seqtile_B, seqtile_c);
    }
    echo(stop_status, "");

    if (filepath2 == nullptr) {
        printf("%d,%d,%d,%lu,%lu,%lu,", seqtile_A->meta_m, seqtile_A->meta_n, seqtile_A->meta_nnz, t32.mp16, t32.mp4, t32.mpnz);
        print_task_result(t32, t64);
        fflush(stdout);
        destroy_TileMatrix(seqtile_A);
    } else {
        double a_sparsity = seqtile_A->meta_nnz * 1.0 / seqtile_A->meta_m / seqtile_A->meta_n;
        double b_sparsity = seqtile_B->meta_nnz * 1.0 / seqtile_B->meta_m / seqtile_B->meta_n;
        printf("%.1lf-%.1lf,%.2lf,%.2lf,%.2lf,%.2lf,%.2lf\n", a_sparsity, b_sparsity, t32.gamma_mac, t32.trapezoid_mac, t32.ds_mac, t32.rm_mac, t32.smu_mac);
    }
    if (seqtile_B != seqtile_A) destroy_TileMatrix(seqtile_B);
    destroy_TileMatrix(seqtile_c, true, false);

    return 0;
}