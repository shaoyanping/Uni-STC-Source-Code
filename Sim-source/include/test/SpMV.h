#pragma once
#include <tmatrix/Utils/timer.h>
#include <tmatrix/MMIO/mmio_highlevel.h>
#include <tmatrix/DataStructure/Factory.h>
#include <tmatrix/DataStructure/transform.h>
#include <tmatrix/Calculation/TileMatrix_x_v.h>

void test_SpMV(const char *filepath)
{
    MatIndex m, n, nnz;
    MatIndex *row_ptr, *col_ptr;
    MatValue *val_ptr;
    Timer read_mtx, csr_to_seqtile;
    bool exists;

    echo(title, "test SpMV on %s", filepath);

    char tile_path[128];
    sprintf(tile_path, "%s.tile", filepath);
    exists = file_exists(tile_path);
    if (!exists) {
        int isSymmetric;
        echo(start_status, "Reading Matrix file: [bold green]%s[/]", filepath);
        timer_start(read_mtx);
        mmio_allinone(filepath, &m, &n, &nnz, &isSymmetric, &row_ptr, &col_ptr, &val_ptr);
        timer_end(read_mtx);
        echo(stop_status, "");
        echo(success, "m = %d, n = %d, nnz = %d; time: %f ms", m, n, nnz, timer_duration(read_mtx));
    }
    echo(start_status, "CSR to TileMatrix");

    timer_start(csr_to_seqtile);
    TileMatrix *seqtile = CSR_to_TileMatrix(m, n, nnz, row_ptr, col_ptr, val_ptr, filepath);
    timer_end(csr_to_seqtile);
    echo(stop_status, "");
    echo(success, "CSR to TileMatrix time: %f ms", timer_duration(csr_to_seqtile));

    if (!exists) {
        free(row_ptr);
        free(col_ptr);
        free(val_ptr);
    }

    echo(start_status, "Prepare SpMV test data");
    MatValue *v = (MatValue *)malloc(n * sizeof(MatValue));
    MatValue *tile_res = (MatValue *)malloc(m * sizeof(MatValue));

#pragma omp parallel for
    for (MatIndex i = 0; i < n; ++i)
        v[i] = 1;
    echo(stop_status, "");
    echo(success, "SpMV test data prepared");
    
    echo(start_status, "Simulate SpMV 128 MAC");
    task_results t32 = TileMatrixSpMV<128>(seqtile);
    echo(start_status, "Simulate SpMV 64 MAC");
    task_results t64 = TileMatrixSpMV<64>(seqtile);
    echo(stop_status, "");
    
    free(v);
    free(tile_res);
    
    // print seqtile's meta info
    printf("%d,%d,%d,%lu,%lu,%lu,", seqtile->meta_m, seqtile->meta_n, seqtile->meta_nnz, t32.mp16, t32.mp4, t32.mpnz);

    print_task_result(t32, t64);
    fflush(stdout);
    destroy_TileMatrix(seqtile);
}
