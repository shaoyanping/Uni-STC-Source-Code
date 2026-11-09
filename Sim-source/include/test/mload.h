#pragma once
#include <tmatrix/Utils/timer.h>
#include <tmatrix/MMIO/mmio_highlevel.h>
#include <tmatrix/DataStructure/Factory.h>
#include <tmatrix/DataStructure/transform.h>

TileMatrix* mmload(const char *filepath, bool need_square=true)
{
    MatIndex m, n, nnz;
    int isSymmetric;
    MatIndex *row_ptr, *col_ptr;
    MatValue *val_ptr;
    Timer read_mtx, csr_to_seqtile;
    bool exists;

    echo(title, "test Transformer on %s", filepath);
    char tile_path[128];
    sprintf(tile_path, "%s.tile", filepath);
    exists = file_exists(tile_path);

    if (!exists) {
        echo(start_status, "Reading Matrix file: [bold green]%s[/]", filepath);
        timer_start(read_mtx);
        mmio_allinone(filepath, &m, &n, &nnz, &isSymmetric, &row_ptr, &col_ptr, &val_ptr);
        timer_end(read_mtx);
        echo(stop_status, "");
        echo(success, "m = %d, n = %d, nnz = %d; time: %f ms", m, n, nnz, timer_duration(read_mtx));
        if (need_square && m != n)
        {
            free(row_ptr);
            free(col_ptr);
            free(val_ptr);
            echo(error, "Matrix is not square, please use square matrix");
            return nullptr;
        }
    }
    echo(start_status, "Converting CSR to TileMatrix");
    timer_start(csr_to_seqtile);
    TileMatrix *seqtile = CSR_to_TileMatrix(m, n, nnz, row_ptr, col_ptr, val_ptr, filepath);
    if (need_square && seqtile->meta_m != seqtile->meta_n)
    {
        echo(error, "Matrix is not symmetric, please use symmetric matrix");
        echo(stop_status, "");
        destroy_TileMatrix(seqtile);
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

TileMatrix* mmload2(const char *filepath, MatIndex**csr_row_ptr, MatIndex**csr_col_ptr, MatValue**csr_val, bool need_square=true)
{
    MatIndex m, n, nnz;
    int isSymmetric;
    MatIndex *row_ptr, *col_ptr;
    MatValue *val_ptr;
    Timer read_mtx, csr_to_seqtile;
    bool exists;

    echo(markdown, "# test Kernel on %s", filepath);
    echo(start_status, "Reading Matrix file: [bold green]%s[/]", filepath);
    timer_start(read_mtx);
    mmio_allinone(filepath, &m, &n, &nnz, &isSymmetric, &row_ptr, &col_ptr, &val_ptr);
    timer_end(read_mtx);
    echo(stop_status, "");
    echo(success, "m = %d, n = %d, nnz = %d; time: %f ms", m, n, nnz, timer_duration(read_mtx));
    if (need_square && m != n)
    {
        free(row_ptr);
        free(col_ptr);
        free(val_ptr);
        echo(error, "Matrix is not square, please use square matrix");
        return nullptr;
    }
    
    echo(start_status, "Converting CSR to TileMatrix");
    timer_start(csr_to_seqtile);
    TileMatrix *seqtile = CSR_to_TileMatrix(m, n, nnz, row_ptr, col_ptr, val_ptr, filepath);
    symbolRateCalculation(seqtile);

    timer_end(csr_to_seqtile);
    echo(stop_status, "");
    echo(success, "Convert CSR to TileMatrix time: %f ms, m: %d, _nnz: %d, l1nnz: %d, nnz: %d", timer_duration(csr_to_seqtile), seqtile->meta_m, seqtile->_nnz, seqtile->tile_val_map[seqtile->_nnz], seqtile->meta_nnz);

    *csr_row_ptr = row_ptr;
    *csr_col_ptr = col_ptr;
    *csr_val = val_ptr;
    return seqtile;
}