#pragma once
#include <tmatrix/Utils/timer.h>
#include <tmatrix/MMIO/mmio_highlevel.h>
#include <tmatrix/DataStructure/Factory.h>
#include <tmatrix/DataStructure/transform.h>
#include <tmatrix/DataStructure/Transform/Tile_7_fmt.h>

void symbolRateCalculation(TileMatrix*m, size_t*ts, size_t*cs, size_t*b16s, size_t*b4s, size_t*t7s)
{
    size_t csr_sym_size = (m->meta_m + 1) * sizeof(MatIndex) + (m->meta_nnz) * sizeof(MatIndex);
    size_t val_size = (m->meta_nnz) * sizeof(MatValue);

    size_t tile_sym_size = (m->_m + 3 * m->_nnz + 3) * sizeof(MatIndex) + (m->_nnz + m->tile_val_map[m->_nnz]) * sizeof(TileBitmap) + (m->tile_val_map[m->_nnz]) * sizeof(TileIndex);

    echo(markdown, "| | CSR | Tile | Compress Rate |\\n| --- | :---: | :---: | :---: |\\n| Symbol Using Size | %.3f MB | %.3f MB | %.2f%% |\\n| Memory Transfer Size | %.3f MB | %.3f MB | %.2f%% |",
         csr_sym_size / 1024.0 / 1024.0, tile_sym_size / 1024.0 / 1024.0, tile_sym_size * 100.0 / csr_sym_size,
         (csr_sym_size + val_size) / 1024.0 / 1024.0, (tile_sym_size + val_size) / 1024.0 / 1024.0, (tile_sym_size + val_size) * 100.0 / (csr_sym_size + val_size));
    echo(debug, "%.2lf%%", (m->tile_val_map[m->_nnz]) * sizeof(TileIndex) * 1.0 / tile_sym_size);    
    *ts = tile_sym_size;
    *cs = csr_sym_size;
    *b16s = m->_m * sizeof(MatIndex) + m->_nnz * sizeof(MatIndex) + (m->_nnz * 256ULL - m->meta_nnz) * sizeof(MatValue);

    if (b4s != nullptr)
    {
        size_t sum = 0;
        MatIndex l1nnz = m->tile_val_map[m->_nnz];

        echo(info, "%llu", m->tile_off_map[m->_nnz]);

        sum += (l1nnz * 16ULL - m->meta_nnz) * sizeof(MatValue);
        echo(info, "%llu %llu %llu %llu", l1nnz * 16ULL, m->meta_nnz, l1nnz * 16ULL - m->meta_nnz, sum);
        sum += (m->_m * 4 + l1nnz) * sizeof(MatIndex);
        *b4s = sum;
    }
    if (t7s != nullptr)
    {
        size_t sum = 0;
        MatIndex nnz, row, col;
        MatIndex l1nnz = m->tile_val_map[m->_nnz];

        for (MatIndex i = 0; i < m->_nnz; ++i) {
            TileBitmap bm = m->tile_16_bitmap_ptr[i];
            sparse_format_switch(bm, &nnz, &row, &col);
            sum += row + col;
        }
        for (MatIndex i = 0; i < l1nnz; ++i)
        {
            TileBitmap bm = m->tile_4_bitmap_ptr[i];
            sparse_format_switch(bm, &nnz, &row, &col);
            sum += (nnz - TileBitmap_popcount(bm)) * sizeof(MatValue) + (row + col);
        }
        sum += (m->_m + m->_nnz + 3 * l1nnz) * sizeof(MatIndex) + l1nnz;
        *t7s = sum;
    }
}

int test_data_format(const char*filepath)
{
    int isSymmetric;
    MatIndex m, n, nnz;
    MatIndex *row_ptr, *col_ptr;
    MatValue *val_ptr;
    Timer read_mtx, csr_to_seqtile;
    bool exists;

    echo(title, "test Data Format on %s", filepath);
    sparse_format_switch_initialize();

    char tile_path[128];
    sprintf(tile_path, "%s.tile", filepath);
    exists = file_exists(tile_path);
    if (!exists) {
        echo(start_status, "Reading Matrix file: [bold green]%s[/]", filepath);
        timer_start(read_mtx);
        int status = mmio_allinone(filepath, &m, &n, &nnz, &isSymmetric, &row_ptr, &col_ptr, &val_ptr);
        timer_end(read_mtx);
        echo(stop_status, "");
        if (status != 0) {
            echo(error, "Failed to read matrix file (%d): %s", status, filepath);
            return -1;
        }
        echo(success, "m = %d, n = %d, nnz = %d; time: %f ms", m, n, nnz, timer_duration(read_mtx));
    }
    echo(start_status, "CSR to TileMatrix");

    timer_start(csr_to_seqtile);
    TileMatrix *t = CSR_to_TileMatrix(m, n, nnz, row_ptr, col_ptr, val_ptr, filepath, false);
    timer_end(csr_to_seqtile);
    echo(stop_status, "");
    echo(success, "CSR to TileMatrix time: %f ms", timer_duration(csr_to_seqtile));

    if (!exists) {
        free(row_ptr);
        free(col_ptr);
        free(val_ptr);
    }

    size_t tileSize, CSRSize, BSR16Size, BSR4Size, Tile7Size;
    symbolRateCalculation(t, &tileSize, &CSRSize, &BSR16Size, &BSR4Size, &Tile7Size);
    destroy_TileMatrix(t, true);
    printf("%llu,%llu,%llu,%llu,%llu\n", tileSize, CSRSize, BSR16Size, BSR4Size, Tile7Size);
    fflush(stdout);
    return 0;
}