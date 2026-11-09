#pragma once

#include <_.h>
#include <msg.h>

typedef struct 
{
    MatIndex meta_m, meta_n, meta_nnz;
    MatIndex _m, _n, _nnz;

    MatIndex *tile_row_ptr;                 // row 0
    MatIndex *tile_col_ptr;                 // col 0
    MatIndex *tile_val_map;                 // val 0
    MatIndex *tile_off_map;                 // val 0
    TileBitmap *tile_16_bitmap_ptr;   // val 0
    
    // meta data of TileLevel 4
    TileBitmap *tile_4_bitmap_ptr;    // val 1
    TileIndex  *tile_4_offmap_ptr;    // val 1
    MatValue   *tile_val_ptr;               // val 2
} TileMatrix;


void symbolRateCalculation(TileMatrix*m)
{
    size_t csr_sym_size = (m->meta_m + 1) * sizeof(MatIndex) + (m->meta_nnz) * sizeof(MatIndex);
    size_t val_size = (m->meta_nnz) * sizeof(MatValue);

    size_t tile_sym_size = (m->_m + 3 * m->_nnz + 3) * sizeof(MatIndex) + (m->_nnz + m->tile_val_map[m->_nnz]) * sizeof(TileBitmap) + (m->tile_val_map[m->_nnz]) * sizeof(TileIndex);

    echo(markdown, "| | CSR | Tile | Compress Rate |\\n| --- | :---: | :---: | :---: |\\n| Symbol Using Size | %.3f MB | %.3f MB | %.2f%% |\\n| Memory Transfer Size | %.3f MB | %.3f MB | %.2f%% |",
         csr_sym_size / 1024.0 / 1024.0, tile_sym_size / 1024.0 / 1024.0, tile_sym_size * 100.0 / csr_sym_size,
         (csr_sym_size + val_size) / 1024.0 / 1024.0, (tile_sym_size + val_size) / 1024.0 / 1024.0, (tile_sym_size + val_size) * 100.0 / (csr_sym_size + val_size));
}