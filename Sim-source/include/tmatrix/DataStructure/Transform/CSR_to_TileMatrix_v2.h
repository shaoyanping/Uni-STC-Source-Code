#pragma once

#include "../TileMatrix.h"
#include "tmatrix/Utils/omp_utils.h"
#include "tmatrix/Utils/hash_bitmap.h"
#include "tmatrix/Calculation/bitmap.h"
#include <msg.h>

#include <map>
#include <unordered_map>

void write_tile_matrix(const char*filepath, TileMatrix*m)
{
    FILE *fp = fopen(filepath, "wb");
    fwrite(&m->meta_m, sizeof(MatIndex), 1, fp);
    fwrite(&m->meta_n, sizeof(MatIndex), 1, fp);
    fwrite(&m->meta_nnz, sizeof(MatIndex), 1, fp);
    fwrite(&m->_m, sizeof(MatIndex), 1, fp);
    fwrite(&m->_n, sizeof(MatIndex), 1, fp);
    fwrite(&m->_nnz, sizeof(MatIndex), 1, fp);
    fwrite(m->tile_row_ptr, sizeof(MatIndex), m->_m + 1, fp);
    fwrite(m->tile_col_ptr, sizeof(MatIndex), m->_nnz, fp);
    fwrite(m->tile_val_map, sizeof(MatIndex), m->_nnz + 1, fp);
    fwrite(m->tile_off_map, sizeof(MatIndex), m->_nnz + 1, fp);
    fwrite(m->tile_16_bitmap_ptr, sizeof(TileBitmap), m->_nnz, fp);
    fwrite(m->tile_4_bitmap_ptr, sizeof(TileBitmap), m->tile_val_map[m->_nnz], fp);
    fwrite(m->tile_4_offmap_ptr, sizeof(TileIndex), m->tile_val_map[m->_nnz], fp);
    fwrite(m->tile_val_ptr, sizeof(MatValue), m->meta_nnz, fp);
    fclose(fp);
}

__attribute__((malloc))
TileMatrix *load_tile_matrix(const char*filepath, bool need_square=false)
{
    TileMatrix*res = (TileMatrix*)malloc(sizeof(TileMatrix));
    FILE *fp = fopen(filepath, "rb");
    fread(&res->meta_m, sizeof(MatIndex), 1, fp);
    fread(&res->meta_n, sizeof(MatIndex), 1, fp);
    if (need_square && res->meta_m != res->meta_n) {
        fclose(fp);
        free(res);
        return nullptr;
    }
    fread(&res->meta_nnz, sizeof(MatIndex), 1, fp);
    fread(&res->_m, sizeof(MatIndex), 1, fp);
    fread(&res->_n, sizeof(MatIndex), 1, fp);
    fread(&res->_nnz, sizeof(MatIndex), 1, fp);
    res->tile_row_ptr = (MatIndex *)malloc((res->_m + 1) * sizeof(MatIndex));
    res->tile_col_ptr = (MatIndex *)malloc(res->_nnz * sizeof(MatIndex));
    res->tile_val_map = (MatIndex *)malloc((res->_nnz + 1) * sizeof(MatIndex));
    res->tile_off_map = (MatIndex *)malloc((res->_nnz + 1) * sizeof(MatIndex));
    res->tile_16_bitmap_ptr = (TileBitmap *)malloc(res->_nnz * sizeof(TileBitmap));
    fread(res->tile_row_ptr, sizeof(MatIndex), res->_m + 1, fp);
    fread(res->tile_col_ptr, sizeof(MatIndex), res->_nnz, fp);
    fread(res->tile_val_map, sizeof(MatIndex), res->_nnz + 1, fp);
    fread(res->tile_off_map, sizeof(MatIndex), res->_nnz + 1, fp);
    fread(res->tile_16_bitmap_ptr, sizeof(TileBitmap), res->_nnz, fp);
    res->tile_4_bitmap_ptr = (TileBitmap *)malloc(res->tile_val_map[res->_nnz] * sizeof(TileBitmap));
    res->tile_4_offmap_ptr = (TileIndex *)malloc(res->tile_val_map[res->_nnz] * sizeof(TileIndex));
    res->tile_val_ptr = (MatValue *)malloc(res->meta_nnz * sizeof(MatValue));
    fread(res->tile_4_bitmap_ptr, sizeof(TileBitmap), res->tile_val_map[res->_nnz], fp);
    fread(res->tile_4_offmap_ptr, sizeof(TileIndex), res->tile_val_map[res->_nnz], fp);
    fread(res->tile_val_ptr, sizeof(MatValue), res->meta_nnz, fp);
    fclose(fp);
    return res;
}


__attribute__((malloc))
TileMatrix *
CSR_to_TileMatrix(const int m, const int n, const int nnz, const MatIndex *csr_row_ptr, const MatIndex *csr_col_ptr, const MatValue *val, const char*filepath, bool lazy_load = true, bool structure_only=false, bool need_square=false)
{
    char new_file_path[256] = {0};
    sprintf(new_file_path, "%s.tile", filepath);

    if (lazy_load && file_exists(new_file_path)) {
        return load_tile_matrix(new_file_path, need_square);
    }

    MatIndex level_1_nnz = 0;
    TileMatrix *seq_tile_matrix = (TileMatrix *)malloc(sizeof(TileMatrix));
    seq_tile_matrix->meta_m = m;
    seq_tile_matrix->meta_n = n;
    seq_tile_matrix->meta_nnz = nnz;

    seq_tile_matrix->_m = (m + 15) >> 4;
    seq_tile_matrix->_n = (n + 15) >> 4;

    seq_tile_matrix->tile_row_ptr = (MatIndex *)malloc((seq_tile_matrix->_m + 1) * sizeof(MatIndex));
    seq_tile_matrix->tile_val_ptr = (MatValue *)malloc(seq_tile_matrix->meta_nnz * sizeof(MatValue));
    seq_tile_matrix->tile_row_ptr[0] = 0;

    std::map<MatIndex, std::map<TileIndex, TileBitmap>>* tile_maps = new std::map<MatIndex, std::map<TileIndex, TileBitmap>>[seq_tile_matrix->_m];
    std::unordered_map<MatIndex, std::unordered_map<MatIndex, MatIndex>> tile_16_colidx_map;

    // echo(debug, "Init Done");

    #pragma omp parallel for
    for (MatIndex i = 0; i < m; i += 16)
    {
        MatIndex start_i = i, end_i = std::min(i + 16, (MatIndex)m);
        auto& tile_map = tile_maps[i >> 4];
        
        for (MatIndex ii = start_i; ii < end_i; ++ii)
        {
            for (MatIndex jj = csr_row_ptr[ii]; jj < csr_row_ptr[ii + 1]; ++jj)
            {
                MatIndex _col = csr_col_ptr[jj] >> 4;
                TileIndex _idx = ((ii & 15) & 0xc) | ((csr_col_ptr[jj] & 15) >> 2), _4idx = ((ii & 3) << 2) | (csr_col_ptr[jj] & 3);
                tile_map[_col][_idx] |= (1 << _4idx);
            }
        }

        seq_tile_matrix->tile_row_ptr[(i >> 4) + 1] = tile_map.size();
    }

    // echo(debug, "Hash Structure Done");

    omp_inclusive_scan(seq_tile_matrix->tile_row_ptr + 1, seq_tile_matrix->_m);
    seq_tile_matrix->_nnz = seq_tile_matrix->tile_row_ptr[seq_tile_matrix->_m];

    seq_tile_matrix->tile_col_ptr = (MatIndex *)malloc(seq_tile_matrix->_nnz * sizeof(MatIndex));
    seq_tile_matrix->tile_val_map = (MatIndex *)malloc((seq_tile_matrix->_nnz + 1) * sizeof(MatIndex));
    seq_tile_matrix->tile_off_map = (MatIndex *)malloc((seq_tile_matrix->_nnz + 1) * sizeof(MatIndex));
    seq_tile_matrix->tile_16_bitmap_ptr = (TileBitmap *)malloc(seq_tile_matrix->_nnz * sizeof(TileBitmap));

    // echo(debug, "Top Level Malloc Done");

    seq_tile_matrix->tile_val_map[0] = seq_tile_matrix->tile_off_map[0] = 0;

    // #pragma omp parallel for
    for (MatIndex i = 0; i < seq_tile_matrix->_m; ++i)
    {
        auto& tile_map = tile_maps[i];
        MatIndex idx = seq_tile_matrix->tile_row_ptr[i];
        MatIndex expect_nnz = csr_row_ptr[std::min((i + 1) << 4, (MatIndex)m)] - csr_row_ptr[i << 4], actual_nnz = 0;
        for (auto& ct : tile_map)
        {
            seq_tile_matrix->tile_col_ptr[idx] = ct.first;
            seq_tile_matrix->tile_val_map[idx + 1] = ct.second.size();

            TileBitmap cur_16 = 0, total_16 = 0;
            for (auto& ib : ct.second)
            {
                cur_16 |= 1 << ib.first;
                total_16 += TileBitmap_popcount(ib.second);
            }

            actual_nnz += total_16;
            seq_tile_matrix->tile_off_map[idx + 1] = total_16;
            seq_tile_matrix->tile_16_bitmap_ptr[idx] = cur_16;
            tile_16_colidx_map[i][ct.first] = idx++;
        }
        // if (expect_nnz != actual_nnz) echo(error, "[%d] Expect NNZ: %u, Actual NNZ: %u", i, expect_nnz, actual_nnz);
    }

    omp_inclusive_scan(seq_tile_matrix->tile_val_map + 1, seq_tile_matrix->_nnz);
    omp_inclusive_scan(seq_tile_matrix->tile_off_map + 1, seq_tile_matrix->_nnz);

    level_1_nnz = seq_tile_matrix->tile_val_map[seq_tile_matrix->_nnz];

    seq_tile_matrix->tile_4_bitmap_ptr = (TileBitmap *)malloc(level_1_nnz * sizeof(TileBitmap));
    seq_tile_matrix->tile_4_offmap_ptr = (TileIndex *)malloc(level_1_nnz * sizeof(TileIndex));
    // echo(debug, "Middle Level Malloc Done");

    #pragma omp parallel for
    for (MatIndex i = 0; i < seq_tile_matrix->_m; ++i)
    {
        auto& tile_map = tile_maps[i];
        MatIndex idx = seq_tile_matrix->tile_row_ptr[i];
        for (auto& ct : tile_map)
        {
            TileIndex j = 0, nnz = TileBitmap_popcount(seq_tile_matrix->tile_16_bitmap_ptr[idx]);
            MatIndex base = seq_tile_matrix->tile_val_map[idx++];
            seq_tile_matrix->tile_4_offmap_ptr[base] = 0;
            for (auto& ib : ct.second)
            {
                seq_tile_matrix->tile_4_bitmap_ptr[base + j] = ib.second;
                if (j + 1 < nnz) seq_tile_matrix->tile_4_offmap_ptr[base + j + 1] = TileBitmap_popcount(ib.second) + seq_tile_matrix->tile_4_offmap_ptr[base + j];
                ++j;
            }
        }
    }
    delete[] tile_maps;
    // echo(debug, "Middle Level Done");
    if (!structure_only) {
        #pragma omp parallel for
        for (MatIndex i = 0; i < m; i += 16)
        {
            MatIndex start_i = i, end_i = std::min(i + 16, (MatIndex)m), row = i >> 4;

            for (MatIndex ii = start_i; ii < end_i; ++ii)
            {
                for (MatIndex jj = csr_row_ptr[ii]; jj < csr_row_ptr[ii + 1]; ++jj)
                {
                    MatIndex _col = csr_col_ptr[jj] >> 4, base = tile_16_colidx_map[row][_col];
                    TileIndex _idx = ((ii & 15) & 0xc) | ((csr_col_ptr[jj] & 15) >> 2), _4idx = ((ii & 3) << 2) | (csr_col_ptr[jj] & 3);
                    MatIndex off = seq_tile_matrix->tile_val_map[base] + TileBitmap_popcount(((1 << _idx) - 1) & seq_tile_matrix->tile_16_bitmap_ptr[base]);
                    TileIndex off_4 = seq_tile_matrix->tile_4_offmap_ptr[off] + TileBitmap_popcount(((1 << _4idx) - 1) & seq_tile_matrix->tile_4_bitmap_ptr[off]);
                    seq_tile_matrix->tile_val_ptr[seq_tile_matrix->tile_off_map[base] + off_4] = val[jj];
                }
            }
        }
    }
    // echo(debug, "Data Done");
    if (lazy_load) write_tile_matrix(new_file_path, seq_tile_matrix);
    return seq_tile_matrix;
}
