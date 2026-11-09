#include "../TileMatrix.h"


void _TileLevel_4_bitmap_to_mtx(FILE *fp, TileBitmap bitmap, MatValue *val_ptr, const MatIndex row_offset, const MatIndex col_offset)
{
    while (bitmap)
    {
        TileBitmap k = TileBitmap_lowbit(bitmap);
        TileIndex ctz = TileBitmap_ctz(k);

        fprintf(fp, "%d %d %f\n", (ctz >> 2) + row_offset + 1, (ctz & 3) + col_offset + 1, *val_ptr++);

        bitmap ^= k;
    }
}

void TileMatrix_to_MtxFile(TileMatrix *tm, const char *dst_path)
{
    MatIndex m = tm->meta_m, n = tm->meta_n, nnz = tm->meta_nnz;
    MatIndex _m = tm->_m;

    FILE *fp = fopen(dst_path, "w");
    fprintf(fp, "%%%%MatrixMarket matrix coordinate real general\n");
    fprintf(fp, "%d %d %d\n", m, n, nnz);

    MatIndex *tile_row_ptr = tm->tile_row_ptr, *tile_col_ptr = tm->tile_col_ptr;

    for (MatIndex i = 0; i < _m; ++i)
    {
        MatIndex tile_row_start = tile_row_ptr[i];
        MatIndex tile_row_end = tile_row_ptr[i + 1];

        for (MatIndex j = tile_row_start; j < tile_row_end; ++j)
        {
            MatIndex _offset = tm->tile_val_map[j];
            MatIndex cur_j = tile_col_ptr[j];
            TileBitmap tile_16_bitmap = tm->tile_16_bitmap_ptr[j];
            MatValue *cur_val_ptr = tm->tile_val_ptr + tm->tile_off_map[j];

            while (tile_16_bitmap)
            {
                TileBitmap _k = TileBitmap_lowbit(tile_16_bitmap);
                TileIndex ctz = TileBitmap_ctz(_k);
                _TileLevel_4_bitmap_to_mtx(
                    fp,                                                // FILE*
                    tm->tile_4_bitmap_ptr[_offset],       // TileBitmap bitmap
                    cur_val_ptr + tm->tile_4_offmap_ptr[_offset], // MatValue* val_ptr
                    i << 4 | ctz & 0xc, cur_j << 4 | ((ctz & 3) << 2)
                );
                ++_offset;
                tile_16_bitmap ^= _k;
            }
        }
    }

    fflush(fp);
    fclose(fp);
}