#pragma once
#include "./TileMatrix.h"

inline void free_t(void*x)
{
    if (x) free(x);
}

void destroy_TileMatrix(TileMatrix* t, bool ignore_value = false, bool detail = true)
{
    free_t(t->tile_row_ptr);
    free_t(t->tile_col_ptr);
    if (detail) {
        free_t(t->tile_val_map);
        free_t(t->tile_off_map);

        free_t(t->tile_16_bitmap_ptr);
        
        free_t(t->tile_4_bitmap_ptr);
        free_t(t->tile_4_offmap_ptr);
    }
    if (!ignore_value)
        free_t(t->tile_val_ptr);

    free_t(t);
}