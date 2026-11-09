#include <_.h>
#include <tmatrix/Calculation/bitmap.h>

void bit_x_bit_255(TileBitmap A_bitmap, MatValue *A_value, TileBitmap B_bitmap, MatValue *B_value, TileBitmap C_bitmap, MatValue *C_value)
{
    TileBitmap idx[4] = {0x0000, 0x0001, 0x0011, 0x0111};
    TileIndex cidx[4] = {0b0000, 0b0001, 0b0011, 0b0111};
    TileIndex nnzA = TileBitmap_popcount(A_bitmap);

    for (TileIndex Ai = 0; Ai < nnzA; ++Ai)
    {
        TileBitmap _ak = TileBitmap_lowbit(A_bitmap);

        TileIndex ctz = TileBitmap_ctz(_ak), row = ctz & 0xc, Acol = ctz & 3, \
                  Bj  = TileBitmap_popcount(B_bitmap & (0xf * idx[Acol])), \
                  Cj  = TileBitmap_popcount(C_bitmap & (0xf * idx[row >> 2]));

        TileBitmap B_row_map = (B_bitmap & (0xf << (Acol << 2))) >> (Acol << 2);
        TileBitmap C_row_map = (C_bitmap & (0xf << row)) >> row;

        while (B_row_map)
        {
            TileBitmap _bk = TileBitmap_lowbit(B_row_map);
            TileIndex _cidx = TileBitmap_popcount(C_row_map & cidx[TileBitmap_ctz(_bk)]);
            C_value[Cj + _cidx] += A_value[Ai] * B_value[Bj++];
            B_row_map ^= _bk;
        }
        A_bitmap ^= _ak;
    }
}