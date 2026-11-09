#include <_.h>

void output_bitset_4(TileBitmap bs)
{
    TileBitmap mask = 0xf;
    for (int i = 0; i < 4; ++i)
    {
        TileBitmap tmp = (bs & mask) >> (i * 4);
        for (int i = 0; i < 4; ++i)
            printf("%d", (tmp >> i) & 1);
        printf("\n");
        mask <<= 4;
    }
}
