#include "../TileMatrix.h"
#include <msg.h>

enum TileSparseFormat
{
    COO,    // COO
    CSR,    // Compress Sparse Row
    ELL,    // Ellpack
    HYB,    // Hybrid
    DRW,    // Dense Row
    DCL,    // Dense Col
    DNS,    // Dense
    BIT     // BIT
};

TileIndex _bitmap_max_row(TileBitmap bitmap)
{
    TileIndex max_row = 0;
    TileBitmap mask = 0xf;
    while (bitmap)
    {
        TileBitmap row = TileBitmap_popcount(bitmap & mask);
        if (row > max_row)
            max_row = row;
        bitmap >>= 4;
    }
    return max_row;
}

TileIndex _bitmap_min_row(TileBitmap bitmap)
{
    TileIndex min_row = 4;
    TileBitmap mask = 0xf;
    while (bitmap)
    {
        TileBitmap row = TileBitmap_popcount(bitmap & mask);
        if (row && row < min_row)
            min_row = row;
        bitmap >>= 4;
    }
    return min_row;
}

int _size_of_coo(TileBitmap bitmap, TileIndex *nnz, TileIndex *row_num, TileIndex *col_num)
{
    *nnz = TileBitmap_popcount(bitmap);
    *row_num = (*nnz + 3) >> 2;
    *col_num = (*nnz + 3) >> 2;
    return *row_num * 8 + *col_num * 8 + *nnz * MatValue_size;
}

int _size_of_csr(TileBitmap bitmap, TileIndex *nnz, TileIndex *row_num, TileIndex *col_num)
{
    *nnz = TileBitmap_popcount(bitmap);
    *row_num = 3;
    *col_num = (*nnz + 3) >> 2;
    return *row_num * 8 + *col_num * 8 + *nnz * MatValue_size;
}

int _size_of_ell(TileBitmap bitmap, TileIndex *nnz, TileIndex *row_num, TileIndex *col_num)
{
    TileIndex max_row = 0;
    TileBitmap mask = 0xf;
    while (bitmap)
    {
        TileBitmap row = TileBitmap_popcount(bitmap & mask);
        if (row > max_row)
            max_row = row;
        bitmap >>= 4;
    }
    *nnz = max_row << 2;
    *row_num = 0;
    *col_num = max_row << 1 | 1;
    return *col_num * 8 + *nnz * MatValue_size;
}

int _size_of_hyb(TileBitmap bitmap, TileIndex *nnz, TileIndex *row_num, TileIndex *col_num)
{
    TileIndex min_row = 4, rest = 0;
    TileBitmap mask = 0xf;
    for (TileIndex i = 0; i < 16; i += 4)
    {
        ushort row = TileBitmap_popcount((bitmap >> i) & mask);
        if (row && row < min_row)
            min_row = row;
    }

    while (bitmap)
    {
        ushort row = TileBitmap_popcount(bitmap & mask);
        rest += std::max(0, row - min_row);
        bitmap >>= 4;
    }

    *nnz = (min_row << 2) + rest;
    *row_num = (rest >> 2) + 1;
    *col_num = (min_row << 1) + ((rest + 3) >> 2);
    return *row_num * 8 + *col_num * 8 + *nnz * MatValue_size;
}

int _size_of_dnsrow(TileBitmap bitmap, TileIndex *nnz, TileIndex *row_num, TileIndex *col_num)
{
    TileIndex cnt_rows = 0;
    TileBitmap mask = 0xf;

    while (bitmap)
    {
        TileBitmap row = TileBitmap_popcount(bitmap & mask);
        if (row)
            cnt_rows++;
        bitmap >>= 4;
    }

    *nnz = cnt_rows << 2;
    *row_num = 0;
    *col_num = (cnt_rows + 4) >> 2; // (cnt_rows + 1 + 3) / 4
    return *col_num * 8 + *nnz * MatValue_size;
}

int _size_of_dnscol(TileBitmap bitmap, TileIndex *nnz, TileIndex *row_num, TileIndex *col_num)
{
    TileIndex cnt_cols = 0;
    TileBitmap mask = 0x1111;

    for (TileIndex _ = 0; _ < 4; ++_)
    {
        TileBitmap col = TileBitmap_popcount(bitmap & mask);
        if (col)
            cnt_cols++;
        bitmap >>= 1;
    }

    *nnz = cnt_cols << 2;
    *row_num = 0;
    *col_num = (cnt_cols + 4) >> 2; // (cnt_cols + 1 + 3) / 4
    return *col_num * 8 + *nnz * MatValue_size;
}

int _size_of_dns(TileBitmap bitmap, TileIndex *nnz, TileIndex *row_num, TileIndex *col_num)
{
    *nnz = 16;
    *row_num = 0;
    *col_num = 0;
    return MatValue_size << 4;
}

int _size_of_bitmap(TileBitmap bitmap, TileIndex *nnz, TileIndex *row_num, TileIndex *col_num)
{
    *nnz = TileBitmap_popcount(bitmap);
    *row_num = 0;
    *col_num = 0;
    return 16 + *nnz * MatValue_size;
}

// function table
int (*size_of_format[8])(TileBitmap, TileIndex *, TileIndex *, TileIndex *) = {
    _size_of_coo,
    _size_of_csr,
    _size_of_ell,
    _size_of_hyb,
    _size_of_dnsrow,
    _size_of_dnscol,
    _size_of_dns,
    _size_of_bitmap};

#ifdef USING_BITMAP
#define _types 8
#else
#define _types 7
#endif // DEBUG

ushort packed_format_table[65536];

//                         3 bits, 4 bits, 4 bits, 4 bits
#define sparse_format_pack(format, nnz, row_num, col_num) ((format & 0xf) | (((nnz - 1) & 0xf) << 4) | ((row_num & 0xf) << 8) | ((col_num & 0xf) << 12))
#define sparse_format_unpack_fmt(info) (info & 0xf)
#define sparse_format_unpack_nnz(info) (((info >> 4) & 0xf) + 1)
#define sparse_format_unpack_row(info) ((info >> 8) & 0xf)
#define sparse_format_unpack_col(info) ((info >> 12) & 0xf)

TileSparseFormat _sparse_format_switch(TileBitmap bitMap, TileIndex *nnz, TileIndex *row_num, TileIndex *col_num)
{
    int min_size = size_of_format[0](bitMap, nnz, row_num, col_num);
    TileSparseFormat res = COO;
    TileIndex _nnz, _row_num, _col_num;
#pragma unroll
    for (TileIndex i = 1; i < _types; ++i)
    {
        const int size = size_of_format[i](bitMap, &_nnz, &_row_num, &_col_num);
        if (size < min_size)
        {
            min_size = size;
            res = (TileSparseFormat)i;
            *nnz = _nnz;
            *row_num = _row_num;
            *col_num = _col_num;
        }
    }
    return res;
}

void sparse_format_switch_initialize()
{
    if (file_exists("_fmt.bin"))
    {
        FILE *fp = fopen("_fmt.bin", "rb");
        fread(packed_format_table, sizeof(ushort), 65536, fp);
        fclose(fp);

        // int cnt[8] = {0};
        // for (TileBitmap i = 1; i > 0; ++i)
        // {
        //     TileSparseFormat fmt = sparse_format_unpack_fmt(packed_format_table[i]);
        //     cnt[fmt]++;
        //     if (fmt == DCL) printf("%d ", TileBitmap_popcount(i));
        // }
        // printf("\n");

        // for (int i = 0; i < 8; ++i)
        // {
        //     echo(debug, "format \"%s\": %d", TileSparseFormatName[i], cnt[i]);
        // }
    }
    else
    {
        for (TileBitmap i = 1; i > 0; ++i)
        {
            TileIndex p_nnz, p_row_num, p_col_num, res = _sparse_format_switch(i, &p_nnz, &p_row_num, &p_col_num);
            packed_format_table[i] = sparse_format_pack(res, p_nnz, p_row_num, p_col_num);
        }
        FILE *fp = fopen("_fmt.bin", "wb");
        fwrite(packed_format_table, sizeof(ushort), 65536, fp);
        fclose(fp);
    }
}

TileSparseFormat sparse_format_switch(TileBitmap bitMap, MatIndex *nnz, MatIndex *row_num, MatIndex *col_num)
{
    if (bitMap) {    
        ushort info = packed_format_table[bitMap];
        *nnz = sparse_format_unpack_nnz(info);
        *row_num = sparse_format_unpack_row(info);
        *col_num = sparse_format_unpack_col(info);

        return (TileSparseFormat) sparse_format_unpack_fmt(info);
    } else {
        *nnz = 0;
        *row_num = 0;
        *col_num = 0;
        return COO;
    }
}

#undef _types
#undef sparse_format_pack
#undef sparse_format_unpack_fmt
#undef sparse_format_unpack_nnz
#undef sparse_format_unpack_row_num
#undef sparse_format_unpack_col_num