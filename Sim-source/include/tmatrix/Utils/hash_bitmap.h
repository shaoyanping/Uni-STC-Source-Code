#pragma once
#include <_.h>

typedef unsigned long long BHash;

BHash*create_table(unsigned int*size)
{
    *size = (*size + 63) >> 6;
    BHash*table = (BHash*)calloc(*size, sizeof(BHash));
    return table;
}

#define insert_table(table, id) table[(id) >> 6] |= 1ULL << ((id) & 63)

unsigned int count_table(BHash*table, unsigned int size)
{
    unsigned int count = 0;
    for (unsigned int i = 0; i < size; i++)
    {
        if (table[i]) count += __builtin_popcountll(table[i]);
    }
    return count;
}

#define free_table(table) free(table)