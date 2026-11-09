#pragma once
#include <_.h>
#include <string.h>
#include <stdlib.h>

struct sim_hash
{
    MatIndex *key;
    TileBitmap *val;
};

// unsigned int nextPow2(unsigned int size)
// {
//     size--;
//     size |= size >> 1;
//     size |= size >> 2;
//     size |= size >> 4;
//     size |= size >> 8;
//     size |= size >> 16;
//     size++;
//     return size;
// }

sim_hash *_create_key_bitmap_hash(unsigned int expect_size)
{
    sim_hash *hash = (sim_hash *)malloc(sizeof(sim_hash));
    hash->key = (MatIndex *)malloc(sizeof(MatIndex) * (expect_size));
    hash->val = (TileBitmap *)malloc(sizeof(TileBitmap) * (expect_size));
    memset(hash->key, -1, sizeof(MatIndex) * (expect_size));
    return hash;
}

u_int32_t gather_and_resize_hash(sim_hash *table, unsigned int expect_size, unsigned int *nz4)
{
    u_int32_t idx = 0;
    for (u_int32_t i = 0; i < expect_size; i++)
    {
        if (table->key[i] != -1)
        {
            *nz4 += TileBitmap_popcount(table->val[i]);
            if (idx != i)
            {
                table->key[idx] = table->key[i];
                table->val[idx] = table->val[i];
                table->key[i] = -1;
            }
            ++idx;
        }
    }
    // *table = (sim_hash *)realloc(*table, sizeof(sim_hash) + sizeof(MatIndex) * (idx) + sizeof(TileBitmap) * (idx));
    table->key = (MatIndex *)realloc(table->key, sizeof(MatIndex) * (idx));
    table->val = (TileBitmap *)realloc(table->val, sizeof(TileBitmap) * (idx));
    return idx;
}

void _insert_key_bitmap_hash(MatIndex *keys, TileBitmap *vals, unsigned int size, MatIndex key, TileBitmap val)
{
    const unsigned int _mask = size - 1;
    unsigned int index = key & _mask;
    while (keys[index] != -1 && keys[index] != key)
    {
        index = (index + 1) & _mask;
    }
    if (keys[index] == key)
    {
        vals[index] |= val;
    }
    else
    {
        keys[index] = key;
        vals[index] = val;
    }
}

void insert_key_bitmap_hash(sim_hash *hash, unsigned int size, MatIndex key, TileBitmap val)
{
    _insert_key_bitmap_hash(hash->key, hash->val, size, key, val);
}

void iter_gathered_key_bitmap_hash(
    // input
    sim_hash *hash, unsigned int size,
    // output
    MatIndex *col_ptr)
{
    for (unsigned int i = 0; i < size; i++)
    {
        *col_ptr++ = hash->key[i];
    }
}

#define free_hash(hash) \
    free(hash->key);    \
    free(hash->val);    \
    free(hash);
