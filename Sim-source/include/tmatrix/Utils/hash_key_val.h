#pragma once
#include <msg.h>
#include <_.h>
#include <string.h>

typedef struct
{
    MatIndex *key;
    MatValue *val;
} hash_key_val_t;

typedef struct
{
    TileBitmap val16;
    TileBitmap val4[16];
} hash_key_bitmap_val_t;

typedef struct
{
    MatIndex *key;
    hash_key_bitmap_val_t *val;
} hash_key_bitmap_t;

unsigned int nextPow2(unsigned int size)
{
    size--;
    size |= size >> 1;
    size |= size >> 2;
    size |= size >> 4;
    size |= size >> 8;
    size |= size >> 16;
    size++;
    return size;
}

hash_key_val_t *create_key_val_hash(unsigned int *expect_size)
{
    *expect_size = nextPow2(*expect_size);
    hash_key_val_t *hash = (hash_key_val_t *)malloc(sizeof(hash_key_val_t));
    hash->key = (MatIndex *)malloc(sizeof(MatIndex) * (*expect_size));
    hash->val = (MatValue *)malloc(sizeof(MatValue) * (*expect_size));

    memset(hash->key, -1, sizeof(MatIndex) * (*expect_size));
    return hash;
}

hash_key_bitmap_t *create_key_bitmap_hash(unsigned int expect_size)
{
    hash_key_bitmap_t *hash = (hash_key_bitmap_t *)malloc(sizeof(hash_key_bitmap_t));
    hash->key = (MatIndex *)malloc(sizeof(MatIndex) * (expect_size));
    hash->val = (hash_key_bitmap_val_t *)malloc(sizeof(hash_key_bitmap_val_t) * (expect_size));
    memset(hash->key, -1, sizeof(MatIndex) * (expect_size));
    return hash;
}

u_int32_t gather_and_resize_hash(hash_key_bitmap_t **table, unsigned int expect_size, unsigned int *nz4)
{
    u_int32_t idx = 0;
    for (u_int32_t i = 0; i < expect_size; i++)
    {
        if ((*table)->key[i] != -1)
        {
            *nz4 += TileBitmap_popcount((*table)->val[i].val16);
            if (idx != i)
            {
                (*table)->key[idx] = (*table)->key[i];
                (*table)->val[idx] = (*table)->val[i];
                (*table)->key[i] = -1;
            }
            ++idx;
        }
    }
    *table = (hash_key_bitmap_t *)realloc(*table, sizeof(hash_key_bitmap_t) + sizeof(MatIndex) * (idx) + sizeof(hash_key_bitmap_val_t) * (idx));
    return idx;
}

void insert_key_val_hash(hash_key_val_t *hash, unsigned int size, MatIndex key, MatValue val)
{
    const unsigned int _mask = size - 1;
    unsigned int index = key & _mask;
    while (hash->key[index] != -1 && hash->key[index] != key)
    {
        index = (index + 1) & _mask;
    }
    if (hash->key[index] == key)
        hash->val[index] += val;
    else
    {
        hash->key[index] = key;
        hash->val[index] = val;
    }
}

void _insert_key_bitmap_hash(MatIndex *keys, hash_key_bitmap_val_t *vals, unsigned int size, MatIndex key, hash_key_bitmap_val_t val)
{
    // const unsigned int _mask = size - 1;
    // unsigned int index = key & _mask;
    unsigned int index = key;
    // while (keys[index] != -1 && keys[index] != key)
    // {
    //     index = (index + 1) & _mask;
    // }
    // if (keys[index] != -1 && keys[index] != key) echo(error, "hash error");
    if (keys[index] == key){
        vals[index].val16 |= val.val16;
        #pragma unroll
        for (TileIndex i = 0; i < 4; i++)
        {
            *((unsigned long long*) (vals[index].val4) + i) |= *((unsigned long long*) (val.val4) + i);
        }
    }
    else
    {
        keys[index] = key;
        vals[index] = val;
    }
}

void insert_key_bitmap_hash(hash_key_bitmap_t *hash, unsigned int size, MatIndex key, hash_key_bitmap_val_t val)
{
    _insert_key_bitmap_hash(hash->key, hash->val, size, key, val);
}

void iter_key_val_hash(hash_key_val_t *hash, unsigned int size, MatIndex *col_ptr, MatValue *val_ptr)
{
    for (unsigned int i = 0; i < size; i++)
    {
        if (hash->key[i] != -1)
        {
            *col_ptr++ = hash->key[i];
            *val_ptr++ = hash->val[i];
        }
    }
}

void iter_key_bitmap_hash(
    // input
    hash_key_bitmap_t *hash, unsigned int size,
    // output
    MatIndex *col_ptr, hash_key_bitmap_val_t *val_ptr)
{
    for (unsigned int i = 0; i < size; i++)
    {
        if (hash->key[i] != -1 && hash->val[i].val16)
        {
            *col_ptr++ = hash->key[i];
            *val_ptr++ = hash->val[i];
        }
    }
}

void iter_gathered_key_bitmap_hash(
    // input
    hash_key_bitmap_t *hash, unsigned int size,
    // output
    MatIndex *col_ptr, hash_key_bitmap_val_t *val_ptr)
{
    for (unsigned int i = 0; i < size; i++)
    {
        *col_ptr++ = hash->key[i];
        *val_ptr++ = hash->val[i];
    }
}

int count_key_bitmap_hash(hash_key_bitmap_t *hash, unsigned int size, unsigned int *nz4)
{
    int count = 0;
    for (unsigned int i = 0; i < size; i++)
    {
        if (hash->key[i] != -1 && hash->val[i].val16)
        {
            count++;
            *nz4 += TileBitmap_popcount(hash->val[i].val16);
        }
    }
    return count;
}

u_int32_t gather_hash(hash_key_bitmap_t *table, unsigned int expect_size)
{
    u_int32_t idx = 0;
    for (u_int32_t i = 0; i < expect_size; i++)
    {
        if (table->key[i] != -1)
        {
            // *nz4 += TileBitmap_popcount((*table)->val[i].val16);
            if (idx != i)
            {
                table->key[idx] = table->key[i];
                table->val[idx] = table->val[i];
            }
            ++idx;
        }
    }
    return idx;
}

#define free_hash(hash) \
    free(hash->key);    \
    free(hash->val);    \
    free(hash);
