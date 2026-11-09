#pragma once
#include <_.h>
#include <glib.h>

GHashTable* create_hash_table()
{
    GHashTable* hash_table = g_hash_table_new(g_int_hash, g_int_equal);
    return hash_table;
}

void destroy_hash_table(GHashTable* hash_table)
{
    g_hash_table_destroy(hash_table);
}

gboolean insert_hash_table(GHashTable* hash_table, MatIndex col)
{
    MatIndex* p = g_new(MatIndex, 1);
    *p = col;
    return g_hash_table_insert(hash_table, p, NULL);
}

guint count_hash_table(GHashTable* hash_table)
{
    return g_hash_table_size(hash_table);
}

gboolean acumulate_key_val_table(GHashTable* hash_table, MatIndex col, MatValue val)
{
    MatIndex* p = g_new(MatIndex, 1);
    *p = col;

    // find the key
    MatValue* old_val = (MatValue*)g_hash_table_lookup(hash_table, p);
    if (old_val)
    {
        *old_val += val;
        g_free(p);
        return TRUE;
    }
    else {
        MatValue* q = g_new(MatValue, 1);
        *q = val;
        return g_hash_table_insert(hash_table, p, q);
    }
}

void iter_acc_hash_table(GHashTable* hash_table, MatIndex*col_ptr, MatValue*val_ptr)
{
    MatIndex _index = 0;
    GHashTableIter iter;
    gpointer key, value;
    g_hash_table_iter_init(&iter, hash_table);
    while (g_hash_table_iter_next(&iter, &key, &value))
    {
        col_ptr[_index] = *(MatIndex*)key;
        val_ptr[_index] = *(MatValue*)value;
        ++_index;        
    }
}