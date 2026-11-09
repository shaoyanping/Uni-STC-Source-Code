#pragma once

#include <_.h>
#include "./hash_key_val.h"

void omp_inclusive_scan_TileLevel_IndexType(TileIndex *input, const int length)
{
    if (length == 0 || length == 1)
        return;

    TileIndex partial_sum=0;
    #pragma omp simd reduction(inscan, +:partial_sum)
    for (int i = 0; i < length; i++)
    {
        partial_sum += input[i];
        #pragma omp scan inclusive(partial_sum)
        input[i] = partial_sum;
    }
}

template <typename T>
void omp_inclusive_scan(T *input, const int length)
{
    if (length == 0 || length == 1)
        return;

    T partial_sum=0;
    #pragma omp simd reduction(inscan, +:partial_sum)
    for (int i = 0; i < length; i++)
    {
        partial_sum += input[i];
        #pragma omp scan inclusive(partial_sum)
        input[i] = partial_sum;
    }
}

template <typename T>
T omp_reduce(T*input, const int length)
{
    T sum = 0;
    #pragma omp simd reduction(+:sum)
    for (int i = 0; i < length; i++)
    {
        sum += input[i];
    }
    return sum;
}

template <typename T>
void inclusive_scan(T *input, const int length)
{
    if (length == 0 || length == 1)
        return;

    for (int i = 1; i < length; i++)
    {
        input[i] += input[i - 1];
    }
}

void omp_quicksort_key_val(MatIndex *key, MatValue *val, int left, int right)
{
    if (left >= right)
        return;
    int i = left, j = right;
    int pivot = key[left];
    double pivot_val = val[left];
    while (i < j)
    {
        while (i < j && key[j] >= pivot)
            j--;
        key[i] = key[j];
        val[i] = val[j];
        while (i < j && key[i] <= pivot)
            i++;
        key[j] = key[i];
        val[j] = val[i];
    }
    key[i] = pivot;
    val[i] = pivot_val;
    #pragma omp task
    omp_quicksort_key_val(key, val, left, i - 1);
    #pragma omp task
    omp_quicksort_key_val(key, val, i + 1, right);
}

template <typename IT, typename VT>
void quicksort_key_val(IT *key, VT *val, int left, int right)
{
    if (left >= right)
        return;
    int i = left, j = right;
    IT pivot = key[left];
    VT pivot_val = val[left];
    while (i < j)
    {
        while (i < j && key[j] >= pivot)
            j--;
        key[i] = key[j];
        val[i] = val[j];
        while (i < j && key[i] <= pivot)
            i++;
        key[j] = key[i];
        val[j] = val[i];
    }
    key[i] = pivot;
    val[i] = pivot_val;
    quicksort_key_val(key, val, left, i - 1);
    quicksort_key_val(key, val, i + 1, right);
}

template <typename VT>
void quicksort_key_bitmap(MatIndex *key, VT *val, int left, int right)
{
    if (left >= right)
        return;
    int i = left, j = right;
    int pivot = key[left];
    VT pivot_val = val[left];
    while (i < j)
    {
        while (i < j && key[j] >= pivot)
            j--;
        key[i] = key[j];
        val[i] = val[j];
        while (i < j && key[i] <= pivot)
            i++;
        key[j] = key[i];
        val[j] = val[i];
    }
    key[i] = pivot;
    val[i] = pivot_val;
    quicksort_key_bitmap(key, val, left, i - 1);
    quicksort_key_bitmap(key, val, i + 1, right);
}