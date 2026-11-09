#pragma once
#include "./mload.h"
#include <tmatrix/Calculation/TileMatrix_x.h>

int test_transformer(const char *pruning, const char *sparsity, int layer, int template_id)
{
    // model path: ../Data/dlmc/transformer/{pruning}/{sparsity}/*.mtx
    TileMatrix *x = mmload("DLMC/x.mtx");
    const char *path_templates[] = {
        "DLMC/transformer/%s/%s/body_decoder_layer_%d_self_attention_multihead_attention_%s_fully_connected.mtx", 
        "DLMC/transformer/%s/%s/body_decoder_layer_%d_self_attention_multihead_attention_%s.mtx"
    };
    const char *path_template = path_templates[template_id];
    task_results sum32, sum64;

    char wq_path[256] = {0}, wk_path[256] = {0}, wv_path[256] = {0};
    sprintf(wq_path, path_template, pruning, sparsity, layer, "q");
    sprintf(wk_path, path_template, pruning, sparsity, layer, "k");
    sprintf(wv_path, path_template, pruning, sparsity, layer, "v");
    TileMatrix *wq = mmload(wq_path);
    TileMatrix *wk = mmload(wk_path);
    TileMatrix *wv = mmload(wv_path);

    TileMatrix *q = TileMatrix_GEMM_Sym_v2(x, wq);
    TileMatrix *k = TileMatrix_GEMM_Sym_v2(x, wk);
    TileMatrix *v = TileMatrix_GEMM_Sym_v2(x, wv);

    sum32 += TileMatrix_GEMM_Num_Sim<64>(x, wq, q);
    sum32 += TileMatrix_GEMM_Num_Sim<64>(x, wk, k);
    sum32 += TileMatrix_GEMM_Num_Sim<64>(x, wv, v);

    sum64 += TileMatrix_GEMM_Num_Sim<128>(x, wq, q);
    sum64 += TileMatrix_GEMM_Num_Sim<128>(x, wk, k);
    sum64 += TileMatrix_GEMM_Num_Sim<128>(x, wv, v);

    TileMatrix *qk = TileMatrix_GEMM_Sym_v2(q, k);
    TileMatrix *qk_v = TileMatrix_GEMM_Sym_v2(qk, v);

    sum32 += TileMatrix_GEMM_Num_Sim<64>(q, k, qk);
    sum32 += TileMatrix_GEMM_Num_Sim<64>(qk, v, qk_v);
    sum64 += TileMatrix_GEMM_Num_Sim<128>(q, k, qk);
    sum64 += TileMatrix_GEMM_Num_Sim<128>(qk, v, qk_v);

    sum32.ds_mac = sum32.mpnz * 1.0 / 64 / sum32.ds_cycle * 100.0;
    sum32.rm_mac = sum32.mpnz * 1.0 / 64 / sum32.rm_cycle * 100.0;
    sum32.smu_mac = sum32.mpnz * 1.0 / 64 / sum32.smu_cycle * 100.0;
    sum32.smu_speedup = std::max(sum32.ds_cycle, sum32.rm_cycle) * 1.0 / sum32.smu_cycle;
    sum64.ds_mac = sum64.mpnz * 1.0 / 128 / sum64.ds_cycle * 100.0;
    sum64.rm_mac = sum64.mpnz * 1.0 / 128 / sum64.rm_cycle * 100.0;
    sum64.smu_mac = sum64.mpnz * 1.0 / 128 / sum64.smu_cycle * 100.0;
    sum64.smu_speedup = std::max(sum64.ds_cycle, sum64.rm_cycle) * 1.0 / sum64.smu_cycle;

    printf("%s,%s,%lu,%lu,%lu,", pruning, sparsity, layer, sum32.mp16, sum32.mpnz);
    print_task_result(sum32, sum64);
    destroy_TileMatrix(wq);
    destroy_TileMatrix(wk);
    destroy_TileMatrix(wv);
    destroy_TileMatrix(q, true);
    destroy_TileMatrix(k, true);
    destroy_TileMatrix(v, true);
    destroy_TileMatrix(qk, true);
    destroy_TileMatrix(qk_v, true);

    return 0;
}