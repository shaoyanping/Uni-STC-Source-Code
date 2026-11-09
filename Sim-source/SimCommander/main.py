from QuickProject import rt_dir, QproErrorString
from QuickProject.Commander import Commander
from . import *


app = Commander(name)

@app.command()
def compile(
    executable: str = name,
    scheduler: int = 8,
):
    """
    🔨 编译
    """
    # * 你可以在此函数的参数中添加你需要的define, bool类型参数将自动转换为-D <参数名>，其他类型参数将自动转换为-D <参数名>=<参数值>。因此推荐为bool类型参数设置默认值为False。
    # ! 不支持list类型参数
    include = " ".join([f"-I {path}" for path in includePath])
    source = " ".join([f"{path}" for path in sourcePath])
    lib_path = " ".join([f"-L {path}" for path in libPath])
    libs = " ".join([f"-l{lib}" for lib in usingLib])
    others = " ".join(other_flags)

    defines = []
    cur_func_fig = app.fig_table[1]

    for item in cur_func_fig["args"]:
        _name = item["name"]
        defines.append(f"-D {name}='{eval(_name)}'")
    for item in cur_func_fig["options"]:
        _name = item["name"].strip("-").replace("-", "_")
        if _name == "executable":
            continue
        if "args" not in item and eval(_name):
            defines.append(f"-D {_name.replace('-', '_').upper()}")
        elif "args" in item:
            defines.append(f"-D {_name.replace('-', '_').upper()}='{eval(_name)}'")
    defines = " ".join(defines)

    # print(
    #     f"{cc} -std={standard} -O{optimization} {include + ' ' if include else ''}{lib_path + ' ' if lib_path else ''}{source + ' ' if source else ''}{defines + ' ' if defines else ''}{libs + ' ' if libs else ''}{others + ' ' if others else ''}-o dist/{executable}"
    # )

    st, _ = external_exec(
        f"{cc} -std={standard} -O{optimization} {include + ' ' if include else ''}{lib_path + ' ' if lib_path else ''}{source + ' ' if source else ''}{defines + ' ' if defines else ''}{libs + ' ' if libs else ''}{others + ' ' if others else ''}-o dist/{executable}", without_output=True
    )
    if st:
        QproDefaultConsole.print(QproErrorString, f"Compile Simulator (Scheduler={scheduler}) Failed!")
        return
    else:
        QproDefaultConsole.print(QproInfoString, f"Compile Simulator (Scheduler={scheduler}) Succeeded!")
        with open('.build_config.txt', 'w') as f:
            f.write(f'{scheduler}\n')
    return st


def ensure_build_config(scheduler: int):
    import os
    build_config_path = '.build_config.txt'
    if not os.path.exists(build_config_path):
        app.real_call("compile", scheduler=scheduler)
    with open(build_config_path, 'r') as f:
        built_scheduler = int(f.readline().strip())
    if built_scheduler != scheduler:
        app.real_call("compile", scheduler=scheduler)


def transformer():
    """
    🏃 测试 Transformer
    """
    import pandas as pd

    result = pd.DataFrame(
        columns=[
            "pruning",
            "sparsity",
            "layer",
            "mp16",
            "mpnz",
        ] + pd_col[8:]
    )

    # prunings = "l0_regularization  magnitude_pruning  random_pruning  variational_dropout".split()
    prunings = ['magnitude_pruning']
    sparsity = "0.7 0.98".split()
    template_id = {
        "l0_regularization": "1",
        "magnitude_pruning": "0",
        "random_pruning": "0",
        "variational_dropout": "1",
    }
    ls = []
    csv_file = "dist/transformer.csv"

    if not os.path.exists(csv_file):
        for pruning in prunings:
            for sp in sparsity:
                for layer in range(1, 5):
                    ls.append(f"{pruning} {sp} {layer} {template_id[pruning]}")
                    st, ct = external_exec(
                        f"./dist/{name} transformer {pruning} {sp} {layer} {template_id[pruning]}",
                        without_output=True,
                    )
                    if st:
                        QproDefaultConsole.print(ct)
                    else:
                        result.loc[result.shape[0]] = ct.strip().split("\n")[-1].split(",")
        result.to_csv("dist/transformer.csv", index=False)
    result = pd.read_csv("dist/transformer.csv")
    result["128-smu-speedup"] = result["128-ds-cycle"] / result["128-smu-cycle"]
    ds128_energy = result[f"128-ds-a-ntr"] * 0.0625 + result[f"128-ds-b-ntr"] * 0.0625 + result[f"128-ds-c-ntr"] * 1.4
    rm128_energy = result[f"128-rm-a-ntr"] * 0.0625 + result[f"128-rm-b-ntr"] * 0.35 + result[f"128-rm-c-ntr"] * 0.7
    smu128_energy = result[f"128-smu-a-nut"] * 0.6875 + result[f"128-smu-b-nut"] * 0.6875 + result[f"128-smu-c-nut"] * 0.6875
    result["128-ds-energy"] = ds128_energy
    result["128-rm-energy"] = rm128_energy
    result["128-smu-energy"] = smu128_energy
    
    cols = 'sparsity,layer,128-ds-mac,128-rm-mac,128-smu-mac,128-ds-energy,128-rm-energy,128-smu-energy'.split(',')

    odf = pd.DataFrame(columns=cols)
    for item in cols:
        odf[item] = result[item]
    odf_07 = odf[odf['sparsity'] == 0.7].copy()
    odf_07 = odf_07.drop(columns=['sparsity'])
    odf_07['layer'] = odf_07['layer'].apply(lambda x: f"Transformer-{x}")
    odf_07 = odf_07.rename(columns={'layer': 'mtx'})
    odf_07.to_csv("dist/transformer_07.csv", index=False)

    odf_98 = odf[odf['sparsity'] == 0.98].copy()
    odf_98 = odf_98.drop(columns=['sparsity'])
    odf_98['layer'] = odf_98['layer'].apply(lambda x: f"Transformer-{x}")
    odf_98 = odf_98.rename(columns={'layer': 'mtx'})
    odf_98.to_csv("dist/transformer_98.csv", index=False)


def find_and_select_mtx(mname_ls):
    os.chdir(mtx_dir_path)
    res = []
    for mname in mname_ls:
        st, ct = external_exec(f"fd {mname}", without_output=True)
        mm = None
        if not st:
            ct = [i for i in ct.strip().split() if i.endswith(".mtx")]
            if len(ct) > 1:
                from . import _ask

                mm = _ask(
                    {
                        "type": "list",
                        "message": f"选择矩阵 ({mname})",
                        "choices": ct,
                        "default": ct[0],
                    }
                )
            elif len(ct) == 0:
                QproDefaultConsole.print(QproInfoString, "未找到矩阵:", mname)
            else:
                mm = ct[0]
        res.append(mm)
    os.chdir(rt_dir)
    return res


@app.command()
def run_sample():
    ensure_build_config(8)
    import pandas
    from .framework import start_framework
    from .draw_utils import draw_all_sample, draw_energy_breakdown, draw_fig19

    ls = ['pdb1HYS', 'cant', 'consph', 'pwtk', 'shipsec1', 'gupta3', 'opt1', 'crankseg_2']
    ls = find_and_select_mtx(ls)
    for kernel in ['spmv', 'spmspv', 'spmm', 'spgemm']:
        csv_path = f"dist/{kernel}/sample.csv"
        if os.path.exists(csv_path):
            continue
        odf = pandas.DataFrame(columns=pd_col)
        copy_ls = ls.copy()
        start_framework(kernel, copy_ls, odf, csv_path, 8, is_all=True)
        odf = pandas.read_csv(csv_path)
        odf['nzp16'] = odf['mpnz'] / odf['mp16']
        odf.sort_values(by='nzp16', inplace=True)
        odf.drop(columns=['nzp16'], inplace=True)
        odf.to_csv(csv_path, index=False)
    transformer()
    os.chdir(os.path.join(rt_dir, 'resnet50', 'sparse'))
    external_exec('qrun flow')
    QproDefaultConsole.print(QproInfoString, "ResNet50 Sparse Finished")
    os.chdir(os.path.join(rt_dir, 'resnet50', 'dense'))
    external_exec('qrun flow')
    QproDefaultConsole.print(QproInfoString, "ResNet50 Dense Finished")
    os.chdir(rt_dir)
    # merge ai results
    # do sparse
    csv_resnet_07 = pandas.read_csv(os.path.join(rt_dir, 'resnet50', 'sparse', 'resnet.csv'))
    csv_transformer_07 = pandas.read_csv(os.path.join(rt_dir, 'dist', 'transformer_07.csv'))
    csv_resnet_07 = csv_resnet_07[csv_resnet_07['mtx'] != 'ResNet50-0']
    csv_transformer_07 = csv_transformer_07[csv_transformer_07['mtx'] != 'Transformer-0']
    csv_transformer_07 = csv_transformer_07[csv_transformer_07['mtx'] != 'Transformer-5']
    sparse_df_list = [csv_resnet_07, csv_transformer_07]
    sparse_df = pandas.concat(sparse_df_list, axis=0)
    odf = sparse_df
    odf.to_csv('dist/ai/sample-Weight-Dense.csv', index=False)
    # do dense
    csv_resnet_dense = pandas.read_csv(os.path.join(rt_dir, 'resnet50', 'dense', 'resnet.csv'))
    csv_transformer_dense = pandas.read_csv(os.path.join(rt_dir, 'dist', 'transformer_98.csv'))
    csv_resnet_dense = csv_resnet_dense[csv_resnet_dense['mtx'] != 'ResNet50-0']
    csv_transformer_dense = csv_transformer_dense[csv_transformer_dense['mtx'] != 'Transformer-0']
    csv_transformer_dense = csv_transformer_dense[csv_transformer_dense['mtx'] != 'Transformer-5']
    dense_df_list = [csv_resnet_dense, csv_transformer_dense]
    dense_df = pandas.concat(dense_df_list, axis=0)
    odf = dense_df
    odf.to_csv('dist/ai/sample-Weight-Sparse.csv', index=False)
    draw_all_sample()
    draw_energy_breakdown()
    draw_fig19()


@app.command()
def spgemm2():
    ensure_build_config(8)
    import pandas as pd
    import matplotlib.pyplot as plt
    from .framework import start_framework

    b_sparsity = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    sigma_max = [ # extract from RM-STC's Paper
        [0.9, 0.798, 0.697, 0.597, 0.706, 0.795, 0.593, 0.391, 0.194], 
        [0.57927797833935, 0.550541516245487, 0.496389891696751, 0.399819494584838, 0.355595667870036, 0.384404332129964, 0.28783393501805, 0.185848375451264, 0.13086642599278], 
        [0.354868061874431, 0.318471337579618, 0.276615104640583, 0.240218380345769, 0.197452229299363, 0.166515013648772, 0.149226569608735, 0.123748862602366, 0.102820746132848]
    ]
    
    QproDefaultConsole.print(QproInfoString, "This experiment will take 3 to 5 hours, press Ctrl+C will use the existing results backup to draw figure.")
    csv = 'dist/spgemm2.csv'
    if not os.path.exists(csv):
        mtx_ls = [(i, 'genData/' + j) for i in ['genData/8192-1.0.mtx', 'genData/8192-0.5.mtx', 'genData/8192-0.2.mtx'] for j in [f'8192-0.{(100 - s) // 10}.mtx' for s in b_sparsity]]
        odf = pd.DataFrame(columns=["index", 'mtx', 'gamma_mac', 'trapezoid_mac', 'ds_mac', 'rm_mac', 'smu_mac'])
        start_framework('spgemm2', mtx_ls[::-1], odf, csv, is_all=True)

    unistc = [[], [], []]
    nvstc = [[], [], []]
    dsstc = [[], [], []]
    rmstc = [[], [], []]
    gamma = [[], [], []]
    trapezoid = [[], [], []]
    odf = pd.read_csv(csv if os.path.exists(csv) else 'dist/spgemm2-bk.csv')
    for a_sp in [1.0, 0.5, 0.2]:
        for b_sp in b_sparsity:
            row = odf[odf['mtx'] == f'{a_sp}-0.{(100 - b_sp) // 10}'].iloc[0]
            unistc_idx = 0 if a_sp == 1.0 else (1 if a_sp == 0.5 else 2)
            unistc[unistc_idx].append(float(row['smu_mac']))
            nvstc[unistc_idx].append(a_sp * (100 - b_sp) / 100)
            dsstc[unistc_idx].append(float(row['ds_mac']))
            rmstc[unistc_idx].append(float(row['rm_mac']))
            gamma[unistc_idx].append(float(row['gamma_mac']))
            trapezoid[unistc_idx].append(float(row['trapezoid_mac']))


    # 创建子图
    fig, axs = plt.subplots(1, 3, figsize=(9, 3.5))
    colors = ['#5A9893', '#FFC300', '#e07a4b', '#9f5657', '#536698', '#639568', '#7E447D']

    from QuickStart_Rhy.TuiTools.Table import qs_default_table
    table = qs_default_table([
        'Density', 'GAMMA', 'Sigma', 'Trapezoid', 'NV-STC', 'DS-STC', 'RM-STC', 'Uni-STC'
    ], title='Sigma, NV-STC, DS-STC, RM-STC, Uni-STC')

    # 绘制每个子图
    titles = ['$A$ Dense', '$A$ 50% Sparse', '$A$ 80% Sparse']
    improvement = [[], [], [], [], [], []]
    for i in range(3):
        sigma_mac = sum(sigma_max[i])
        nvstc_mac = sum(nvstc[i])
        dsstc_mac = sum(dsstc[i])
        rmstc_mac = sum(rmstc[i])
        unistc_mac = sum(unistc[i])
        trapezoid_mac = sum(trapezoid[i])
        gamma_mac = sum(gamma[i])
        # fixed_gamma = [(dsstc[i][j] + rmstc[i][j]) / 2 * 0.95 for j in range(9)]
        # gamma_mac = sum(fixed_gamma)
        # QproDefaultConsole.print(f"Sigma-MStr: {sigma_mac:.2f}, Uni-STC: {unistc_mac:.2f}, Improvement: {unistc_mac / sigma_mac:.2f}")
        # use table show
        table.add_row(
            titles[i], f"{gamma_mac:.2f}", f"{sigma_mac:.2f}", f"{trapezoid_mac:.2f}", f"{nvstc_mac:.2f}", f"{dsstc_mac:.2f}", f"{rmstc_mac:.2f}", f"{unistc_mac:.2f}"
        )
        improvement[0].append(unistc_mac / gamma_mac)
        improvement[1].append(unistc_mac / sigma_mac)
        improvement[2].append(unistc_mac / dsstc_mac)
        improvement[3].append(unistc_mac / rmstc_mac)
        improvement[4].append(unistc_mac / trapezoid_mac)
        improvement[5].append(unistc_mac / nvstc_mac)

        # axs[i].plot(b_sparsity, sigma_mstr[i], 'v-', label='SIGMA (M-str)', color=colors[0])
        axs[i].plot(b_sparsity, gamma[i], 'd-', label='GAMMA', color=colors[0])
        axs[i].plot(b_sparsity, sigma_max[i], 's-', label='SIGMA', color=colors[1])
        axs[i].plot(b_sparsity, nvstc[i], 'D-', label='NV-STC', color=colors[6])
        axs[i].plot(b_sparsity, dsstc[i], 'h-', label='DS-STC', color=colors[2])
        axs[i].plot(b_sparsity, rmstc[i], 'o-', label='RM-STC', color=colors[3])
        axs[i].plot(b_sparsity, unistc[i], '^-', label='Uni-STC (this work)', color=colors[5])
        axs[i].plot(b_sparsity, trapezoid[i], 'x-', label='Trapezoid', color=colors[4])

        axs[i].text(0.99, 0.92, titles[i], horizontalalignment='right', verticalalignment='center', transform=axs[i].transAxes, fontsize=14)
        axs[i].set_xlabel('$B$-sparsity(%)', fontsize=18)
        if i == 0:
            axs[i].set_ylabel('MAC Utilisation (%)', fontsize=18)
        axs[i].set_ylim(0, 1)
        ticks = axs[i].get_yticks()
        axs[i].set_yticks(ticks)
        axs[i].set_yticklabels(axs[i].get_yticklabels(), fontsize=12)
        axs[i].set_xticks(b_sparsity)
        axs[i].set_xticklabels(b_sparsity, fontsize=12)
        axs[i].grid(axis='y', linestyle='--')

    table.add_row(
        'Improvement', 
        f"{sum(improvement[0]) / len(improvement[0]):.2f}", 
        f"{sum(improvement[1]) / len(improvement[1]):.2f}", 
        f"{sum(improvement[4]) / len(improvement[4]):.2f}",
        f"{sum(improvement[5]) / len(improvement[5]):.2f}",
        f"{sum(improvement[2]) / len(improvement[2]):.2f}", 
        f"{sum(improvement[3]) / len(improvement[3]):.2f}",
        "1.0"
    )
    QproDefaultConsole.print(table, justify='center')

    # 调整布局
    handles, labels = axs[0].get_legend_handles_labels()
    reorder_idx = [0, 2, 1, 3, 6, 4, 5]  # Reorder indices to match the desired order
    new_handles = [handles[i] for i in reorder_idx]
    new_labels = [labels[i] for i in reorder_idx]
    fig.legend(new_handles, new_labels, loc='upper center', bbox_to_anchor=(0.50, 1.04), ncol=4, frameon=False, columnspacing=0.5, handletextpad=0.2, prop={'weight': 'bold', 'size': 14})
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig('fig/16.png', dpi=300)
    QproDefaultConsole.print()
    QproDefaultConsole.print(QproInfoString, "Figure saved to fig/16.png, press Ctrl+C to exit if the program is still running.")


@app.command()
def run_amg():
    ensure_build_config(8)
    mtx = ['spmsrtls', 'thermal1', 'Pres_Poisson', 'Chevron2', 'venkat25', 'bcsstk39', 'mc2depi', 'stomach', 'parabolic_fem', 'cant', 'TSOPF_RS_b300_c3', 'af_shell4', 'msdoor', 'CoupCons3D', 'nd24k', 'ldoor']
    spgemm_times = [3,3,6,3,6,9,12,3,6,18,18,3,6,6,18,6]
    spmv_times = [351,251,551,251,601,851,1101,351,601,1701,1701,351,601,601,1701,601]

    import pandas as pd
    from .framework import start_framework

    file_ls = find_and_select_mtx(mtx)
    for kernel in ['spmv', 'spgemm']:
        csv_path = f"dist/{kernel}/amg.csv"
        if os.path.exists(csv_path):
            continue
        odf = pd.DataFrame(columns=pd_col)
        copy_ls = file_ls.copy()
        start_framework(kernel, copy_ls, odf, csv_path, 8, is_all=True)

    spmv_8 = pd.read_csv('dist/spmv/amg.csv')
    spgemm_8 = pd.read_csv('dist/spgemm/amg.csv')
    spmv_8['mtx'] = spmv_8['mtx'].apply(lambda x: '_'.join(x.split('.')[0].split('_')[1:]) if '.mtx' in x else x)
    spgemm_8['mtx'] = spgemm_8['mtx'].apply(lambda x: '_'.join(x.split('.')[0].split('_')[1:]) if '.mtx' in x else x)

    odf = pd.DataFrame(columns=['matrix', 'DS-SpMV' ,'DS-SpGEMM','RM-SpMV','RM-SpGEMM','Uni-SpMV','Uni-SpGEMM','Tr-SpMV','Tr-SpGEMM','Ga-SpMV','Ga-SpGEMM','SpGEMM-times','SpMV-times'])
    odf['matrix'] = mtx
    odf['SpGEMM-times'] = spgemm_times
    odf['SpMV-times'] = spmv_times
    for i, row in odf.iterrows():
        mtx_name = row['matrix']
        spmv_row = spmv_8[spmv_8['mtx'] == mtx_name].iloc[0]
        spgemm_row = spgemm_8[spgemm_8['mtx'] == mtx_name].iloc[0]
        odf.at[i, 'DS-SpMV'] = spmv_row['64-ds-cycle']
        odf.at[i, 'DS-SpGEMM'] = spgemm_row['64-ds-cycle']
        odf.at[i, 'RM-SpMV'] = spmv_row['64-rm-cycle']
        odf.at[i, 'RM-SpGEMM'] = spgemm_row['64-rm-cycle']
        odf.at[i, 'Uni-SpMV'] = spmv_row['64-smu-cycle']
        odf.at[i, 'Uni-SpGEMM'] = spgemm_row['64-smu-cycle']
        odf.at[i, 'Tr-SpMV'] = spmv_row['64-trapezoid-cycle']
        odf.at[i, 'Tr-SpGEMM'] = spgemm_row['64-trapezoid-cycle']
        odf.at[i, 'Ga-SpMV'] = spmv_row['64-gamma-cycle']
        odf.at[i, 'Ga-SpGEMM'] = spgemm_row['64-gamma-cycle']
    odf.to_csv('dist/amg.csv', index=False)

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gmean

    df = pd.read_csv('dist/amg.csv')

    accelerators = ['DS', 'Ga', 'RM', 'Tr', 'Uni']
    categories = ['TOTAL', 'SpMV', 'SpGEMM']
    speedup_gmeans = {cat: [] for cat in categories}

    for acc in accelerators:
        df[f'{acc}-TOTAL'] = df[f'{acc}-SpMV'] * df['SpMV-times'] + df[f'{acc}-SpGEMM'] * 18
        for cat in categories:
            baseline_col = f'DS-{cat}'
            acc_col = f'{acc}-{cat}'
            
            speedup_series = df[baseline_col] / df[acc_col]
            speedup_gmeans[cat].append(gmean(speedup_series))

    speedup_gmeans['TOTAL'].insert(1, 1.42)  # SIGMA
    speedup_gmeans['SpMV'].insert(1, 1.78)  # SIGMA
    speedup_gmeans['SpGEMM'].insert(1, 1.12)  # SIGMA

    labels = ['SIGMA', 'GAMMA', 'RM-STC', 'Trapezoid', 'Uni-STC (ours)']
    total_means = speedup_gmeans['TOTAL'][1:]
    spmv_means = speedup_gmeans['SpMV'][1:]
    spgemm_means = speedup_gmeans['SpGEMM'][1:]

    x = np.arange(len(labels))
    width = 0.25

    QproDefaultConsole.print(x, total_means, spmv_means, spgemm_means)

    colors = ['#639568', '#9f5657', '#536698']

    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(figsize=(8, 3))
    rects2 = ax.bar(x - width, spmv_means, width, label='SpMV', color=colors[1], zorder=3)
    rects3 = ax.bar(x, spgemm_means, width, label='SpGEMM', color=colors[2], zorder=3)
    rects1 = ax.bar(x + width, total_means, width, label='All', color=colors[0], zorder=3)

    ax.set_ylabel('Geomean Speedup\n(vs. DS-STC)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

    ax.bar_label(rects1, padding=1, fmt='%.2f', fontsize=12)
    ax.bar_label(rects2, padding=1, fmt='%.2f', fontsize=12)
    ax.bar_label(rects3, padding=1, fmt='%.2f', fontsize=12)

    fig.tight_layout()
    plt.savefig("fig/21.png", dpi=300)
    QproDefaultConsole.print(QproInfoString, "Figure saved to fig/21.png")


def run_all_spmv(scheduler: int):
    """
    测试所有矩阵
    """
    import pandas
    csv_path = f"dist/spmv/odf-{scheduler}.csv"
    ls = [
        i
        for i in os.listdir(mtx_dir_path)
        if i.endswith(".mtx")
    ]
    if os.path.exists(csv_path):
        odf = pandas.read_csv(csv_path)
        have_mtx = set(odf["mtx"].to_list())
        ls = [i for i in ls if i not in have_mtx]
    else:
        odf = pandas.DataFrame(columns=pd_col)

    from .framework import start_framework

    start_framework("spmv", ls, odf, csv_path, scheduler, is_all=True)


def run_all_spmspv(scheduler: int):
    """
    测试所有矩阵
    """
    import pandas

    csv_path = f"dist/spmspv/odf-{scheduler}.csv"
    ls = [
        i
        for i in os.listdir(mtx_dir_path)
        if i.endswith(".mtx")
    ]
    if os.path.exists(csv_path):
        odf = pandas.read_csv(csv_path)
        have_mtx = set(odf["mtx"].to_list())
        ls = [i for i in ls if i not in have_mtx]
    else:
        odf = pandas.DataFrame(columns=pd_col)
        odf.to_csv(csv_path, index=False)

    from .framework import start_framework

    start_framework("spmspv", ls, odf, csv_path, scheduler, is_all=True)


def run_all_spgemm(snum: int, port: int = 8000):
    """
    测试csv文件
    """
    import pandas

    def use_mtx(mtx) -> bool:
        return (
            "mawi" not in mtx
            and "kmer" not in mtx
            and mtx
            not in [
                "sk-2005.mtx",
                "2826_kron_g500-logn21.mtx",
                "2829_com-Orkut.mtx",
                "2792_circuit5M.mtx",
                "2848_twitter7.mtx",
                "2742_FullChip.mtx",
                "2766_sx-stackoverflow.mtx",
                "2808_kron_g500-logn20.mtx",
                "2778_wikipedia-20070206.mtx",
                "2775_wikipedia-20061104.mtx",
                "2846_webbase-2001.mtx",
                "2844_mycielskian19.mtx",
                "2844_mycielskian20.mtx", # take too much time
                "2849_mycielskian20.mtx",
                "2562_rajat29.mtx" # l1nnz can't be stored in int
            ]
        )

    csv_path = f"dist/spgemm/odf-{snum}.csv"
    ls = [
        i
        for i in os.listdir(mtx_dir_path)
        if i.endswith(".mtx") and use_mtx(i)
    ]
    if os.path.exists(csv_path):
        odf = pandas.read_csv(csv_path)
        have_mtx = set(odf["mtx"].to_list())
        ls = [i for i in ls if i not in have_mtx]
        print(ls)
        if len(ls) == 0:
            return
    else:
        odf = pandas.DataFrame(columns=pd_col)
    from .framework import start_framework

    start_framework("spgemm", ls, odf, csv_path, snum, port, is_all=True)


def run_all_spmm(snum: int, port: int = 8000):
    """
    测试csv文件
    """
    import pandas

    def use_mtx(mtx) -> bool:
        return (
            "mawi" not in mtx
            and "kmer" not in mtx
            and mtx
            not in [
                "sk-2005.mtx",
                "2826_kron_g500-logn21.mtx",
                "2829_com-Orkut.mtx",
                "2792_circuit5M.mtx",
                "2848_twitter7.mtx",
                "2742_FullChip.mtx",
                "2766_sx-stackoverflow.mtx",
                "2808_kron_g500-logn20.mtx",
                "2778_wikipedia-20070206.mtx",
                "2775_wikipedia-20061104.mtx",
                "2846_webbase-2001.mtx",
                "2844_mycielskian19.mtx",
                "2562_rajat29.mtx" # l1nnz用int存不下
            ]
        )

    csv_path = f"dist/spmm/odf-{snum}.csv"
    ls = [
        i
        for i in os.listdir(mtx_dir_path)
        if i.endswith(".mtx") and use_mtx(i)
    ]
    if os.path.exists(csv_path):
        odf = pandas.read_csv(csv_path)
        have_mtx = set(odf["mtx"].to_list())
        ls = [i for i in ls if i not in have_mtx]
        print(ls)
        if len(ls) == 0:
            return
    else:
        odf = pandas.DataFrame(columns=pd_col)
    from .framework import start_framework

    start_framework("spmm", ls, odf, csv_path, snum, port, is_all=True)


@app.command()
def format():
    from .draw_utils import draw_format
    draw_format()


@app.command()
def run_all(snum: int = 8, do_not_draw: bool = False):
    """
    测试所有矩阵
    """
    from .draw_utils import four_kernels_boxplot
    ensure_build_config(snum)
    run_all_spmv(snum)
    run_all_spmspv(snum)
    run_all_spmm(snum)
    run_all_spgemm(snum)
    if not do_not_draw:
        four_kernels_boxplot()


@app.command()
def eed():
    from .draw_utils import auto_bk, pd, np
    import matplotlib.pyplot as plt
    import seaborn as sns

    app.real_call('compile', scheduler=4)
    app.real_call('run_all', snum=4, do_not_draw=True)
    app.real_call('compile', scheduler=8)
    app.real_call('run_all', snum=8, do_not_draw=True)
    app.real_call('compile', scheduler=16)
    app.real_call('run_all', snum=16, do_not_draw=True)

    DS_AREA = 0.032 + 0.003
    RM_AREA = 0.021 + 0.008 + 0.006
    SMU_4_AREA = 0.0385
    SMU_8_AREA = 0.0425
    SMU_16_AREA = 0.0605
    SMU_AREA = [SMU_4_AREA, SMU_8_AREA, SMU_16_AREA]

    sns.set_theme(style="whitegrid")

    KERNELS = ['SpMV', 'SpMSpV', 'SpMM', 'SpGEMM']
    MAC_CONFIGS = [64, 128][::-1]  # 128 first for row ordering
    PREC_CONFIGS = [64, 32][::-1]
    SCHEDULERS = [4, 8, 16]
    COLOR_ARR = ['#9f5657', '#536698', '#7ba05a', '#639568', '#4a7c59']  # Distinct colors per scheduler
    X_LABELS = ['DS-STC', 'RM-STC', 'Uni-STC (4)', 'Uni-STC (8)', 'Uni-STC (16)']

    # Initialize figure
    fig, axs = plt.subplots(2, 4, figsize=(8, 3), sharey='row')

    # Main processing loop
    for row_idx, mac in enumerate(MAC_CONFIGS):
        for col_idx, kernel in enumerate(KERNELS):
            ax = axs[row_idx, col_idx]
            eff_values = []

            once = pd.read_csv(auto_bk(f'dist/{kernel.lower()}/odf-8.csv'))
            if mac == 128:
                ds_energy = once[f"128-ds-a-ntr"]*0.0625 + once[f"128-ds-b-ntr"]*0.0625 + once[f"128-ds-c-ntr"]*1.4
                rm_energy = once[f"128-rm-a-ntr"]*0.0625 + once[f"128-rm-b-ntr"]*0.35 +  once[f"128-rm-c-ntr"]*0.7
            else:
                ds_energy = once[f"64-ds-a-ntr"]*0.0625 + once[f"64-ds-b-ntr"]*0.0625 + once[f"64-ds-c-ntr"]
                rm_energy = once[f"64-rm-a-ntr"]*0.0625 + once[f"64-rm-b-ntr"]*0.25 + once[f"64-rm-c-ntr"]*0.5
            rm_ep = (ds_energy / rm_energy) * (once[f'{mac}-ds-cycle'] / once[f'{mac}-rm-cycle'])
            rm_ep = rm_ep.mean()
            ds_ep = 1.0  # DS-STC is always 1.0
            eff_values.append(ds_ep / DS_AREA)
            eff_values.append(rm_ep / RM_AREA)
            # Process each scheduler for current MAC and kernel
            for sched_idx, sched in enumerate(SCHEDULERS):
                data = pd.read_csv(auto_bk(f'dist/{kernel.lower()}/odf-{sched}.csv'))
                if mac == 128:
                    ds_energy = data[f"128-ds-a-ntr"]*0.0625 + data[f"128-ds-b-ntr"]*0.0625 + data[f"128-ds-c-ntr"]*1.4
                    smu_energy = data[f"128-smu-a-nut"]*0.6875 + data[f"128-smu-b-nut"]*0.6875 + data[f"128-smu-c-nut"]*0.6875
                else:
                    ds_energy = data[f"64-ds-a-ntr"]*0.0625 + data[f"64-ds-b-ntr"]*0.0625 + data[f"64-ds-c-ntr"]
                    smu_energy = data[f"64-smu-a-nut"]*0.6875 + data[f"64-smu-b-nut"]*0.6875 + data[f"64-smu-c-nut"]*0.6875

                smu_ep = (ds_energy / smu_energy) * (data[f'{mac}-ds-cycle'] / data[f'{mac}-smu-cycle'])
                smu_ep = smu_ep.mean()
                eff_values.append(smu_ep / SMU_AREA[sched_idx])

            bar_width = 0.7
            x_pos = np.arange(5)

            for i in range(1, len(eff_values)):
                eff_values[i] = eff_values[i] / eff_values[0]
            eff_values[0] = 1.0  # DS-STC is always 1.0
            ax.bar(x_pos, eff_values, width=bar_width, color=COLOR_ARR, edgecolor='k', linewidth=0.5)
            
            ax.grid(axis='y', linestyle='--', alpha=0.6)
            ax.set_ylim(0, 12 if mac == 64 else 16)
            ax.set_xticks([])
            if col_idx == 0:
                if mac == 64:
                    ax.set_yticks([0, 4, 8, 12])
                else:
                    ax.set_yticks([0, 4, 8, 12, 16])
                ax.set_ylabel(f'EED\n(FP{PREC_CONFIGS[row_idx]})', fontsize=16, labelpad=10, fontweight='bold')
                ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)

            if row_idx == 1:
                ax.set_xlabel(kernel, fontsize=16, labelpad=8, fontweight='bold')

    legend_elements = [plt.Rectangle((0,0),1,1, color=COLOR_ARR[_id], label=s) for _id, s in enumerate(X_LABELS)]
    fig.legend(handles=legend_elements, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.02), fontsize=14, handlelength=0.5)
    # Final layout adjustments
    fig.align_labels()
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0.1, bottom=0.1)
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.savefig('fig/22.png', bbox_inches='tight', dpi=300)


def main():
    """
    * 注册为全局命令时, 默认采用main函数作为命令入口, 请勿将此函数用作它途.
    * When registering as a global command, default to main function as the command entry, do not use it as another way.
    """
    app()


if __name__ == "__main__":
    main()
