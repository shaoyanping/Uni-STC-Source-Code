from . import *
import pandas
import numpy as np
import pandas as pd

def data_withdraw(filepath: str, key: str, kernel: str = "SpGEMM", update_mac: bool = True):
    data = pandas.read_csv(filepath)  # df['mtx'], cycle0, cycle1, ...
    
    available_prefixes = []
    for prefix in ["128", "64"]:
        if any(col.startswith(f"{prefix}-") for col in data.columns):
            available_prefixes.append(prefix)

    if 'mpnz' in data:
        data['x'] = data["mpnz"] / data["mp16"]

    if key == 'mac' and update_mac:
        for p in available_prefixes:
            if f'{p}-ds-cycle' in data.columns:
                width = int(p)
                data[f'{p}-ds-mac'] = data["mpnz"] / data[f'{p}-ds-cycle'] / width * 100
                data[f'{p}-rm-mac'] = data["mpnz"] / data[f'{p}-rm-cycle'] / width * 100
                data[f'{p}-smu-mac'] = data["mpnz"] / data[f'{p}-smu-cycle'] / width * 100
    
    mtx = data["mtx"]
    
    if key == 'energy' or key == 'efficiency':
        if "128" in available_prefixes:
            if '128-ds-a-ntr' in data.columns:
                ds128_energy = (data["128-ds-a-ntr"] * 0.0625 + 
                                data["128-ds-b-ntr"] * 0.0625 + 
                                data["128-ds-c-ntr"] * 1.4)
                rm128_energy = (data["128-rm-a-ntr"] * 0.0625 + 
                                data["128-rm-b-ntr"] * 0.35 + 
                                data["128-rm-c-ntr"] * 0.7)
                smu128_energy = (data["128-smu-a-nut"] * 0.6875 + 
                                 data["128-smu-b-nut"] * 0.6875 + 
                                 data["128-smu-c-nut"] * 0.6875)
                
                data['128-rm-energy'] = ds128_energy / rm128_energy
                data['128-smu-energy'] = ds128_energy / smu128_energy
            else:
                if "128-ds-energy" in data and "128-rm-energy" in data:
                    data["128-rm-energy"] = data["128-ds-energy"] / data["128-rm-energy"]
                    data["128-smu-energy"] = data["128-ds-energy"] / data["128-smu-energy"]
            data['128-ds-energy'] = 1.0

        # 针对 64 的处理 (保留原始系数，注意 64-ds-c-ntr 没有乘数)
        if "64" in available_prefixes:
            if '64-ds-a-ntr' in data.columns:
                ds64_energy = (data["64-ds-a-ntr"] * 0.0625 + 
                               data["64-ds-b-ntr"] * 0.0625 + 
                               data["64-ds-c-ntr"]) # Original logic: no multiplier
                rm64_energy = (data["64-rm-a-ntr"] * 0.0625 + 
                               data["64-rm-b-ntr"] * 0.25 + 
                               data["64-rm-c-ntr"] * 0.5)
                smu64_energy = (data["64-smu-a-nut"] * 0.6875 + 
                                data["64-smu-b-nut"] * 0.6875 + 
                                data["64-smu-c-nut"] * 0.6875)

                data['64-rm-energy'] = ds64_energy / rm64_energy
                data['64-smu-energy'] = ds64_energy / smu64_energy
            else:
                if "64-ds-energy" in data and "64-rm-energy" in data:
                    data["64-rm-energy"] = data["64-ds-energy"] / data["64-rm-energy"]
                    data["64-smu-energy"] = data["64-ds-energy"] / data["64-smu-energy"]
            data['64-ds-energy'] = 1.0

    # 4. Cycle 计算处理 (动态)
    if key == 'cycle' or key == 'efficiency':
        print(update_mac)
        for p in available_prefixes:
            if not update_mac:
                data[f"{p}-rm-cycle"] = data[f"{p}-rm-mac"] / data[f"{p}-ds-mac"]
                data[f"{p}-smu-cycle"] = data[f"{p}-smu-mac"] / data[f"{p}-ds-mac"]
            else:
                # 只有当列存在时才计算
                if f"{p}-ds-cycle" in data and f"{p}-rm-cycle" in data:
                    data[f"{p}-rm-cycle"] = data[f"{p}-ds-cycle"] / data[f"{p}-rm-cycle"]
                    data[f"{p}-smu-cycle"] = data[f"{p}-ds-cycle"] / data[f"{p}-smu-cycle"]
            data[f'{p}-ds-cycle'] = 1.0

    # 5. Efficiency 最终合并 (动态)
    if key == "efficiency":
        for p in available_prefixes:
            data[f"{p}-rm-efficiency"] = data[f"{p}-rm-cycle"] * data[f"{p}-rm-energy"]
            data[f"{p}-smu-efficiency"] = data[f"{p}-smu-cycle"] * data[f"{p}-smu-energy"]
            data[f'{p}-ds-efficiency'] = 1.0
    
    # Check data has nan
    if data.isnull().values.any():
        QproDefaultConsole.print(f"Data has NaN in {filepath}, kernel {kernel}")
        for row in data.index:
            if data.loc[row].isnull().values.any():
                QproDefaultConsole.print(f"Row {row} has NaN")
                QproDefaultConsole.print(data.loc[row])
    # Check data has inf
    if data.isin([np.inf, -np.inf]).values.any():
        QproDefaultConsole.print(f"Data has Inf in {filepath}, kernel {kernel}")
        for row in data.index:
            if data.loc[row].isin([np.inf, -np.inf]).values.any():
                QproDefaultConsole.print(f"Row {row} has Inf")
                QproDefaultConsole.print(data.loc[row])

    # 6. 动态重组输出 DataFrame
    # 构建目标列列表，顺序为 [128相关, 64相关, mtx]
    target_columns = []
    # 保持 128 在前，64 在后的顺序 (如果都存在)
    sort_priority = ["128", "64"]
    final_prefixes = [p for p in sort_priority if p in available_prefixes]
    
    for p in final_prefixes:
        target_columns.extend([f"{p}-ds-{key}", f"{p}-rm-{key}", f"{p}-smu-{key}"])
    
    target_columns.append("mtx")

    if 'mpnz' in data:
        # 如果有 mpnz，需要按照 'x' 排序，并且 'x' 放在第一列
        extract_cols = ['x'] + target_columns
        # 使用 pandas 的 sort_values 替代 zip 排序，更健壮
        data = data[extract_cols].sort_values(by='x').reset_index(drop=True)
    else:
        # 如果没有 mpnz，不排序，x 设为 -1
        data = data[target_columns].copy()
        data['x'] = -1
        
    return data


def draw_all_sample():
    import math
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style="white")

    # 分组柱状图
    def df_scatter(sdf, mac_num):
        idx = 0
        df_num = pandas.DataFrame(
            columns=[
                'mtx',
                'mac',
                'energy',
                'edp',
                'type'
            ]
        )
        for i in range(len(sdf)):
            df_num.loc[idx] = [
                sdf.loc[i]['mtx'],
                sdf.loc[i][f'{mac_num}-ds-mac'],
                sdf.loc[i][f'{mac_num}-ds-energy'],
                sdf.loc[i][f'{mac_num}-ds-edp'],
                'DS-STC'
            ]
            idx += 1
            df_num.loc[idx] = [
                sdf.loc[i]['mtx'],
                sdf.loc[i][f'{mac_num}-rm-mac'],
                sdf.loc[i][f'{mac_num}-rm-energy'],
                sdf.loc[i][f'{mac_num}-rm-edp'],
                'RM-STC'
            ]
            idx += 1
            df_num.loc[idx] = [
                sdf.loc[i]['mtx'],
                sdf.loc[i][f'{mac_num}-smu-mac'],
                sdf.loc[i][f'{mac_num}-smu-energy'],
                sdf.loc[i][f'{mac_num}-smu-edp'],
               'Uni-STC'
            ]
            idx += 1
        return df_num
    
    def plot_barplot(ax, x, y, title, ylabel, ylim, xlabel='', data=None, rotate_xticks=False, ylabel_blank=False, legend: bool = False, yscale: bool = False, ticks: list = None):
        sns.lineplot(
            x=x,
            y=y,
            hue='type',
            data=data,
            ax=ax,
            hue_order=['DS-STC', 'RM-STC', 'Uni-STC'],
            palette=['#9f5657', '#536698', '#639568'],
            legend=legend,
            markers=True,
            style='type'
        )
        ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
        ax.set_ylabel('' if ylabel_blank else ylabel, fontweight='bold', fontsize=16)
        ax.set_xlabel(xlabel, fontweight='bold', fontsize=16)
        ax.set_yticks(ticks)
        if ticks:
            ax.set_yticklabels(ticks, fontsize=14)
        else:
            ax.set_yticklabels(['' for _ in ticks], fontsize=14)
        if legend:
            pass
        if rotate_xticks:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, horizontalalignment='center', fontsize=14)
        else:
            ax.set_xticklabels([])
        if yscale:
            ax.set_yscale('log', base=10)
        if ticks:
            ax.set_ylim(0, ticks[-1])
        else:
            ax.set_ylim(0, ylim)
        if ylabel_blank:
            ax.set_yticklabels(['' for _ in ticks], fontsize=14)

    def format_df(df, ax, column_idx, xlabel, ylabel_blank: bool = False, set_ylabel: bool = False, use_64: bool = True, set_yscale: bool = False):
        df['mtx'] = df['mtx'].apply(lambda x: '_'.join(x.split('.')[0].split('_')[1:]) if '.mtx' in x else x)
        if '128-ds-a-ntr' in df:
            df["128-ds-energy"] = df[f"128-ds-a-ntr"] * 0.0625 + df[f"128-ds-b-ntr"] * 0.0625 + df[f"128-ds-c-ntr"] * 1.4
            df["128-rm-energy"] = df[f"128-rm-a-ntr"] * 0.0625 + df[f"128-rm-b-ntr"] * 0.35 + df[f"128-rm-c-ntr"] * 0.7
            df["128-smu-energy"] = df[f"128-smu-a-nut"] * 0.6875 + df[f"128-smu-b-nut"] * 0.6875 + df[f"128-smu-c-nut"] * 0.6875
            if use_64:
                df["64-ds-energy"] = df[f"64-ds-a-ntr"] * 0.0625 + df[f"64-ds-b-ntr"] * 0.0625 + df[f"64-ds-c-ntr"]
                df["64-rm-energy"] = df[f"64-rm-a-ntr"] * 0.0625 + df[f"64-rm-b-ntr"] * 0.25 + df[f"64-rm-c-ntr"] * 0.5
                df["64-smu-energy"] = df[f"64-smu-a-nut"] * 0.6875 + df[f"64-smu-b-nut"] * 0.6875 + df[f"64-smu-c-nut"] * 0.6875
            

        df["128-smu-energy"] = df[f"128-ds-energy"] / df[f"128-smu-energy"]
        df["128-rm-energy"] = df[f"128-ds-energy"] / df[f"128-rm-energy"]
        df["128-ds-energy"] = 1.0

        df['128-smu-mac'] = df["128-smu-mac"] / df['128-ds-mac']
        df['128-rm-mac'] = df['128-rm-mac'] / df['128-ds-mac']
        df['128-ds-mac'] = 1.0

        df['128-smu-edp'] =  df['128-smu-mac'] * df['128-smu-energy']
        df['128-rm-edp'] = df['128-rm-mac'] * df['128-rm-energy']
        df['128-ds-edp'] = df['128-ds-mac'] * df['128-ds-energy']

        if use_64:
            df["64-smu-energy"] = df[f"64-ds-energy"] / df[f"64-smu-energy"]
            df["64-rm-energy"] =  df[f"64-ds-energy"] / df[f"64-rm-energy"]
            df["64-ds-energy"] = 1.0
            df['64-smu-mac'] = df['64-smu-mac'] / df['64-ds-mac']
            df['64-rm-mac'] = df['64-rm-mac'] / df['64-ds-mac']
            df['64-ds-mac'] = 1.0
            df['64-smu-edp'] = df['64-smu-mac'] * df['64-smu-energy']
            df['64-rm-edp'] = df['64-rm-mac'] * df['64-rm-energy']
            df['64-ds-edp'] = df['64-ds-mac'] * df['64-ds-energy']

            # print(column_idx, df[['mtx', '64-ds-energy', '64-rm-energy', '64-smu-energy']])

        find_max_128, find_max_64 = 0, 0
        for i in range(len(df)):
            if (t := df.loc[i]['128-smu-edp'] / df.loc[i]['128-rm-edp']) > find_max_128:
                find_max_128 = t
            if use_64 and (t := df.loc[i]['64-smu-edp'] / df.loc[i]['64-rm-edp']) > find_max_64:
                find_max_64 = t

        df_128 = df_scatter(df, 128)
        if use_64:
            df_64 = df_scatter(df, 64)

        column_idx_2_ticks = {
            0: {
                'mac': [0, 2, 4, 6],
                'energy': [0, 2, 4],
                'edp': [0, 12, 24],
            },
            1: {
                'mac': [0, 2, 4, 6, 7],
                'energy': [0, 2, 4],
                'edp': [0, 12, 24],
            },
            2: {
                'mac': [0, 1, 2, 3],
                'energy': [0, 4, 8],
                'edp': [0, 7, 14],
            },
            3: {
                'mac': [0, 1, 2, 3],
                'energy': [0, 4, 8],
                'edp': [0, 7, 14],
            },
            4: {
                'mac': [0, 1, 2, 3],
                'energy': [0, 6, 12],
                'edp': [0, 7, 14],
            },
            5: {
                'mac': [0, 1, 2, 3],
                'energy': [0, 6, 12],
                'edp': [0, 7, 14],
            }
        }
        plot_barplot(ax=ax[0][column_idx], x='mtx', y='mac', title='A100', ylim=mac_max, ylabel='Speedup' if set_ylabel else '', data=df_64 if use_64 else df_128, ylabel_blank=ylabel_blank, yscale=set_yscale, ticks=column_idx_2_ticks[column_idx]['mac'])
        plot_barplot(ax=ax[1][column_idx], x='mtx', y='energy', title='', ylim=energy_max, ylabel='Energy\nReduction' if set_ylabel else '', data=df_64 if use_64 else df_128, ylabel_blank=ylabel_blank, yscale=set_yscale, ticks=column_idx_2_ticks[column_idx]['energy'])
        plot_barplot(ax=ax[2][column_idx], x='mtx', y='edp', title='', xlabel=xlabel, ylim=edp_max, ylabel='Energy\nEfficiency' if set_ylabel else '', data=df_64 if use_64 else df_128, ylabel_blank=ylabel_blank, rotate_xticks=True, yscale=set_yscale, ticks=column_idx_2_ticks[column_idx]['edp'])

    def max_df(csvs: list):
        max_mac, max_energy, max_edp = 0, 0, 0
        for csv in csvs:
            df = pd.read_csv(csv)
            df["128-smu-energy"] = df["128-ds-energy"] / df["128-smu-energy"]
            df["128-rm-energy"] = df["128-ds-energy"] / df["128-rm-energy"]
            df["128-ds-energy"] = 1.0

            df["64-smu-energy"] = df["64-ds-energy"] / df["64-smu-energy"]
            df["64-rm-energy"] = df["64-ds-energy"] / df["64-rm-energy"]
            df["64-ds-energy"] = 1.0

            df['128-smu-mac'] = df["128-smu-mac"] / df['128-ds-mac']
            df['128-rm-mac'] = df['128-rm-mac'] / df['128-ds-mac']
            df['128-ds-mac'] = 1.0

            df['64-smu-mac'] = df['64-smu-mac'] / df['64-ds-mac']
            df['64-rm-mac'] = df['64-rm-mac'] / df['64-ds-mac']
            df['64-ds-mac'] = 1.0

            df['128-smu-edp'] = df['128-smu-energy'] * df['128-smu-mac']
            df['128-rm-edp'] = df['128-rm-energy'] * df['128-rm-mac']
            df['128-ds-edp'] = df['128-ds-energy'] * df['128-ds-mac']

            df['64-smu-edp'] = df['64-smu-energy'] * df['64-smu-mac']
            df['64-rm-edp'] = df['64-rm-energy'] * df['64-rm-mac']
            df['64-ds-edp'] = df['64-ds-energy'] * df['64-ds-mac']
            df128 = df_scatter(df, 128)
            df64 = df_scatter(df, 64)
            max_mac = max(max(max(df128['mac']), max(df64['mac'])), max_mac)
            max_energy = max(max(max(df128['energy']), max(df64['energy'])), max_energy)
            max_edp = max(max(max(df128['edp']), max(df64['edp'])), max_edp)
        return math.ceil(max_mac), math.ceil(max_energy), math.ceil(max_edp)

    def analyze(kernel: str, csv: str = ''):
        if not csv:
            csv = f'dist/{kernel.lower()}/sample.csv'
        data_speed = data_withdraw(csv, "mac", kernel, update_mac=False)
        data_energy = data_withdraw(csv, "energy", kernel, update_mac=False)
        data_efficiency = data_withdraw(csv, "efficiency", kernel, update_mac=False)

        x, ds128_speed, rm128_speed, smu128_speed, ds64_speed, rm64_speed, smu64_speed = data_speed['x'], data_speed['128-ds-mac'], data_speed['128-rm-mac'], data_speed['128-smu-mac'], data_speed['64-ds-mac'], data_speed['64-rm-mac'], data_speed['64-smu-mac']
        ds128_energy, rm128_energy, smu128_energy, ds64_energy, rm64_energy, smu64_energy = data_energy['128-ds-energy'], data_energy['128-rm-energy'], data_energy['128-smu-energy'], data_energy['64-ds-energy'], data_energy['64-rm-energy'], data_energy['64-smu-energy']
        ds128_efficiency, rm128_efficiency, smu128_efficiency, ds64_efficiency, rm64_efficiency, smu64_efficiency = data_efficiency['128-ds-efficiency'], data_efficiency['128-rm-efficiency'], data_efficiency['128-smu-efficiency'], data_efficiency['64-ds-efficiency'], data_efficiency['64-rm-efficiency'], data_efficiency['64-smu-efficiency']

        speedup_to_rm_128 = smu128_speed / rm128_speed
        speedup_to_rm_64 = smu64_speed / rm64_speed
        energy_reduction_to_rm_128 = smu128_energy / rm128_energy
        energy_reduction_to_rm_64 = smu64_energy / rm64_energy
        efficiency_to_rm_128 = smu128_efficiency / rm128_efficiency
        efficiency_to_rm_64 = smu64_efficiency / rm64_efficiency
        smu128_speed = smu128_speed / ds128_speed
        smu64_speed = smu64_speed / ds64_speed

        from QuickStart_Rhy.TuiTools.Table import qs_default_table
        table = qs_default_table([
            'Work', 'Max P(128)', 'Max P(64)', 'Max E(128)', 'Max E(64)', 'Max E(128)', 'Max E(64)', 'Aver P(128)', 'Aver P(64)', 'Aver E(128)', 'Aver E(64)', 'Aver EDP(128)', 'Aver EDP(64)'
        ], title=f"[bold]{kernel}[/] Speedup(P), Energy Reduction(E), Efficiency(EDP)")
        table.add_row(
            'DS-STC', f"{max(smu128_speed):.2f}", f"{max(smu64_speed):.2f}", f"{max(smu128_energy):.2f}", f"{max(smu64_energy):.2f}", f"{max(smu128_efficiency):.2f}", f"{max(smu64_efficiency):.2f}", f"{np.mean(smu128_speed):.2f}", f"{np.mean(smu64_speed):.2f}", f"{np.mean(smu128_energy):.2f}", f"{np.mean(smu64_energy):.2f}", f"{np.mean(smu128_efficiency):.2f}", f"{np.mean(smu64_efficiency):.2f}"
        )
        table.add_row(
            'RM-STC', f"{max(speedup_to_rm_128):.2f}", f"{max(speedup_to_rm_64):.2f}", f"{max(energy_reduction_to_rm_128):.2f}", f"{max(energy_reduction_to_rm_64):.2f}", f"{max(efficiency_to_rm_128):.2f}", f"{max(efficiency_to_rm_64):.2f}", f"{np.mean(speedup_to_rm_128):.2f}", f"{np.mean(speedup_to_rm_64):.2f}", f"{np.mean(energy_reduction_to_rm_128):.2f}", f"{np.mean(energy_reduction_to_rm_64):.2f}", f"{np.mean(efficiency_to_rm_128):.2f}", f"{np.mean(efficiency_to_rm_64):.2f}"
        )
        
        QproDefaultConsole.print(table, justify="center")
        QproDefaultConsole.print()
    
    def ai_analyze(csv):
        data_speed = data_withdraw(csv, "mac", 'AI', update_mac=False)
        data_energy = data_withdraw(csv, "energy", 'AI', update_mac=False)
        data_efficiency = data_withdraw(csv, "efficiency", 'AI', update_mac=False)

        resnet_speed = data_speed[data_speed['mtx'].str.contains('ResNet50', case=False)]
        transformer_speed = data_speed[data_speed['mtx'].str.contains('Transformer', case=False)]
        resnet_energy = data_energy[data_energy['mtx'].str.contains('ResNet50', case=False)]
        transformer_energy = data_energy[data_energy['mtx'].str.contains('Transformer', case=False)]
        resnet_efficiency = data_efficiency[data_efficiency['mtx'].str.contains('ResNet50', case=False)]
        transformer_efficiency = data_efficiency[data_efficiency['mtx'].str.contains('Transformer', case=False)]

        resnet_speedup_to_rm_128 = resnet_speed['128-smu-mac'] / resnet_speed['128-rm-mac']
        transformer_speedup_to_rm_128 = transformer_speed['128-smu-mac'] / transformer_speed['128-rm-mac']

        resnet_energy_reduction_to_rm_128 = resnet_energy['128-smu-energy'] / resnet_energy['128-rm-energy']
        transformer_energy_reduction_to_rm_128 = transformer_energy['128-smu-energy'] / transformer_energy['128-rm-energy']

        resnet_efficiency_to_rm_128 = resnet_efficiency['128-smu-efficiency'] / resnet_efficiency['128-rm-efficiency']
        transformer_efficiency_to_rm_128 = transformer_efficiency['128-smu-efficiency'] / transformer_efficiency['128-rm-efficiency']

        resnet_smu_speedup_128 = resnet_speed['128-smu-mac'] / resnet_speed['128-ds-mac']
        transformer_smu_speedup_128 = transformer_speed['128-smu-mac'] / transformer_speed['128-ds-mac']

        from QuickStart_Rhy.TuiTools.Table import qs_default_table

        table_resnet = qs_default_table([
            'Work', 'Max P(128)', 'Max E(128)', 'Max EDP(128)', 'Aver P(128)', 'Aver E(128)', 'Aver EDP(128)',
        ], title=f"[bold]ResNet50[/] Speedup(P), Energy Reduction(E), Efficiency(EDP)")

        table_resnet.add_row(
            'DS-STC', f"{max(resnet_smu_speedup_128):.2f}", f"{max(resnet_energy_reduction_to_rm_128):.2f}", f"{max(resnet_efficiency_to_rm_128):.2f}", f"{np.mean(resnet_smu_speedup_128):.2f}", f"{np.mean(resnet_energy_reduction_to_rm_128):.2f}", f"{np.mean(resnet_efficiency_to_rm_128):.2f}"
        )

        table_resnet.add_row(
            'RM-STC', f"{max(resnet_speedup_to_rm_128):.2f}", f"{max(resnet_energy_reduction_to_rm_128):.2f}", f"{max(resnet_efficiency_to_rm_128):.2f}", f"{np.mean(resnet_speedup_to_rm_128):.2f}", f"{np.mean(resnet_energy_reduction_to_rm_128):.2f}", f"{np.mean(resnet_efficiency_to_rm_128):.2f}"
        )

        table_transformer = qs_default_table([
            'Work', 'Max P(128)', 'Max E(128)', 'Max EDP(128)', 'Aver P(128)', 'Aver E(128)', 'Aver EDP(128)',
        ], title=f"[bold]Transformer[/] Speedup(P), Energy Reduction(E), Efficiency(EDP)")

        table_transformer.add_row(
            'DS-STC', f"{max(transformer_smu_speedup_128):.2f}", f"{max(transformer_energy_reduction_to_rm_128):.2f}", f"{max(transformer_efficiency_to_rm_128):.2f}", f"{np.mean(transformer_smu_speedup_128):.2f}", f"{np.mean(transformer_energy_reduction_to_rm_128):.2f}", f"{np.mean(transformer_efficiency_to_rm_128):.2f}"
        )

        table_transformer.add_row(
            'RM-STC', f"{max(transformer_speedup_to_rm_128):.2f}", f"{max(transformer_energy_reduction_to_rm_128):.2f}", f"{max(transformer_efficiency_to_rm_128):.2f}", f"{np.mean(transformer_speedup_to_rm_128):.2f}", f"{np.mean(transformer_energy_reduction_to_rm_128):.2f}", f"{np.mean(transformer_efficiency_to_rm_128):.2f}"
        )

        QproDefaultConsole.print(table_resnet, justify="center")
        QproDefaultConsole.print()
        QproDefaultConsole.print(table_transformer, justify="center")
        QproDefaultConsole.print()


    kernels = ['SpMV', 'SpMSpV', 'SpMM', 'SpGEMM']
    for i in range(4):
        QproDefaultConsole.rule(f"Analyze {kernels[i]}")
        analyze(kernels[i])
    QproDefaultConsole.rule("Analyze AI Dense")
    ai_analyze('dist/ai/sample-Weight-Dense.csv')
    QproDefaultConsole.rule("Analyze AI Sparse")
    ai_analyze('dist/ai/sample-Weight-Sparse.csv')

    fig, ax = plt.subplots(3, 6, figsize=(14, 6), width_ratios=[10, 10, 10, 10, 10, 10])
    mac_max, energy_max, edp_max = max_df(['dist/spmv/sample.csv', 'dist/spmspv/sample.csv'])
    
    format_df(pd.read_csv('dist/spmv/sample.csv'), ax, 0, 'SpMV', set_ylabel=True)
    format_df(pd.read_csv('dist/spmspv/sample.csv'), ax, 1, 'SpMSpV', ylabel_blank=True)

    mac_max, energy_max, edp_max = max_df(['dist/spmm/sample.csv', 'dist/spgemm/sample.csv'])
    mac_max, edp_max = 3, 8
    format_df(pd.read_csv('dist/spmm/sample.csv'), ax, 2, 'SpMM')
    format_df(pd.read_csv('dist/spgemm/sample.csv'), ax, 3, 'SpGEMM', ylabel_blank=True)

    mac_max, energy_max, edp_max = 3, 6, 8
    format_df(pd.read_csv('dist/ai/sample-Weight-Dense.csv'), ax, 4, 'Inference (Dense)', use_64=False)
    format_df(pd.read_csv('dist/ai/sample-Weight-Sparse.csv'), ax, 5, 'Inference (Sparse)', ylabel_blank=True, use_64=False)
    
    handles = [
        plt.Line2D([0], [0], color='#9f5657', lw=1.5, label='DS-STC', marker='o'),
        plt.Line2D([0], [0], color='#536698', lw=1.5, label='RM-STC', marker='x'),
        plt.Line2D([0], [0], color='#639568', lw=1.5, label='Uni-STC', marker='s')
    ]
    labels = ['DS-STC (baseline, always $1.0\\times$ in the sub-figures)', 'RM-STC', 'Uni-STC']
    fig.legend(handles, labels, loc='upper left', ncol=3, fontsize=14, bbox_to_anchor=(0.06, 1))
    fig.align_labels()
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0.1, bottom=0.1)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'fig/17.png', dpi=300)
    plt.close()


def draw_energy_breakdown():
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    csv = 'dist/spgemm/sample.csv'
    df = pd.read_csv(csv)
    df['nzp16'] = df['mpnz'] / df['mp16']

    # 按照 nzp16 排序
    df = df.sort_values(by='nzp16', ascending=False)

    # 选取特定的样本行
    ilocs = [7,6,5,4,3,2,1,0]
    df = df.iloc[ilocs]

    fig, ax = plt.subplots(2, 4, figsize=(61,35))
    x = ['DS-STC', 'RM-STC', 'Uni-STC']
    colors = ['#5E71A2' , '#6E9F73','#A96162']

    df['mtx'] = df['mtx'].apply(lambda x: '_'.join(x.split('.')[0].split('_')[1:]) if '.mtx' in x else x)
    ds128_energy = df[f"128-ds-a-ntr"] * 0.0625 + df[f"128-ds-b-ntr"] * 0.0625 + df[f"128-ds-c-ntr"] * 1.4
    rm128_energy = df[f"128-rm-a-ntr"] * 0.0625 + df[f"128-rm-b-ntr"] * 0.35 + df[f"128-rm-c-ntr"] * 0.7
    smu128_energy = df[f"128-smu-a-nut"] * 0.6875 + df[f"128-smu-b-nut"] * 0.6875 + df[f"128-smu-c-nut"] * 0.6875

    ds64_energy = df[f"64-ds-a-ntr"] * 0.0625 + df[f"64-ds-b-ntr"] * 0.0625 + df[f"64-ds-c-ntr"]
    rm64_energy = df[f"64-rm-a-ntr"] * 0.0625 + df[f"64-rm-b-ntr"] * 0.25 + df[f"64-rm-c-ntr"] * 0.5
    smu64_energy = df[f"64-smu-a-nut"] * 0.6875 + df[f"64-smu-b-nut"] * 0.6875 + df[f"64-smu-c-nut"] * 0.6875

    df['128-ds-energy'] = ds128_energy
    df['128-rm-energy'] = rm128_energy
    df['128-smu-energy'] = smu128_energy
    df['64-ds-energy'] = ds64_energy
    df['64-rm-energy'] = rm64_energy
    df['64-smu-energy'] = smu64_energy
    df['128-ds-a-energy'] = df['128-ds-a-ntr'] * 0.0625
    df['128-ds-b-energy'] = df['128-ds-b-ntr'] * 0.0625
    df['128-ds-c-energy'] = df['128-ds-c-ntr'] * 1.4
    df['128-rm-a-energy'] = df['128-rm-a-ntr'] * 0.0625
    df['128-rm-b-energy'] = df['128-rm-b-ntr'] * 0.35
    df['128-rm-c-energy'] = df['128-rm-c-ntr'] * 0.7
    df['128-smu-a-energy'] = df['128-smu-a-nut'] * 0.6875
    df['128-smu-b-energy'] = df['128-smu-b-nut'] * 0.6875
    df['128-smu-c-energy'] = df['128-smu-c-nut'] * 0.6875
    df['64-ds-a-energy'] = df['64-ds-a-ntr'] * 0.0625
    df['64-ds-b-energy'] = df['64-ds-b-ntr'] * 0.0625
    df['64-ds-c-energy'] = df['64-ds-c-ntr'] * 1.0
    df['64-rm-a-energy'] = df['64-rm-a-ntr'] * 0.0625
    df['64-rm-b-energy'] = df['64-rm-b-ntr'] * 0.25
    df['64-rm-c-energy'] = df['64-rm-c-ntr'] * 0.5
    df['64-smu-a-energy'] = df['64-smu-a-nut'] * 0.6875
    df['64-smu-b-energy'] = df['64-smu-b-nut'] * 0.6875
    df['64-smu-c-energy'] = df['64-smu-c-nut'] * 0.6875

    df['128-ds-a-energy'] = df['128-ds-a-energy'] / df['128-ds-energy']
    df['128-ds-b-energy'] = df['128-ds-b-energy'] / df['128-ds-energy']
    df['128-ds-c-energy'] = df['128-ds-c-energy'] / df['128-ds-energy']
    df['128-rm-a-energy'] = df['128-rm-a-energy'] / df['128-rm-energy']
    df['128-rm-b-energy'] = df['128-rm-b-energy'] / df['128-rm-energy']
    df['128-rm-c-energy'] = df['128-rm-c-energy'] / df['128-rm-energy']
    df['128-smu-a-energy'] = df['128-smu-a-energy'] / df['128-smu-energy']
    df['128-smu-b-energy'] = df['128-smu-b-energy'] / df['128-smu-energy']
    df['128-smu-c-energy'] = df['128-smu-c-energy'] / df['128-smu-energy']

    df['64-ds-a-energy'] = df['64-ds-a-energy'] / df['64-ds-energy']
    df['64-ds-b-energy'] = df['64-ds-b-energy'] / df['64-ds-energy']
    df['64-ds-c-energy'] = df['64-ds-c-energy'] / df['64-ds-energy']
    df['64-rm-a-energy'] = df['64-rm-a-energy'] / df['64-rm-energy']
    df['64-rm-b-energy'] = df['64-rm-b-energy'] / df['64-rm-energy']
    df['64-rm-c-energy'] = df['64-rm-c-energy'] / df['64-rm-energy']
    df['64-smu-a-energy'] = df['64-smu-a-energy'] / df['64-smu-energy']
    df['64-smu-b-energy'] = df['64-smu-b-energy'] / df['64-smu-energy']
    df['64-smu-c-energy'] = df['64-smu-c-energy'] / df['64-smu-energy']

    df["128-smu-energy"] = df["128-smu-energy"] / df["128-ds-energy"]
    df["128-rm-energy"] = df["128-rm-energy"] / df["128-ds-energy"]
    df["128-ds-energy"] = 1.0
        
    df["64-smu-energy"] = df["64-smu-energy"] / df["64-ds-energy"]
    df["64-rm-energy"] =  df["64-rm-energy"] / df["64-ds-energy"]
    df["64-ds-energy"] = 1.0

    df['128-rm-a-energy'] = df['128-rm-a-energy'] * df['128-rm-energy']
    df['128-rm-b-energy'] = df['128-rm-b-energy'] * df['128-rm-energy']
    df['128-rm-c-energy'] = df['128-rm-c-energy'] * df['128-rm-energy']
    df['128-smu-a-energy'] = df['128-smu-a-energy'] * df['128-smu-energy']
    df['128-smu-b-energy'] = df['128-smu-b-energy'] * df['128-smu-energy']
    df['128-smu-c-energy'] = df['128-smu-c-energy'] * df['128-smu-energy']
    df['64-rm-a-energy'] = df['64-rm-a-energy'] * df['64-rm-energy']
    df['64-rm-b-energy'] = df['64-rm-b-energy'] * df['64-rm-energy']
    df['64-rm-c-energy'] = df['64-rm-c-energy'] * df['64-rm-energy']
    df['64-smu-a-energy'] = df['64-smu-a-energy'] * df['64-smu-energy']
    df['64-smu-b-energy'] = df['64-smu-b-energy'] * df['64-smu-energy']
    df['64-smu-c-energy'] = df['64-smu-c-energy'] * df['64-smu-energy']

    df = df.iloc[ilocs]

    # 绘制堆叠柱状图
    for i in range(2):
        for j in range(4):
            mt125 = np.array(df.iloc[i*4+j][['64-ds-a-energy', '64-rm-a-energy', '64-smu-a-energy']])
            mt25 = np.array(df.iloc[i*4+j][['64-ds-b-energy', '64-rm-b-energy', '64-smu-b-energy']]) 
            mt50 = np.array(df.iloc[i*4+j][['64-ds-c-energy', '64-rm-c-energy', '64-smu-c-energy']]) 

            ax[i][j].bar(x, mt125, color=colors[0], label='Read A', width=0.47, align='center')
            ax[i][j].bar(x, mt25, color=colors[1], label='Read B', bottom=mt125, width=0.47, align='center')
            ax[i][j].bar(x, mt50, color=colors[2], label='Write C', bottom=(mt125 + mt25), width=0.47, align='center')
            total = mt125 + mt25 + mt50

            ax[i][j].set_ylim(0, max(total) * 1.2)
            ax[i][j].yaxis.set_major_formatter(ScalarFormatter())
            ax[i][j].tick_params(axis='y', labelsize=100)

            ax[i][j].set_xticks([-0.07, 1, 2.15]) 
            ax[i][j].set_xticklabels(x, fontsize=80) 
            
            for tick in ax[i][j].get_xticklabels():
                if tick.get_text() == 'Uni-STC':
                    tick.set_color('black')
                    tick.set_fontweight('bold')
                        
                        
            ax[i][j].tick_params(axis='y', pad=40)
            ax[i][j].tick_params(axis='x', pad=60) 
            ax[i][j].set_xlabel(f'{df.iloc[i*4+j]["mtx"]}\n\n', fontsize=140)

        fig.text(0.018, 0.5, 'Normalized Energy', ha='center', va='center', rotation='vertical', fontsize=140)

        handles, labels = ax[0][0].get_legend_handles_labels() 

    fig.legend(handles[:3], labels[:3],
        loc='upper center', 
        fontsize=140, 
        frameon=False, 
        bbox_to_anchor=(0.517, 1.043),
        ncol=3,
        columnspacing=1,
        handlelength=0.8
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.935,bottom= 0.12,left=0.09,right=0.97,hspace=0.35, wspace=0.33)
    plt.savefig(f"fig/18.png", dpi=300)
    plt.close()

def draw_fig19():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    csv = 'dist/spgemm/sample.csv'
    df = pd.read_csv(csv)
    df['nzp16'] = df['mpnz'] / df['mp16']
    df['mtx'] = df['mtx'].apply(lambda x: '_'.join(x.split('.')[0].split('_')[1:]) if '.mtx' in x else x)

    fig, ax = plt.subplots(2, 1, figsize=(57, 25))  # 4 行 4 列
    x = ['DS-STC', 'RM-STC', 'Uni-STC']
    colors = ['#A96162', '#5E71A2', '#6E9F73']

    # 用于存储图例句柄
    handles = []
    labels = ['DS-STC', 'RM-STC', 'Uni-STC']

    fixed_height = 1

    width = 0.3 
    group_spacing = 1
    
    network_traffic_data = []
    average_scale_data = []

    # 绘制第一个柱状图
    for i in range(8):
        ax[0].set_ylim(0,1)
        ax[0].set_xlim(-2.35, 7.37)
        ax[0].set_yticks(np.arange(0, 1.5, 0.5))
        mt125 = np.array(df.iloc[i][['64-ds-c-ntr', '64-rm-c-ntr', '64-smu-c-ntr']]) / 1e6

        current_first_height = mt125[0]
        scale_factor = fixed_height / current_first_height
        mt125[1:] = mt125[1:] * scale_factor 
        mt125[0] = fixed_height

        # 存储数据
        network_traffic_data.append({
            'matrix': df.iloc[i]['mtx'],
            'DS-STC': mt125[0],
            'RM-STC': mt125[1],
            'Uni-STC': mt125[2]
        })

        x_positions = (np.arange(0, 0.85, 0.3)-group_spacing) + i * (len(x) + group_spacing) * width - group_spacing

        ax[0].bar(x_positions, mt125, color=colors, width=width, label=f'{df.iloc[i]["mtx"]}')
        ax[0].set_ylabel('(a) Network\nTraffic', fontsize=100, fontweight='bold', labelpad=45)


    for i in range(8):
        ax[1].set_ylim(0, 0.1)
        ax[1].set_xlim(-2.35, 7.37)

        ax[1].set_yticks(np.arange(0, 0.15, 0.05))  # 0, 0.5, 1.0, 1.5, 2.0
        mt125 = np.array(df.iloc[i][['64-ds-c-ntr', '64-rm-c-ntr', '64-smu-c-ntr']]) 

        # 计算第一个柱子的当前高度
        current_first_height = mt125[0]

        # 将第一个柱子的高度固定
        scale_factor = fixed_height / current_first_height
        mt125[1:] = mt125[1:] * scale_factor  # 调整其他柱子的高度
        mt125[0] = fixed_height
        
        abc = mt125[1]/0.0625
        mt125[1] = 0.0625
        mt125[2] = mt125[2]/abc

        x_positions = (np.arange(0, 0.85, 0.3)-group_spacing) + i * (len(x) + group_spacing) * width - group_spacing
        

        ax[1].bar(x_positions, mt125, color=colors, width=width, label=f'{df.iloc[i]["mtx"]}')
        ax[1].set_xlabel('')
        ax[1].set_ylabel('(b) Average\nScale', fontsize=100, fontweight='bold')

        # 存储数据
        average_scale_data.append({
            'matrix': df.iloc[i]['mtx'],
            'DS-STC': mt125[0],
            'RM-STC': mt125[1],
            'Uni-STC': mt125[2]
        })

    vpos = 0.23
    # 设置每一组柱状图的标签
    for i in range(8):
        if(i == 0):
            fig.text(0.134, vpos, f'{df.iloc[i]["mtx"]}', ha='center', va='top', fontsize=90,rotation=45)
        if(i == 1):
            fig.text(0.230, vpos, f'{df.iloc[i]["mtx"]}', ha='center', va='top', fontsize=90,rotation=45)
        if(i==2):
            fig.text(0.335, vpos, f'{df.iloc[i]["mtx"]}', ha='center', va='top', fontsize=90,rotation=45)
        if(i==3):
            fig.text(0.470, vpos, f'{df.iloc[i]["mtx"]}', ha='center', va='top', fontsize=90,rotation=45)
        if(i==4):
            fig.text(0.584, vpos, f'{df.iloc[i]["mtx"]}', ha='center', va='top', fontsize=90,rotation=45) 
        if(i==5):
            fig.text(0.674, vpos, f'{df.iloc[i]["mtx"]}', ha='center', va='top', fontsize=90,rotation=45) 
        if(i==6):
            fig.text(0.804, vpos, f'{df.iloc[i]["mtx"]}', ha='center', va='top', fontsize=90,rotation=45) 
        if(i==7):
            fig.text(0.904, vpos, f'{df.iloc[i]["mtx"]}', ha='center', va='top', fontsize=90,rotation=45) 


    # 获取图例句柄：获取所有柱状图的句柄
    handles = [bar for bar in ax[0].get_children() if isinstance(bar, plt.Rectangle)]
    fig.legend(handles, labels,
            loc='upper center',
            fontsize=140,
            frameon=False,
            bbox_to_anchor=(0.55, 1.05),  # 调整图例位置，确保它不会超出画布
            ncol=3,  # 设置图例为横向排列，ncol 是列数
            columnspacing=3.5,  # 控制列之间的间距
            handlelength=1)  # 控制图例符号的长度

    for axis in ax:
        axis.tick_params(axis='y', labelsize=75)  # 设置 y 轴刻度标签的字体大小为 40

    vlabel_pos = 0.57
    fig.text(0.132, vlabel_pos, f'1.0', ha='center', va='top', fontsize=90)  # 第一个1.0
    fig.text(0.242, vlabel_pos, f'1.0', ha='center', va='top', fontsize=90)  # 第二个1.0
    fig.text(0.352, vlabel_pos, f'1.0', ha='center', va='top', fontsize=90)  # 第三个1.0
    fig.text(0.459, vlabel_pos, f'1.0', ha='center', va='top', fontsize=90)  # 第四个1.0
    fig.text(0.571, vlabel_pos, f'1.0', ha='center', va='top', fontsize=90)  # 第五个1.0
    fig.text(0.681, vlabel_pos, f'1.0', ha='center', va='top', fontsize=90)  # 第六个1.0
    fig.text(0.791, vlabel_pos, f'1.0', ha='center', va='top', fontsize=90)  # 第七个1.0
    fig.text(0.902, vlabel_pos, f'1.0', ha='center', va='top', fontsize=90)  # 第八个1.0

    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=vpos, left=0.1, right=0.99, hspace=0.29, wspace=0.2)
    plt.savefig(f"fig/19.png", dpi=300)


def draw_format():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    data = pd.read_csv("dist/spmv/odf-8-bk.csv")[['mtx', 'm', 'n', 'mp16', 'mp4', 'mpnz']]
    data['ipb'] = data['mpnz'] / data['mp16']
    data['_m'] = np.ceil(data['m'] / 16).astype(int)
    data['_m4'] = np.ceil(data['m'] / 4).astype(int)
    data['csr size'] = (data['m'] + data['mpnz'] + 1) * 4
    data['bsr16 size'] = (data['_m'] + data['mp16'] + 1) * 4 + (data['mp16'] * 256 - data['mpnz']) * 8
    data['bsr4 size'] = (data['_m4'] + data['mp4'] + 1) * 4 + (data['mp4'] * 16 - data['mpnz']) * 8
    data['tile size'] = (data['_m'] + 3 * data['mp16'] + 3) * 4 + (data['mp16'] + data['mp4']) * 2 + data['mp4']
    
    # Calculate space reduction relative to CSR
    base_label = 'csr'
    for label in ['tile', 'bsr16', 'bsr4']:
        data[f'{label} size'] = data[f'{base_label} size'] / data[f'{label} size']
    
    # --- Binning by 'ipb' ---
    # Create power-of-two bins for the 'ipb' column
    data_filtered = data[data['ipb'] > 0].copy() # Use only data with ipb > 0 for log scale
    min_power = np.floor(np.log2(data_filtered['ipb'].min()))
    max_power = np.ceil(np.log2(data_filtered['ipb'].max()))
    
    powers = np.arange(min_power, max_power + 1)
    bins = 2**powers
    labels = [f'[$2^{{{int(powers[i])}}} \\to 2^{{{int(powers[i+1])}}}$)' for i in range(len(powers) - 1)]
    
    data_filtered['ipb_bin'] = pd.cut(data_filtered['ipb'], bins=bins, labels=labels, right=False, include_lowest=True)

    # --- Data Reshaping for Plotting ---
    # Melt the dataframe to have a long format suitable for Seaborn's hue
    melt_data = data_filtered.melt(
        id_vars=['ipb_bin'],
        value_vars=['bsr16 size', 'bsr4 size', 'tile size'],
        var_name='Format',
        value_name='Space Reduction'
    ).dropna(subset=['ipb_bin', 'Space Reduction']) # Drop rows if bin or value is missing

    # Clean up format names for the legend
    format_mapping = {
        'tile size': 'BBC\n(this work)',
        'bsr16 size': 'BSR\n(16x16)',
        'bsr4 size': 'BSR\n(4x4)'
    }
    melt_data['Format'] = melt_data['Format'].map(format_mapping)
    
    # --- Plotting ---
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(8, 4), dpi=300)
    
    ax = sns.boxplot(
        x='ipb_bin',
        y='Space Reduction',
        hue='Format',
        data=melt_data,
        palette={'BSR\n(16x16)': '#9f5657', 'BSR\n(4x4)': '#536698', 'BBC\n(this work)': '#639568'},
        hue_order=['BSR\n(16x16)', 'BSR\n(4x4)', 'BBC\n(this work)'],
        showfliers=False, # Hide outliers for a cleaner plot
        linewidth=1.2
    )

    # --- Styling and Annotations ---
    # Add CSR baseline
    ax.axhline(1, color='#ffb74d', linestyle='--', linewidth=2, zorder=3, label='CSR\n(baseline)')
    
    # Set log scale for y-axis to handle large range of reduction values
    ax.set_yscale('log')
    
    ax.set_xlabel("Non-zero elements per block", fontsize=18)
    ax.set_ylabel("Space\nReduction", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Improve legend
    handles, labels = ax.get_legend_handles_labels()
    # 将图例放在图表上方中央
    ax.legend(
        handles, 
        labels, 
        fontsize=14, 
        loc='lower center',            # 将图例的下边缘中心作为锚点
        bbox_to_anchor=(0.5, 0.95),    # 将锚点放置在图表顶部中央稍上的位置
        ncol=4,                        # 将4个图例项排成一行
        frameon=False                  # 移除图例边框，更美观
    )

    # 调整布局，为顶部的图例留出空间 (rect的最后一个参数控制顶部边界)
    plt.tight_layout(rect=[0, 0, 1, 0.98]) 
    plt.savefig(f"fig/15.png", bbox_inches='tight', dpi=300)
    QproDefaultConsole.print(QproInfoString, "Figure saved as fig/15.png")


def auto_bk(csv_path):
    bk_path = csv_path.replace('.csv', '-bk.csv')
    if not os.path.exists(csv_path):
        return bk_path
    else:
        csv = pd.read_csv(csv_path)
        if len(csv) < 2100:
            return bk_path
        return csv_path

def four_kernels_boxplot():
    """
    Generates a 3x4 grid of box plots to visualize performance, energy, and efficiency
    for four different sparse kernels across three architectures.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    colors = {'DS-STC': '#9f5657', 'RM-STC': '#536698', 'Uni-STC': '#639568'}

    def _pack_data(kernel):
        kernel = kernel.lower()
        data_mac = data_withdraw(auto_bk(f"dist/{kernel}/odf-8.csv"), "mac", kernel)
        data_cycle = data_withdraw(auto_bk(f"dist/{kernel}/odf-8.csv"), "cycle", kernel)
        data_energy = data_withdraw(auto_bk(f"dist/{kernel}/odf-8.csv"), "energy", kernel)
        data_efficiency = data_withdraw(auto_bk(f"dist/{kernel}/odf-8.csv"), "efficiency", kernel)

        df = pd.DataFrame({
            'x': data_cycle['x'],
            'Performance_DS-STC': data_mac['64-ds-mac'],
            'Performance_RM-STC': data_mac['64-rm-mac'],
            'Performance_Uni-STC': data_mac['64-smu-mac'],
            'Energy_DS-STC': data_energy['64-ds-energy'],
            'Energy_RM-STC': data_energy['64-rm-energy'],
            'Energy_Uni-STC': data_energy['64-smu-energy'],
            'Efficiency_DS-STC': data_efficiency['64-ds-efficiency'],
            'Efficiency_RM-STC': data_efficiency['64-rm-efficiency'],
            'Efficiency_Uni-STC': data_efficiency['64-smu-efficiency'],
        })

        safe_x_min = df['x'][df['x'] > 0].min()
        min_power = np.floor(np.log2(safe_x_min))
        max_power = np.ceil(np.log2(df['x'].max()))
        
        powers = np.arange(min_power, max_power + 1)
        bins = 2**powers
        
        labels = [f'$2^{{{int(p)}}}$' for p in powers[:-1]]
        df['bin'] = pd.cut(df['x'], bins=bins, labels=labels, right=False, include_lowest=True)

        return df

    spmv_df = _pack_data("SpMV")
    spmspv_df = _pack_data("SpMSpV")
    spmm_df = _pack_data("SpMM")
    spgemm_df = _pack_data("SpGEMM")
    
    kernel_dfs = {"SpMV": spmv_df, "SpMSpV": spmspv_df, "SpMM": spmm_df, "SpGEMM": spgemm_df}

    fig, axes = plt.subplots(2, 4, figsize=(22, 4), sharey='row')
    
    metrics = ['Performance', 'Efficiency']
    y_labels = ['Mac\nUtilisation', 'Energy\nEfficiency']
    y_limits = [(0, 100), None]

    for i, metric in enumerate(metrics):
        for j, (kernel_name, df) in enumerate(kernel_dfs.items()):
            ax = axes[i][j]
            
            df_long = df.melt(
                id_vars=['bin'],
                value_vars=[f'{metric}_DS-STC', f'{metric}_RM-STC', f'{metric}_Uni-STC'],
                var_name='Architecture',
                value_name='Value'
            )
            df_long['Architecture'] = df_long['Architecture'].str.replace(f'{metric}_', '')

            sns.boxplot(
                data=df_long,
                x='bin',
                y='Value',
                hue='Architecture',
                palette=colors,
                ax=ax,
                showfliers=False,
                hue_order=['DS-STC', 'RM-STC', 'Uni-STC']
            )

            ax.get_legend().remove()
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.tick_params(axis='y', labelsize=18)
            
            if i == 1:
                ax.set_xlabel(kernel_name, fontsize=18)
                plt.setp(ax.get_xticklabels(), fontsize=18)
            else:
                ax.set_xlabel('')
                ax.set_xticklabels([])

            if j == 0:
                ax.set_ylabel(y_labels[i], fontsize=18)
            else:
                ax.set_ylabel('')

            ax.set_ylim(y_limits[i])

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.align_ylabels(axes[:, 0])
    fig.legend(handles, labels, loc='upper left', fontsize=18, ncol=3, bbox_to_anchor=(0.05, 1.02))

    plt.tight_layout(rect=[0.02, 0.03, 1, 0.9]) # Adjust layout
    plt.savefig("fig/20.png", dpi=300)
