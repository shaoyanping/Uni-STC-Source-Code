from QuickProject import QproErrorString
from QuickProject.Commander import Commander
from . import *


app = Commander(name)


@app.command()
def compile():
    """
    🔨 编译
    """
    import shutil
    st, _ = external_exec('./compile.sh', without_output=True)
    if st:
        QproDefaultConsole.print(QproErrorString, "Compile ResNet50 (dense) Failed!")
        return
    else:
        QproDefaultConsole.print(QproInfoString, "Compile ResNet50 (dense) Succeeded!")
    # move resnet to dist/
    shutil.move("resnet", f"dist/{name}")


@app.command()
def brun(pic: str):
    QproDefaultConsole.print(QproInfoString, f"Benchmark ResNet50 (dense)")
    csv = "resnet_benchmark.csv"
    if os.path.exists(csv):
        return
    p = external_exec(f"./dist/{name} {pic}", __no_wait=True)
    import pandas as pd

    odf = pd.DataFrame(
        columns=[
            'Layer',
            'mp16',
            'mpnz'
        ] + pd_col
    )

    layer, df = None, None

    line_gen = iter(p.stdout.readline, '')
    error_gen = iter(p.stderr.readline, '')

    for line in line_gen:
        line = line.strip()
        # QproDefaultConsole.print(QproInfoString, line)
        if line.startswith('[bold cyan][提示][/] Layer'):
            if layer:
                add_line = [layer] + df.strip().split(',')
                # QproDefaultConsole.print(QproInfoString, f"Layer {layer} Add:", add_line)
                odf.loc[odf.shape[0]] = add_line
                odf.to_csv(csv, index=False)
                # QproDefaultConsole.print(QproInfoString, f"Layer {layer} Added.")
                # send_text(f"Layer {layer} Added.")
            layer = line.split(' ')[-1]
        df = line
    
    st = p.wait()
    p.stdout.close()
    p.stderr.close()


def _update(df, drop: bool = False):
    df['128-smu-mac'] = df['mpnz'] / (df['128-smu-cycle'] * 128) * 100
    df['128-rm-mac'] = df['mpnz'] / (df['128-rm-cycle'] * 128) * 100
    df['128-ds-mac'] = df['mpnz'] / (df['128-ds-cycle'] * 128) * 100



@app.command()
def fmt_brun():
    import pandas as pd

    raw_df = pd.read_csv("resnet_benchmark.csv")
    df = pd.read_csv("resnet_benchmark.csv")
    # df.drop(columns=['128-ds-mac', '128-rm-mac', '128-smu-mac', '64-ds-mac', '64-rm-mac', '64-smu-mac', '128-smu-speedup', '64-smu-speedup'], inplace=True)
    # 倒序遍历，当前行减去上一行
    for i in range(df.shape[0] - 1, 0, -1):
        df.loc[i] = df.loc[i] - df.loc[i - 1]
    
    ds128_energy = df[f"128-ds-a-ntr"] * 0.0625 + df[f"128-ds-b-ntr"] * 0.0625 + df[f"128-ds-c-ntr"] * 1.4
    rm128_energy = df[f"128-rm-a-ntr"] * 0.0625 + df[f"128-rm-b-ntr"] * 0.35 + df[f"128-rm-c-ntr"] * 0.7
    smu128_energy = df[f"128-smu-a-nut"] * 0.6875 + df[f"128-smu-b-nut"] * 0.6875 + df[f"128-smu-c-nut"] * 0.6875
    df["128-ds-energy"] = ds128_energy
    df["128-rm-energy"] = rm128_energy
    df["128-smu-energy"] = smu128_energy
    df['Layer'] = raw_df['Layer']
    _update(df)

    cols = 'Layer,mpnz,128-ds-cycle,128-rm-cycle,128-smu-cycle,128-ds-mac,128-rm-mac,128-smu-mac,128-ds-energy,128-rm-energy,128-smu-energy'.split(',')
    odf = pd.DataFrame(columns=cols)
    for i in cols:
        odf[i] = df[i]
    odf.to_csv("resnet_benchmark_fmt.csv", index=False)
    # df.to_csv("resnet_benchmark_fmt.csv", index=False)


@app.command()
def ss():
    import pandas as pd
    # sum layer == 0
    df = pd.read_csv("resnet_benchmark_fmt.csv")
    odf = None
    for layer in range(5):
        tdf = df.loc[df['Layer'] == layer].sum()
        tdf = tdf.to_frame().T
        tdf['Layer'] = layer
        _update(tdf, True)
        if odf is None:
            odf = tdf.copy()
        else:
            odf = pd.concat([odf, tdf], axis=0)

    cols = 'Layer,128-ds-mac,128-rm-mac,128-smu-mac,128-ds-energy,128-rm-energy,128-smu-energy'.split(',')
    ndf = pd.DataFrame(columns=cols)
    for i in cols:
        ndf[i] = odf[i]
    ndf['Layer'] = ndf['Layer'].apply(lambda x: f'ResNet50-{x}')
    ndf.rename(columns={'Layer': 'mtx'}, inplace=True)
    ndf.to_csv("resnet.csv", index=False)


@app.command()
def flow():
    # app.real_call('compile')
    app.real_call('brun', '../pics/zhongguolong.jpg')
    app.real_call('fmt_brun')
    app.real_call('ss')


def main():
    """
    * 注册为全局命令时, 默认采用main函数作为命令入口, 请勿将此函数用作它途.
    * When registering as a global command, default to main function as the command entry, do not use it as another way.
    """
    app()


if __name__ == "__main__":
    main()
