import time
import queue
from . import external_exec, mtx_dir_path, QproDefaultConsole, QproInfoString

task_q = queue.Queue()
info_q = queue.Queue()
erro_q = queue.Queue()


def workload(task, mtx, snum, process=None):
    st, rt = external_exec(f'./dist/Sim {task} "{mtx_dir_path}/{mtx}"', without_output=True)
    if not st:
        rt = rt.splitlines()[-1]
        info_q.put([mtx] + rt.strip().split(','))
    elif st != 1:
        erro_q.put([mtx, rt])
    if process:
        process[0].advance(process[1])


def spgemm_2(mtx, process=None):
    st, rt = external_exec(f'./dist/Sim spgemm2 "{mtx[0]}" "{mtx[1]}"', without_output=True)
    if not st:
        rt = rt.splitlines()[-1]
        info_q.put(rt.strip().split(','))
    elif st != 1:
        erro_q.put([mtx[0] + '-' + mtx[1], rt])
    if process:
        process[0].advance(process[1])


def info_consumer(odf, csv):
    while True:
        info = info_q.get()
        if not info:
            break
        odf.loc[odf.shape[0]] = [odf.shape[0]] + info
        odf.to_csv(csv, index=False)

def erro_consumer(task, scheduler):
    while True:
        erro = erro_q.get()
        if not erro:
            break
        with open(f'dist/odf/{task}/err-{scheduler}.log', 'a') as f:
            f.write(f'{erro[0]}\n{erro[1]}\n\n')

def consumer(task, snum, ls):
    while ls:
        mtx = ls.pop(0)
        workload(task, mtx, snum)
    info_q.put(None)
    erro_q.put(None)


def start_framework(task, ls, odf, csv, scheduler: int = 8, port: int = 8000, is_all: bool = False):
    from concurrent.futures import ThreadPoolExecutor, wait

    max_worker = {
        'spmv': 22,
        'spgemm': 9,
        'spgemm2': 3,
        'spmm': 22,
        'spmspv': 22,
        'transformer': 16,
        'format': 22,
        'tile': 22,
    }
    
    executor = ThreadPoolExecutor(max_workers=(max_worker[task]) + (0 if is_all else 1))
    thread_manage = []
    init_thread_manage = []
    
    try:
        init_thread_manage.append(executor.submit(info_consumer, odf, csv))
        init_thread_manage.append(executor.submit(erro_consumer, task, scheduler))
        from QuickStart_Rhy.TuiTools.Bar import NormalProgressBar

        process, task_id = NormalProgressBar(task, len(ls))
        process.start()
        while ls:
            mtx = ls.pop(0)
            if task != 'spgemm2':
                thread_manage.append(executor.submit(workload, task, mtx, scheduler, process=(process, task_id)))
            else:
                thread_manage.append(executor.submit(spgemm_2, mtx, process=(process, task_id)))

        wait(thread_manage)
        info_q.put(None)
        erro_q.put(None)
        if process:
            process.stop()
        wait(init_thread_manage)
    except KeyboardInterrupt:
        for t in thread_manage:
            t.cancel()
        wait(thread_manage)
        if process:
            process.stop()
        info_q.put(None)
        erro_q.put(None)
        wait(init_thread_manage)
        QproDefaultConsole.print(QproInfoString, "Test interrupted, use backuped results to draw figures.")
