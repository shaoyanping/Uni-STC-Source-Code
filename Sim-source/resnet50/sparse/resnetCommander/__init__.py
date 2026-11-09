# -*- coding: utf-8 -*-

name = "resnet"

from .__config__ import *

config: resnetCommanderConfig = None
if enable_config:
    config = resnetCommanderConfig()

import sys
from QuickProject import user_pip, _ask, external_exec, get_config


cc = "g++"
optimization = "fast"  # [fast, 0, 1, 2, 3, s, g]
standard = "c++17"  # [c++98, c++03, c++11, c++14, c++17, c++2a]

includePath = [
    "include",
]

sourcePath = [get_config()["entry_point"]]

libPath = []

usingLib = [
    "m",
]

other_flags = []


def requirePackage(
    pname: str,
    module: str = "",
    real_name: str = "",
    not_exit: bool = True,
    not_ask: bool = False,
    set_pip: str = user_pip,
):
    """
    获取本机上的python第三方库，如没有则询问安装

    :param not_ask: 不询问，无依赖项则报错
    :param set_pip: 设置pip路径
    :param pname: 库名
    :param module: 待引入的模块名，可缺省
    :param real_name: 用于 pip3 install 的名字
    :param not_exit: 安装后不退出
    :return: 库或模块的地址
    """
    try:
        exec(f"from {pname} import {module}" if module else f"import {pname}")
    except (ModuleNotFoundError, ImportError):
        if not_ask:
            return None
        if _ask(
            {
                "type": "confirm",
                "message": f"""{name} require {pname + (' -> ' + module if module else '')}, confirm to install?
  {name} 依赖 {pname + (' -> ' + module if module else '')}, 是否确认安装?""",
                "default": True,
            }
        ):
            with QproDefaultConsole.status(
                "Installing..." if user_lang != "zh" else "正在安装..."
            ):
                external_exec(
                    f"{set_pip} install {pname if not real_name else real_name} -U",
                    True,
                )
            if not_exit:
                exec(f"from {pname} import {module}" if module else f"import {pname}")
            else:
                QproDefaultConsole.print(
                    QproInfoString,
                    f'just run again: "{" ".join(sys.argv)}"'
                    if user_lang != "zh"
                    else f'请重新运行: "{" ".join(sys.argv)}"',
                )
                exit(0)
        else:
            exit(-1)
    finally:
        return eval(f"{module if module else pname}")


pd_col = [
    "128-ds-cycle",
    "128-ds-mac",
    "128-rm-cycle",
    "128-rm-mac",
    "128-smu-cycle",
    "128-smu-mac",
    "128-ds-energy",
    "128-rm-energy",
    "128-smu-energy",
    "128-smu-speedup",
    "64-ds-cycle",
    "64-ds-mac",
    "64-rm-cycle",
    "64-rm-mac",
    "64-smu-cycle",
    "64-smu-mac",
    "64-ds-energy",
    "64-rm-energy",
    "64-smu-energy",
    "64-smu-speedup",
    "128-ds-mt125",
    "128-ds-mt25",
    "128-ds-mt50",
    "128-ds-mt75",
    "128-ds-mt100",
    "128-rm-mt125",
    "128-rm-mt25",
    "128-rm-mt50",
    "128-rm-mt75",
    "128-rm-mt100",
    "128-smu-mt125",
    "128-smu-mt25",
    "128-smu-mt50",
    "128-smu-mt75",
    "128-smu-mt100",
    "64-ds-mt125",
    "64-ds-mt25",
    "64-ds-mt50",
    "64-ds-mt75",
    "64-ds-mt100",
    "64-rm-mt125",
    "64-rm-mt25",
    "64-rm-mt50",
    "64-rm-mt75",
    "64-rm-mt100",
    "64-smu-mt125",
    "64-smu-mt25",
    "64-smu-mt50",
    "64-smu-mt75",
    "64-smu-mt100",
    "128-ds-a-ntr",
    "128-ds-b-ntr",
    "128-ds-c-ntr",
    "128-rm-a-ntr",
    "128-rm-b-ntr",
    "128-rm-c-ntr",
    "128-smu-a-ntr",
    "128-smu-b-ntr",
    "128-smu-c-ntr",
    "64-ds-a-ntr",
    "64-ds-b-ntr",
    "64-ds-c-ntr",
    "64-rm-a-ntr",
    "64-rm-b-ntr",
    "64-rm-c-ntr",
    "64-smu-a-ntr",
    "64-smu-b-ntr",
    "64-smu-c-ntr",
    '128-ds-nt125',
    '128-ds-nt25',
    '128-ds-nt50',
    '128-ds-nt75',
    '128-ds-nt100',
    '128-rm-nt125',
    '128-rm-nt25',
    '128-rm-nt50',
    '128-rm-nt75',
    '128-rm-nt100',
    '128-smu-nt125',
    '128-smu-nt25',
    '128-smu-nt50',
    '128-smu-nt75',
    '128-smu-nt100',
    '64-ds-nt125',
    '64-ds-nt25',
    '64-ds-nt50',
    '64-ds-nt75',
    '64-ds-nt100',
    '64-rm-nt125',
    '64-rm-nt25',
    '64-rm-nt50',
    '64-rm-nt75',
    '64-rm-nt100',
    '64-smu-nt125',
    '64-smu-nt25',
    '64-smu-nt50',
    '64-smu-nt75',
    '64-smu-nt100',
    '128-rm-c-nut',
    '128-smu-a-nut',
    '128-smu-b-nut',
    '128-smu-c-nut',
    '64-rm-c-nut',
    '64-smu-a-nut',
    '64-smu-b-nut',
    '64-smu-c-nut',
]