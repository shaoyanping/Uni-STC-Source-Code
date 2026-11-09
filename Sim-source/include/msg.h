#pragma once

#ifdef __cplusplus
extern "C"
{
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#ifdef DEBUG
    char is_debug = 1;
#else
char is_debug = 1;
#endif // DEBUG

    const char *msg[] = {
        "[bold cyan][提示][/]",
        "[bold red][错误][/]",
        "[bold yellow][调试][/]",
        "[bold green][成功][/]",
        "[bold yellow][警告][/]",
        "__TITLE__",
        "__RULE__",
        "__MARKDOWN__",
        "__START__",
        "__STOP__"};

    typedef enum msg_type
    {
        info,
        error,
        debug,
        success,
        warning,
        title,
        rule,
        markdown,
        start_status,
        stop_status
    } msg_type;

    void echo(msg_type type, const char *format, ...)
    {
        if (type == debug && !is_debug)
        {
            return;
        }

        printf("%s ", msg[type]);
        va_list args;
        va_start(args, format);
        vprintf(format, args);
        va_end(args);
        printf("\n");
        fflush(stdout);
    }

#ifdef __cplusplus
}
#endif