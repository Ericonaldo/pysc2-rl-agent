# (D)RL Agent For PySC2 Environment

A fork from https://github.com/inoryy/pysc2-rl-agent

Add some notes in Chinese

在看的过程中加入了一些中文注释

修改了一些BUG，在runner中若使用多个进程，其中一个进程done=1后若其他进程还未结束，则该进程会重新开始游戏及计分，reward将会累加
