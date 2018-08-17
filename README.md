# (D)RL Agent For PySC2 Environment

A fork from https://github.com/inoryy/pysc2-rl-agent

Add some notes in Chinese

在看的过程中加入了一些中文注释

修改了一些BUG，在runner中若使用多个进程，其中一个进程done=1后若其他进程还未结束，则该进程会重新开始游戏及计分，reward将会累加，修改了reward的累计方式

增加了模仿学习，使用脚本生成录像，然后使用录像初始化模型，再使用强化学习，加速收敛。

## 模仿学习示例，详细参数见文件：

**使用脚本生成录像replay**

python replay_script_agent.py --map=CollectMineralShards --sz=16 --DATA_SIZE=50000

**监督训练**
 
python imitation_learning.py --map=CollectMineralShards --sz=16

**强化学习，从预训练好的Model中读取**

python main.py --map=CollectMineralShards --sz=16 --env=1 --updates=50000 --restore=True --steps=12

## 强化学习A2C训练

python main.py --map=CollectMineralShards --sz=16 --env=1 --updates=50000 --steps=12 --restrict=True

restrict限制输出动作为：

_NOOP = actions.FUNCTIONS.no_op.id

_SELECT_POINT = actions.FUNCTIONS.select_point.id

_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id

此举是为了限制动作空间，在第二个minimap上学出agent分开，后续加入multi-step
