python replay_script_agent.py --map=CollectMineralShards --sz=16 --DATA_SIZE=100
python imitation_learning.py --map=CollectMineralShards --sz=16
python main.py --map=CollectMineralShards --sz=16 --env=1 --updates=200000 --restore=True --steps=12
