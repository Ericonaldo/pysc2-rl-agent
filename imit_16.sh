python replay_script_agent.py --map=CollectMineralShards --sz=16 --DATA_SIZE=500
python imitation_learning.py --map=CollectMineralShards --sz=16 --epochs=100
python main.py --map=CollectMineralShards --sz=16 --env=1 --updates=200000 --restore=True --steps=12 --imitation=True
