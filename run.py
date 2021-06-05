"""
To run the experiment:
$ python run.py config.yaml

To see more options:
$ python run.py config.yaml -h
"""

from rlberry.experiment import experiment_generator
from rlberry.stats.multiple_stats import MultipleStats

mstats = MultipleStats()

for agent_stats in experiment_generator():
    print(agent_stats.agent_class)
    print(agent_stats.init_kwargs)
    mstats.append(agent_stats)

mstats.run()
mstats.save()