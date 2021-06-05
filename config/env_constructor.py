from rlberry.envs.finite import GridWorld

def constructor(nrows, ncols, success_probability):
    env = GridWorld(nrows=nrows, ncols=ncols, walls=(), success_probability=success_probability)
    return env