import pstats

# https://docs.python.org/2/library/profile.html#pstats.Stats.sort_stats
def profile(f='last_replay_sarsa_space_invaders/profile.txt'):
    p = pstats.Stats(f)
    p.strip_dirs()
    p.sort_stats('module', 'tottime', 'cumtime').reverse_order().print_stats()
    return p
