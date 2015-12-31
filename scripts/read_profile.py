import pstats

def last_profile():
    p = pstats.Stats('last_replay_sarsa_space_invaders/profile.txt')
    p.strip_dirs()
    p.sort_stats('module', 'tottime', 'cumtime').reverse_order().print_stats()
    return p
