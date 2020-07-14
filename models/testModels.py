from egaARModel import EGAAR
from egaTDModel import EGATD
from egaARTDModel import EGAARTD
from egaTDRandModel import EGATDRand

# GA = EGAAR(iterations=500, pop_size=100, mut_prob=0.1, elite_ratio=0.01, cross_prob=0.5, par_ratio=0.3, ar_N=2)
# GA.run()
# GA = EGAAR(iterations=500, pop_size=100, mut_prob=0.3, elite_ratio=0.01, cross_prob=0.5, par_ratio=0.3, ar_N=2)
# GA.run()
# GA = EGAAR(iterations=500, pop_size=100, mut_prob=0.7, elite_ratio=0.01, cross_prob=0.5, par_ratio=0.3, ar_N=2)
# GA.run()

# GA = EGATD(iterations=500, pop_size=100, mut_prob=0.1, elite_ratio=0.01, cross_prob=0.5, par_ratio=0.3, td_N=2)
# GA.run()
# GA = EGATD(iterations=500, pop_size=100, mut_prob=0.3, elite_ratio=0.01, cross_prob=0.5, par_ratio=0.3, td_N=2)
# GA.run()
# GA = EGATD(iterations=500, pop_size=100, mut_prob=0.7, elite_ratio=0.01, cross_prob=0.5, par_ratio=0.3, td_N=2)
# GA.run()

# GA = EGAARTD(iterations=500, pop_size=100, mut_prob=0.1, elite_ratio=0.01, cross_prob=0.5, par_ratio=0.3, ar_N=1, td_N=1)
# GA.run()
# GA = EGAARTD(iterations=500, pop_size=100, mut_prob=0.3, elite_ratio=0.01, cross_prob=0.5, par_ratio=0.3, ar_N=1, td_N=1)
# GA.run()
# GA = EGAARTD(iterations=500, pop_size=100, mut_prob=0.7, elite_ratio=0.01, cross_prob=0.5, par_ratio=0.3, ar_N=1, td_N=1)
# GA.run()

# GA = EGATD(iterations=500, pop_size=100, mut_prob=0.1, elite_ratio=0.01, cross_prob=0.5, par_ratio=0.3, td_N=3)
# GA.run()
# GA = EGATD(iterations=500, pop_size=100, mut_prob=0.3, elite_ratio=0.01, cross_prob=0.5, par_ratio=0.3, td_N=3)
# GA.run()
# GA = EGATD(iterations=500, pop_size=100, mut_prob=0.7, elite_ratio=0.01, cross_prob=0.5, par_ratio=0.3, td_N=3)
# GA.run()

GA = EGATDRand(iterations=500, pop_size=100, mut_prob=0.1, elite_ratio=0.01, cross_prob=0.5, par_ratio=0.3, td_N=2, rand_N=3)
GA.run()
GA = EGATDRand(iterations=500, pop_size=100, mut_prob=0.3, elite_ratio=0.01, cross_prob=0.5, par_ratio=0.3, td_N=2, rand_N=3)
GA.run()
GA = EGATDRand(iterations=500, pop_size=100, mut_prob=0.7, elite_ratio=0.01, cross_prob=0.5, par_ratio=0.3, td_N=2, rand_N=3)
GA.run()
