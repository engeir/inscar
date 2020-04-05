import cProfile
import pstats

import main


profile = cProfile.Profile()
profile.runcall(main.plot_IS_spectrum)
ps = pstats.Stats(profile)
ps.print_stats()
