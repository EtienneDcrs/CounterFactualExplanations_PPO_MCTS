import pstats
from pstats import SortKey

# Load the stats file
p = pstats.Stats('output.prof')

# Sort by cumulative time and print top 20 functions
p.sort_stats(SortKey.CUMULATIVE).print_stats(20)

# Sort by total time
p.sort_stats(SortKey.TIME).print_stats(40)