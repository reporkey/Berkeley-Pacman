import numpy as np
rewards = np.ones((4, 4), dtype=None)
poslist = [index for index, x in np.ndenumerate(rewards) if x == 0]