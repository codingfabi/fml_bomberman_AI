import numpy as np
import settings as s


initialMockState = {
    'round': 0,
    'step': 0,
    'field': np.zeros((s.COLS,s.ROWS)),
    'bombs': [((0,0),0)],
    'explosion_map': np.zeros((s.COLS, s.ROWS)),
    'coins': [(0,0)],
    'self': ('BombyMcBombface', 0, True, (0,0)),
    'others':[('EnemyAgent1', 0, True, (20,20)),('EnemyAgent1', 0, True, (20,20))]}
