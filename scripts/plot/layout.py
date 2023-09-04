import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', size=15)
rc('text', usetex=True)
rc('text.latex', preamble=[r'\usepackage{amsmath}', r'\usepackage{amsfonts}'])



color_choice = {'maml': 'blue', 'tldr': 'red', 'coda': 'green', 'anil': 'blue'}
title_choice = {'maml': r'MAML', 'tldr': r'TLDR', 'anil': 'ANIL', 'coda': 'CoDA'}
