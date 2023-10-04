import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', size=15)
rc('text', usetex=True)
rc('text.latex', preamble=[r'\usepackage{amsmath}', r'\usepackage{amsfonts}'])



color_choice = {'maml': 'cyan', 'tldr': 'red', 'coda': 'indigo', 'anil': 'blue', 'analytic':'slategrey'}
title_choice = {'maml': r'MAML', 'tldr': r'CAMEL', 'anil': 'ANIL', 'coda': 'CoDA'}
