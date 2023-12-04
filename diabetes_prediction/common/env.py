"""Environment module
"""

# Internal libraries
import os
import sys
from os.path import join, abspath, dirname, exists
import json
import re
import subprocess
from collections import defaultdict
from copy import deepcopy as copy
import warnings

# External libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# from analysis_tools import eda


# Options
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', None)

plt.rc('axes', unicode_minus=False)
plt.style.use('ggplot')

# ignore warnings
warnings.filterwarnings('ignore')
