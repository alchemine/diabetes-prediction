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


# External libraries
import numpy as np
import pandas as pd
from analysis_tools import eda

import matplotlib.pyplot as plt
import seaborn as sns


# Options
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_colwidth', None)

plt.rc('axes', unicode_minus=False)
plt.style.use('ggplot')
