"""Utility module
"""

from diabetes_prediction.common.config import *
from diabetes_prediction.common.env import *


lmap         = lambda fn, arr: list(map(fn, arr))
inverse_dict = lambda dic: {value: key for key, value in dic.items()}
