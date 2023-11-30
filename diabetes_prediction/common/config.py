"""Configuration module
"""

from diabetes_prediction.common.env import *


class PATH:
    root   = abspath(dirname(dirname(dirname(__file__))))
    data   = join(root, 'data')
    script = join(root, 'script')

    family        = join(data, "familyxx", "familyxx")
    household     = join(data, "househld", "househld")
    person        = join(data, "personsx", "personsx")
    sample_child  = join(data, "samchild", "samchild")
    sample_adult  = join(data, "samadult", "samadult")

    @classmethod
    def get(cls, data_id):
        rst = {}
        if data_id not in ('family', 'sample_child', 'sample_adult', 'household', 'person'):
            raise ValueError(f"Invalid data_id: {data_id}")

        src = getattr(cls, f"{data_id}")
        rst['summary']    = f"{src}_summary.pdf"
        rst['layout']     = f"{src}_layout.pdf"
        rst['metadata']   = f"{src}_metadata.csv"
        rst['data']       = f"{src}.csv"
        rst['final_data'] = f"{src}_final.ftr"
        return rst


class PARAMS:
    figsize = (28, 4)
    seed    = 42
    target  = 'DIBEV1'
