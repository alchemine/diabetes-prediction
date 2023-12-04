"""Configuration module
"""

from diabetes_prediction.common.env import *


class PATH:
    root   = abspath(dirname(dirname(dirname(__file__))))
    script = join(root, 'script')
    data   = join(root, 'data')
    proc   = join(data, "processed")

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
        return rst

    @classmethod
    def get_proc(cls):
        return {
            'data': join(cls.proc, "data.csv"),
            'metadata': join(cls.proc, "metadata.csv"),
            'train': join(cls.proc, "train.ftr"),
            'val': join(cls.proc, "val.ftr"),
            'test': join(cls.proc, "test.ftr")
        }


class PARAMS:
    figsize = (28, 4)
    seed    = 42
    target  = 'DIBEV1'
