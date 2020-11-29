from pathlib import Path

from pydantic import BaseModel
from pyspark.sql import DataFrame

ROOT_DIR: Path = Path(__file__).parent.parent
DEPLOY_DIR: Path = ROOT_DIR / "deploy"
OUTPUT_DIR: Path = ROOT_DIR / "output"
RESOURCES_DIR: Path = ROOT_DIR / "resources"
MODEL_DIR: Path = OUTPUT_DIR / "model"


def are_dfs_equal(df1: DataFrame, df2: DataFrame) -> bool:
    # This method can be improved or a third party library could be used
    s1 = df1.schema.fieldNames()
    s2 = df2.schema.fieldNames()
    c1 = df1.collect()
    c2 = df2.collect()
    if s1 != s2:
        print(s1)
        print(s2)
        return False
    if c1 != c2:
        print(c1)
        print(c2)
        return False
    return True


class Sample(BaseModel):
    feat_01: float
    feat_02: float
    feat_03: float
    feat_04: float
    feat_05: float
    feat_06: float
    feat_07: float
    feat_08: float
    feat_09: float
    feat_10: float
    feat_11: float
    feat_12: float
    feat_13: float
    feat_14: float
    feat_15: float
    feat_16: float
    feat_17: float
    feat_18: float
    feat_19: float
    feat_20: float
    feat_21: float
    feat_22: float
    feat_23: float
    feat_24: float
    feat_25: float
    feat_26: float
    feat_27: float
    feat_28: float
    feat_29: float
    feat_30: float
    feat_31: float
    feat_32: float
    feat_33: float
    feat_34: float
    feat_35: float
    feat_36: float
    feat_37: float
    feat_38: float
    feat_39: float
    feat_40: float
    feat_41: float
    feat_42: float
    feat_43: float
    feat_44: float
    feat_45: float
    feat_46: float
    feat_47: float
