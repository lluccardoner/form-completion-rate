from pathlib import Path

from pydantic import BaseModel
from pyspark.sql import DataFrame

ROOT_DIR: Path = Path(__file__).parent.parent
OUTPUT_DIR: Path = ROOT_DIR.joinpath("output")
RESOURCES_DIR: Path = ROOT_DIR.joinpath("resources")
MODEL_DIR: Path = OUTPUT_DIR.joinpath("model")


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
    feat_01: int
    feat_02: int
    feat_03: int
    feat_04: int
    feat_05: int
    feat_06: int
    feat_07: int
    feat_08: int
    feat_09: int
    feat_10: int
    feat_11: int
    feat_12: int
    feat_13: int
    feat_14: int
    feat_15: int
    feat_16: int
    feat_17: int
    feat_18: int
    feat_19: int
    feat_20: int
    feat_21: int
    feat_22: int
    feat_23: int
    feat_24: int
    feat_25: int
    feat_26: int
    feat_27: int
    feat_28: int
    feat_29: int
    feat_30: int
    feat_31: int
    feat_32: int
    feat_33: int
    feat_34: int
    feat_35: int
    feat_36: int
    feat_37: int
    feat_38: int
    feat_39: int
    feat_40: int
    feat_41: int
    feat_42: int
    feat_43: int
    feat_44: int
    feat_45: int
    feat_46: int
    feat_47: int
