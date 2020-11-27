from pyspark.sql import DataFrame


def are_dfs_equal(df1: DataFrame, df2: DataFrame) -> bool:
    if df1.schema != df2.schema:
        return False
    if df1.collect() != df2.collect():
        return False
    return True
