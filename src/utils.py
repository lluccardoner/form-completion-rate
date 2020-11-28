from pyspark.sql import DataFrame


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