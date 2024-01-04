import re

from woodwork.column_schema import ColumnSchema

from featuretools.demo import load_mock_customer
from featuretools.tests.testing_utils import make_ecommerce_entityset
import featuretools as ft
from woodwork.logical_types import Categorical, PostalCode, Age, AgeNullable, NaturalLanguage

import pandas as pd
import numpy as np

from featuretools.primitives import TransformPrimitive


class Separate(TransformPrimitive):
    name = "separate"
    input_types = [
        ColumnSchema(semantic_tags={"numeric"}),
        ColumnSchema(semantic_tags={"numeric"})
    ]
    return_type = ColumnSchema(semantic_tags={"numeric"})

    def get_function(self):
        def separate(column_and_edge):
            if column_and_edge is None or len(column_and_edge) != 2:
                return -998
            elif column_and_edge[0] is None:
                return -999
            else:
                return 0 if column_and_edge[0] < column_and_edge[1] else 1

        return separate


def gen_custom_names(primitive, base_feature_names):
    return [
        "Above18(%s)" % base_feature_names,
        "Above21(%s)" % base_feature_names,
        "Above65(%s)" % base_feature_names,
    ]


class IsGreater(TransformPrimitive):
    name = "is_greater"
    input_types = [
        ColumnSchema(semantic_tags={"numeric"}),
        ColumnSchema(semantic_tags={"numeric"})
    ]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    number_output_features = 1

    def get_function(self):
        def is_greater(v1, v2):
            if v1.dropna().empty or v2.dropna().empty:
                return pd.Series([-997] * len(v2))
            df = pd.DataFrame({
                "value": v1,
                "const": v2
            })
            df['compare'] = df['value'] > df['const']
            df['compare'] = df['compare'].astype(int)

            return df['compare'].values

        return is_greater

    def generate_names(primitive, base_feature_names):
        return "%s > %s" % (base_feature_names[0], base_feature_names[1])


class CaseCount(TransformPrimitive):
    """Return the count of upper case and lower case letters of a text."""

    name = "case_count"
    input_types = [ColumnSchema(logical_type=NaturalLanguage)]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    number_output_features = 2

    def get_function(self):
        def case_count(array):
            # this is a naive implementation used for clarity
            upper = np.array([len(re.findall("[A-Z]", i)) for i in array])
            lower = np.array([len(re.findall("[a-z]", i)) for i in array])
            return upper, lower

        return case_count

    def generate_names(self, base_feature_names):
        name = self.generate_name(base_feature_names)
        return f"{name}[upper]", f"{name}[lower]"


def main_1():
    data = load_mock_customer()
    transactions_df = data["transactions"]
    sessions_df = data["sessions"]
    customers_df = data['customers']
    products_df = data["products"]

    es = ft.EntitySet(id="total")
    es = es.add_dataframe(
        dataframe_name="transactions",
        dataframe=transactions_df,
        index="transaction_id",
        time_index="transaction_time",
        logical_types={
            "product_id": Categorical
        },
    )
    print(es["transactions"].ww)

    es = es.add_dataframe(
        dataframe_name="products",
        dataframe=products_df,
        index="product_id"
    )
    print(es["transactions"].ww)

    es.add_relationship("products", "product_id", "transactions", "product_id")

    es = es.add_dataframe(
        dataframe=sessions_df,
        dataframe_name="sessions",
        index="session_id",
        time_index="session_start"
    )
    es.add_relationship("sessions", "session_id", "transactions", "session_id")

    es = es.add_dataframe(
        dataframe=customers_df,
        dataframe_name="customers",
        index="customer_id",
        time_index="join_date",
        logical_types={
            "zip_code": PostalCode
        },
    )
    es.add_relationship("customers", "customer_id", "sessions", "customer_id")

    feature_product_defs = ft.dfs(
        entityset=es,
        target_dataframe_name="products",
        max_depth=3,
        agg_primitives=['sum', 'std', 'count'],
        features_only=True,
    )
    print(feature_product_defs)


def main_2():
    es = ft.demo.load_mock_customer(return_entityset=True)
    # seed Features
    expensive_purchase = ft.Feature(es["transactions"].ww["amount"]) > 125

    feature_matrix, feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name="customers",
        agg_primitives=["percent_true"],
        seed_features=[expensive_purchase]
    )
    print(feature_defs)
    print(ft.describe_feature(feature_defs[9]))


def main_3():
    es = ft.demo.load_mock_customer(return_entityset=True)

    window_fm, window_features = ft.dfs(
        entityset=es,
        target_dataframe_name="customers",
        instance_ids=[1, 2, 3],
        training_window="2 hour",
    )

    print(window_fm.head())


def main_4():
    es = ft.demo.load_mock_customer(return_entityset=True)
    feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name="transactions",
        max_depth=3,
        agg_primitives=['sum', 'std', 'count'],
        features_only=True,
    )

    print(feature_defs)

    print("point")


def main_5(threshold):
    """
    单表transactions
    构造一点点简单的扣字段变量
    如：本次交易的金额是否大于100，是则输出1，否则输出0，为空输出-999
    需要自定义一个方法，还算比较通用
    Returns:

    """
    raw_data = ft.demo.load_mock_customer()
    data = raw_data['transactions']
    data['const_v'] = threshold
    es = ft.EntitySet(id="my_es")
    es = es.add_dataframe(
        dataframe_name="transactions",
        dataframe=data,
        index="transaction_id",
        time_index="transaction_time",
        # logical_types={
        #     "product_id": Categorical,
        #     "zip_code": PostalCode,
        # },
    )

    """
    primitive_option 见名知意，是对primitive的配置
    是一个一个算子的配置，每个配置下辖的列数应当与其类定义一致
    如下，就定义了要对transactions下的amount和count_v构造一个isGreater是否大于的特征
    和transactions下amount和const_v的相乘的特征
    """
    options = {
        IsGreater: [
            {"include_columns": {"transactions": ["amount"]}},
            {"include_columns": {"transactions": ["const_v"]}}
        ],
        "multiply_numeric": [
            {"include_columns": {"transactions": ["amount"]}},
            {"include_columns": {"transactions": ["const_v"]}}
        ],
    }

    # matrix, feats = ft.dfs(
    feats = ft.dfs(
        entityset=es,
        target_dataframe_name="transactions",
        max_depth=1,
        agg_primitives={},
        trans_primitives=[IsGreater],
        # trans_primitives=["multiply_numeric"],
        primitive_options=options,
        features_only=True,
    )

    # print(matrix.head(10))
    print(feats)


def main_6():
    es = make_ecommerce_entityset()
    features_def = ft.dfs(
        entityset=es,
        target_dataframe_name="stores",
        agg_primitives=["sum"],
        trans_primitives=[CaseCount],
        max_depth=4,
        features_only=True,
    )
    print(features_def)


def main_self_join_by_duplicate_df():
    """
    Error with non-unique OR can not related with the same two columns
    parent's relation col must be index col, and change index col to busi col its not unique
    anyway to resolve it (in java service design)?
    Returns:

    """
    data = ft.demo.load_mock_customer()
    sessions1 = data['sessions'].copy()
    sessions2 = data['sessions'].copy()

    dataframes = {
        "sessions1": (sessions1, "customer_id"),
        "sessions2": (sessions2, "session_id")
    }

    relationships = {
        ("sessions1", "customer_id", "sessions2", "customer_id")
    }

    features_defs = ft.dfs(
        dataframes=dataframes,
        relationships=relationships,
        target_dataframe_name="sessions1",
        features_only=True
    )
    print(features_defs)


def main_7():
    data = load_mock_customer()
    transactions_df = data["transactions"]
    sessions_df = data["sessions"]
    customers_df = data['customers']
    products_df = data["products"]

    es = ft.EntitySet(id="total")
    es = es.add_dataframe(
        dataframe_name="transactions",
        dataframe=transactions_df,
        index="transaction_id",
        time_index="transaction_time",
        logical_types={
            "product_id": Categorical
        },
    )
    print(es["transactions"].ww)

    es = es.add_dataframe(
        dataframe_name="products",
        dataframe=products_df,
        index="product_id"
    )
    print(es["transactions"].ww)

    es.add_relationship("products", "product_id", "transactions", "product_id")

    es = es.add_dataframe(
        dataframe=sessions_df,
        dataframe_name="sessions",
        index="session_id",
        time_index="session_start"
    )
    es.add_relationship("sessions", "session_id", "transactions", "session_id")

    es = es.add_dataframe(
        dataframe=customers_df,
        dataframe_name="customers",
        index="customer_id",
        time_index="join_date",
        logical_types={
            "zip_code": PostalCode
        },
        semantic_tags={
            "birthday": "date_of_birth"
        }
    )
    es.add_relationship("customers", "customer_id", "sessions", "customer_id")

    feature_product_defs = ft.dfs(
        entityset=es,
        target_dataframe_name="sessions",
        max_depth=3,
        trans_primitives=['age'],
        agg_primitives=['sum', 'count'],
        features_only=True,

    )
    print(feature_product_defs)


def main_main():
    es = ft.demo.load_mock_customer(return_entityset=True)
    values_dict = {"device": ["desktop", "mobile", "tablet"]}
    es.add_interesting_values(dataframe_name="sessions", values=values_dict)
    es.add_interesting_values(dataframe_name="sessions",
                              values={"customer_id": es["sessions"]["customer_id"].unique().tolist()})
    es.add_interesting_values(dataframe_name="transactions",
                              values={"product_id": es["transactions"]["product_id"].unique().tolist()})
    es.add_interesting_values(dataframe_name="products", values={"brand": es["products"]["brand"].unique().tolist()})

    feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name="customers",
        max_depth=5,
        trans_primitives=['not_equal', 'day'],
        agg_primitives=['sum', 'mode', 'count'],
        features_only=True,
    )

    print(feature_defs)

    print("point")


if __name__ == "__main__":
    main_main()
