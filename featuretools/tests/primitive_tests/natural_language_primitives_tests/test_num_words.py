import numpy as np
import pandas as pd

from featuretools.primitives import NumWords
from featuretools.tests.primitive_tests.utils import (
    PrimitiveTestBase,
    find_applicable_primitives,
    valid_dfs,
)


class TestNumWords(PrimitiveTestBase):
    primitive = NumWords

    def test_general(self):
        x = pd.Series(
            [
                "test test test test",
                "test TEST test TEST,test test test",
                "and subsequent lines...",
            ],
        )
        expected = pd.Series([4, 6, 3])
        actual = self.primitive().get_function()(x)
        pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_special_characters_and_whitespace(self):
        x = pd.Series(["50% 50 50% \t\t\t\n\n", "$5,3040 a test* test"])
        expected = pd.Series([3, 4])
        actual = self.primitive().get_function()(x)
        pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_unicode_input(self):
        x = pd.Series(
            [
                "Ángel Angel Ángel ángel",
            ],
        )
        expected = pd.Series([4])
        actual = self.primitive().get_function()(x)
        pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_contractions(self):
        x = pd.Series(
            [
                "can't won't don't can't aren't won't don't they'd there's",
            ],
        )
        expected = pd.Series([9])
        actual = self.primitive().get_function()(x)
        pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_multiple_spaces(self):
        x = pd.Series(
            [
                "    word  word            word word     .",
                "This is                      \nthird line \nthird line",
            ],
        )
        expected = pd.Series([4, 6])
        actual = self.primitive().get_function()(x)
        pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_null(self):
        x = pd.Series([np.nan, pd.NA, None, "This is a test file."])
        actual = self.primitive().get_function()(x)
        expected = pd.Series([pd.NA, pd.NA, pd.NA, 5])
        pd.testing.assert_series_equal(
            actual,
            expected,
            check_names=False,
            check_dtype=False,
        )

    def test_with_featuretools(self, es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        transform.append(primitive_instance)
        valid_dfs(es, aggregation, transform, self.primitive)
