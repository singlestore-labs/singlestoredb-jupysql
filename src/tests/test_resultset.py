from unittest.mock import Mock, call

from sqlalchemy import create_engine
from sql.connection import Connection
from pathlib import Path
from unittest.mock import Mock

import pytest
import pandas as pd
import polars as pl
import sqlalchemy

from sql.run import ResultSet
from sql import run as run_module

import re


# @pytest.fixture
# def config():
#     config = Mock()
#     config.displaylimit = 5
#     config.autolimit = 100
#     return config


# @pytest.fixture
# def result():
#     df = pd.DataFrame({"x": range(3)})  # noqa
#     engine = sqlalchemy.create_engine("duckdb://")

#     conn = engine.connect()
#     result = conn.execute(sqlalchemy.text("select * from df"))
#     yield result
#     conn.close()


# @pytest.fixture
# def result_set(result, config):
#     return ResultSet(result, config).fetch_results()


# def test_resultset_getitem(result_set):
#     assert result_set[0] == (0,)
#     assert result_set[0:2] == [(0,), (1,)]


# def test_resultset_dict(result_set):
#     assert result_set.dict() == {"x": (0, 1, 2)}


# def test_resultset_dicts(result_set):
#     assert list(result_set.dicts()) == [{"x": 0}, {"x": 1}, {"x": 2}]


# def test_resultset_dataframe(result_set, monkeypatch):
#     monkeypatch.setattr(run_module.Connection, "current", Mock())

#     assert result_set.DataFrame().equals(pd.DataFrame({"x": range(3)}))


# def test_resultset_polars_dataframe(result_set, monkeypatch):
#     assert result_set.PolarsDataFrame().frame_equal(pl.DataFrame({"x": range(3)}))


# def test_resultset_csv(result_set, tmp_empty):
#     result_set.csv("file.csv")

#     assert Path("file.csv").read_text() == "x\n0\n1\n2\n"


# def test_resultset_str(result_set):
#     assert str(result_set) == "+---+\n| x |\n+---+\n| 0 |\n| 1 |\n| 2 |\n+---+"


# def test_resultset_repr_html(result_set):
#     assert result_set._repr_html_() == (
#         "<table>\n    <thead>\n        <tr>\n            "
#         "<th>x</th>\n        </tr>\n    </thead>\n    <tbody>\n        "
#         "<tr>\n            <td>0</td>\n        </tr>\n        <tr>\n            "
#         "<td>1</td>\n        </tr>\n        <tr>\n            <td>2</td>\n        "
#         "</tr>\n    </tbody>\n</table>"
#     )


# def test_resultset_config_autolimit_dict(result, config):
#     config.autolimit = 1
#     resultset = ResultSet(result, config).fetch_results()
#     assert resultset.dict() == {"x": (0,)}


# def test_resultset_with_non_sqlalchemy_results(config):
#     df = pd.DataFrame({"x": range(3)})  # noqa
#     conn = Connection(engine=create_engine("duckdb://"))
#     result = conn.execute("SELECT * FROM df")
#     assert ResultSet(result, config).fetch_results() == [(0,), (1,), (2,)]


# def test_none_pretty(config):
#     conn = Connection(engine=create_engine("sqlite://"))
#     result = conn.execute("create table some_table (name, age)")
#     result_set = ResultSet(result, config)
#     assert result_set.pretty is None
#     assert "" == str(result_set)


# def test_lazy_loading(result, config):
#     resultset = ResultSet(result, config)
#     assert len(resultset._results) == 0
#     resultset.fetch_results()
#     assert len(resultset._results) == 3


# @pytest.mark.parametrize(
#     "autolimit, expected",
#     [
#         (None, 3),
#         (False, 3),
#         (0, 3),
#         (1, 1),
#         (2, 2),
#         (3, 3),
#         (4, 3),
#     ],
# )
# def test_lazy_loading_autolimit(result, config, autolimit, expected):
#     config.autolimit = autolimit
#     resultset = ResultSet(result, config)
#     assert len(resultset._results) == 0
#     resultset.fetch_results()
#     assert len(resultset._results) == expected


# @pytest.mark.parametrize(
#     "displaylimit, expected",
#     [
#         (0, 3),
#         (1, 1),
#         (2, 2),
#         (3, 3),
#         (4, 3),
#     ],
# )
# def test_lazy_loading_displaylimit(result, config, displaylimit, expected):
#     config.displaylimit = displaylimit
#     result_set = ResultSet(result, config)

#     assert len(result_set._results) == 0
#     result_set.fetch_results()
#     html = result_set._repr_html_()
#     row_count = _get_number_of_rows_in_html_table(html)
#     assert row_count == expected


# @pytest.mark.parametrize(
#     "displaylimit, expected_display",
#     [
#         (0, 3),
#         (1, 1),
#         (2, 2),
#         (3, 3),
#         (4, 3),
#     ],
# )
# def test_lazy_loading_displaylimit_fetch_all(
#     result, config, displaylimit, expected_display
# ):
#     max_results_count = 3
#     config.autolimit = False
#     config.displaylimit = displaylimit
#     result_set = ResultSet(result, config)

#     # Initialize result_set without fetching results
#     assert len(result_set._results) == 0

#     # Fetch the min number of rows (based on configuration)
#     result_set.fetch_results()

#     html = result_set._repr_html_()
#     row_count = _get_number_of_rows_in_html_table(html)
#     expected_results = (
#         max_results_count
#         if expected_display + 1 >= max_results_count
#         else expected_display + 1
#     )

#     assert len(result_set._results) == expected_results
#     assert row_count == expected_display

#     # Fetch the the rest results, but don't display them in the table
#     result_set.fetch_results(fetch_all=True)

#     html = result_set._repr_html_()
#     row_count = _get_number_of_rows_in_html_table(html)

#     assert len(result_set._results) == max_results_count
#     assert row_count == expected_display


# @pytest.mark.parametrize(
#     "displaylimit, expected_display",
#     [
#         (0, 3),
#         (1, 1),
#         (2, 2),
#         (3, 3),
#         (4, 3),
#     ],
# )
# def test_lazy_loading_list(result, config, displaylimit, expected_display):
#     max_results_count = 3
#     config.autolimit = False
#     config.displaylimit = displaylimit
#     result_set = ResultSet(result, config)

#     # Initialize result_set without fetching results
#     assert len(result_set._results) == 0

#     # Fetch the min number of rows (based on configuration)
#     result_set.fetch_results()

#     expected_results = (
#         max_results_count
#         if expected_display + 1 >= max_results_count
#         else expected_display + 1
#     )

#     assert len(result_set._results) == expected_results
#     assert len(list(result_set)) == max_results_count


# @pytest.mark.parametrize(
#     "autolimit, expected_results",
#     [
#         (0, 3),
#         (1, 1),
#         (2, 2),
#         (3, 3),
#         (4, 3),
#     ],
# )
# def test_lazy_loading_autolimit_list(result, config, autolimit, expected_results):
#     config.autolimit = autolimit
#     result_set = ResultSet(result, config)
#     assert len(result_set._results) == 0

#     result_set.fetch_results()

#     assert len(result_set._results) == expected_results
#     assert len(list(result_set)) == expected_results


# def _get_number_of_rows_in_html_table(html):
#     """
#     Returns the number of <tr> tags within the <tbody> section
#     """
#     pattern = r"<tbody>(.*?)<\/tbody>"
#     tbody_content = re.findall(pattern, html, re.DOTALL)[0]
#     row_count = len(re.findall(r"<tr>", tbody_content))

#     return row_count

# TODO: add dbapi tests


@pytest.fixture
def results(ip_empty):
    engine = create_engine("duckdb://")

    engine.execute("CREATE TABLE a (x INT,);")

    engine.execute("INSERT INTO a(x) VALUES (1),(2),(3),(4),(5);")

    sql = "SELECT * FROM a"
    results = engine.execute(sql)

    results.fetchmany = Mock(wraps=results.fetchmany)
    results.fetchall = Mock(wraps=results.fetchall)
    results.fetchone = Mock(wraps=results.fetchone)

    # results.fetchone = Mock(side_effect=ValueError("fetchone called"))

    yield results


@pytest.mark.parametrize("uri", ["duckdb://", "sqlite://"])
def test_convert_to_dataframe(ip_empty, uri):
    engine = create_engine(uri)

    mock = Mock()
    mock.displaylimit = 100
    mock.autolimit = 100000

    results = engine.execute("CREATE TABLE a (x INT);")
    # results = engine.execute("CREATE TABLE test (n INT, name TEXT)")

    rs = ResultSet(results, mock)
    df = rs.DataFrame()

    assert df.to_dict() == {}


def test_convert_to_dataframe_2(ip_empty):
    engine = create_engine("duckdb://")

    mock = Mock()
    mock.displaylimit = 100
    mock.autolimit = 100000

    engine.execute("CREATE TABLE a (x INT,);")
    results = engine.execute("INSERT INTO a(x) VALUES (1),(2),(3),(4),(5);")
    rs = ResultSet(results, mock)
    df = rs.DataFrame()

    assert df.to_dict() == {"Count": {0: 5}}


@pytest.mark.parametrize("uri", ["duckdb://", "sqlite://"])
def test_convert_to_dataframe_3(ip_empty, uri):
    engine = create_engine(uri)

    mock = Mock()
    mock.displaylimit = 100
    mock.autolimit = 100000

    engine.execute("CREATE TABLE a (x INT);")
    engine.execute("INSERT INTO a(x) VALUES (1),(2),(3),(4),(5);")
    results = engine.execute("SELECT * FROM a")
    # from ipdb import set_trace

    # set_trace()
    rs = ResultSet(results, mock, sql="SELECT * FROM a", engine=engine)
    df = rs.DataFrame()

    # TODO: check native duckdb was called if using duckb
    assert df.to_dict() == {"x": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}}


def test_done_fetching_if_reached_autolimit(results):
    mock = Mock()
    mock.autolimit = 2
    mock.displaylimit = 100

    rs = ResultSet(results, mock)

    assert rs.did_finish_fetching() is True


def test_done_fetching_if_reached_autolimit_2(results):
    mock = Mock()
    mock.autolimit = 4
    mock.displaylimit = 100

    rs = ResultSet(results, mock)
    list(rs)

    assert rs.did_finish_fetching() is True


@pytest.mark.parametrize("method", ["__repr__", "_repr_html_"])
@pytest.mark.parametrize("autolimit", [1000_000, 0])
def test_no_displaylimit(results, method, autolimit):
    mock = Mock()
    mock.displaylimit = 0
    mock.autolimit = autolimit

    rs = ResultSet(results, mock)
    getattr(rs, method)()

    assert rs._results == [(1,), (2,), (3,), (4,), (5,)]
    assert rs.did_finish_fetching() is True


def test_no_fetching_if_one_result():
    engine = create_engine("duckdb://")
    engine.execute("CREATE TABLE a (x INT,);")
    engine.execute("INSERT INTO a(x) VALUES (1);")

    mock = Mock()
    mock.displaylimit = 100
    mock.autolimit = 1000_000

    results = engine.execute("SELECT * FROM a")
    results.fetchmany = Mock(wraps=results.fetchmany)
    results.fetchall = Mock(wraps=results.fetchall)
    results.fetchone = Mock(wraps=results.fetchone)

    rs = ResultSet(results, mock)

    assert rs.did_finish_fetching() is True
    results.fetchall.assert_not_called()
    results.fetchmany.assert_called_once_with(size=2)
    results.fetchone.assert_not_called()

    str(rs)
    list(rs)

    results.fetchall.assert_not_called()
    results.fetchmany.assert_called_once_with(size=2)
    results.fetchone.assert_not_called()


# TODO: try with other values, and also change the display limit and re-run the repr
# TODO: try with __repr__ and __str__
def test_resultset_fetches_required_rows(results):
    mock = Mock()
    mock.displaylimit = 3
    mock.autolimit = 1000_000

    rs = ResultSet(results, mock)
    # rs.fetch_results()
    # rs._repr_html_()
    # str(rs)

    results.fetchall.assert_not_called()
    results.fetchmany.assert_called_once_with(size=2)
    results.fetchone.assert_not_called()


def test_fetches_remaining_rows(results):
    mock = Mock()
    mock.displaylimit = 1
    mock.autolimit = 1000_000

    rs = ResultSet(results, mock)

    # this will trigger fetching two
    str(rs)

    results.fetchall.assert_not_called()
    results.fetchmany.assert_has_calls([call(size=2)])
    results.fetchone.assert_not_called()

    # this will trigger fetching the rest
    assert list(rs) == [(1,), (2,), (3,), (4,), (5,)]

    results.fetchall.assert_called_once_with()
    results.fetchmany.assert_has_calls([call(size=2)])
    results.fetchone.assert_not_called()

    rs.sqlaproxy.fetchmany = Mock(side_effect=ValueError("fetchmany called"))
    rs.sqlaproxy.fetchall = Mock(side_effect=ValueError("fetchall called"))
    rs.sqlaproxy.fetchone = Mock(side_effect=ValueError("fetchone called"))

    # this should not trigger any more fetching
    assert list(rs) == [(1,), (2,), (3,), (4,), (5,)]


@pytest.mark.parametrize(
    "method, repr_",
    [
        [
            "__repr__",
            "+---+\n| x |\n+---+\n| 1 |\n| 2 |\n| 3 |\n+---+",
        ],
        [
            "_repr_html_",
            "<table>\n    <thead>\n        <tr>\n            <th>x</th>\n        "
            "</tr>\n    </thead>\n    <tbody>\n        <tr>\n            "
            "<td>1</td>\n        </tr>\n        <tr>\n            "
            "<td>2</td>\n        </tr>\n        <tr>\n            "
            "<td>3</td>\n        </tr>\n    </tbody>\n</table>",
        ],
    ],
)
def test_resultset_fetches_required_rows_repr_html(results, method, repr_):
    mock = Mock()
    mock.displaylimit = 3
    mock.autolimit = 1000_000

    rs = ResultSet(results, mock)
    rs_repr = getattr(rs, method)()

    assert repr_ in rs_repr
    assert rs.did_finish_fetching() is False
    results.fetchall.assert_not_called()
    results.fetchmany.assert_has_calls([call(size=2), call(size=1)])
    results.fetchone.assert_not_called()


def test_resultset_fetches_no_rows(results):
    mock = Mock()
    mock.displaylimit = 1
    mock.autolimit = 1000_000

    ResultSet(results, mock)

    results.fetchmany.assert_has_calls([call(size=2)])
    results.fetchone.assert_not_called()
    results.fetchall.assert_not_called()


def test_resultset_autolimit_one(results):
    mock = Mock()
    mock.displaylimit = 10
    mock.autolimit = 1

    rs = ResultSet(results, mock)
    repr(rs)
    str(rs)
    rs._repr_html_()
    list(rs)

    results.fetchmany.assert_has_calls([call(size=1)])
    results.fetchone.assert_not_called()
    results.fetchall.assert_not_called()


# TODO: try with more values of displaylimit
# TODO: test some edge cases. e.g., displaylimit is set to 10 but we only have 5 rows
def test_displaylimit_message(results):
    mock = Mock()
    mock.displaylimit = 1
    mock.autolimit = 0

    rs = ResultSet(results, mock)

    assert "Truncated to displaylimit of 1" in rs._repr_html_()
