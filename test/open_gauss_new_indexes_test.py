import json
import warnings
from unittest.mock import MagicMock

import numpy as np
import pytest

from ann_benchmarks.algorithms.openGauss import module as m


CLASSES = [m.OpenGaussHNSWRaBitQ, m.OpenGaussIVFFlatRaBitQ, m.OpenGaussDiskANN]


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("metric,operator,opclass", [
    ("euclidean", "<->", "vector_l2_ops"), ("angular", "<=>", "vector_cosine_ops")])
def test_distance_mapping(cls, metric, operator, opclass):
    algo = cls(metric, {})
    assert algo.operator == operator
    assert algo.opclass == opclass
    assert operator in algo.query_sql.as_string()
    assert "%s::vector" in algo.query_sql.as_string()


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("params", [{"unknown": 1}, {"concurrents": 0}, {"concurrents": True},
    {"parallel_workers": 0}, {"parallel_workers": 129}, {"enable_opfusion": "off"}, {"enable_seqscan": 0},
    {"statement_timeout_ms": 0}, {"lock_timeout_ms": 1.5}, {"schema": ""},
    {"maintenance_work_mem_mb": 0}, {"maintenance_work_mem_mb": True}, {"maintenance_work_mem_mb": "256"}])
def test_invalid_common_parameters(cls, params):
    with pytest.raises(ValueError):
        cls("euclidean", params)


@pytest.mark.parametrize("metric", ["dot", "l2", None, "hamming"])
def test_unsupported_distance(metric):
    with pytest.raises(ValueError):
        m.OpenGaussHNSWRaBitQ(metric, {})


@pytest.mark.parametrize("params", [{"m": 1}, {"m": 101}, {"m": 16.0}, {"m": "16"},
    {"ef_construction": 1001}, {"m": 32, "ef_construction": 63}, {"enable_rabitq": False},
    {"rabitq_refine_type": "PQ"}, {"rabitq_fht": "on"}, {"enable_pq": True}])
def test_invalid_hnsw_parameters(params):
    with pytest.raises(ValueError):
        m.OpenGaussHNSWRaBitQ("euclidean", params)


@pytest.mark.parametrize("value", [0, 32769, True, "20"])
def test_invalid_lists(value):
    with pytest.raises(ValueError):
        m.OpenGaussIVFFlatRaBitQ("euclidean", {"lists": value})


@pytest.mark.parametrize("params", [{"index_size": 15}, {"index_size": 1001}, {"pq_m": 0},
    {"pq_m": 193}, {"enable_pq": "false"}, {"enable_rabitq": True}])
def test_invalid_diskann_parameters(params):
    with pytest.raises(ValueError):
        m.OpenGaussDiskANN("euclidean", params)


@pytest.mark.parametrize("refine", ["fp32", "sq8", "none"])
def test_refine_types(refine):
    algo = m.OpenGaussHNSWRaBitQ("euclidean", {"rabitq_refine_type": refine})
    assert algo._index_options()["rabitq_refine_type"] == refine.upper()


@pytest.mark.parametrize("cls", CLASSES)
def test_not_fitted_and_unique_objects(cls):
    a, b = cls("euclidean", {}), cls("euclidean", {})
    assert a.table_name != b.table_name
    assert len(a.index_name.encode()) < 63
    assert a.get_memory_usage() == 0
    with pytest.raises(RuntimeError):
        a.query([1, 2], 1)
    a.done()
    a.done()


@pytest.mark.parametrize("data", [[], [1, 2], [[float("nan"), 1]], [[1, float("inf")]],
    [[1e100, 2]], [["1", "2"]], np.empty((0, 2)), np.empty((1, 2001)), [[1j, 2j]]])
def test_invalid_data_before_connect(data, monkeypatch):
    algo = m.OpenGaussHNSWRaBitQ("euclidean", {})
    connect = MagicMock()
    monkeypatch.setattr(algo, "_connect", connect)
    with pytest.raises(ValueError):
        algo.fit(data)
    connect.assert_not_called()


def test_dimension_and_pq_divisibility():
    with pytest.raises(ValueError):
        m.OpenGaussDiskANN("euclidean", {})._validate_matrix(np.ones((100, 1537)), fitting=True)
    with pytest.raises(ValueError):
        m.OpenGaussDiskANN("euclidean", {"enable_pq": True, "pq_m": 8})._validate_matrix(np.ones((100, 15)), fitting=True)
    with pytest.raises(ValueError):
        m.OpenGaussIVFFlatRaBitQ("euclidean", {"lists": 101})._validate_matrix(np.ones((100, 64)), fitting=True)
    with pytest.raises(ValueError):
        m.OpenGaussHNSWRaBitQ("angular", {})._validate_matrix(np.zeros((100, 64)), fitting=True)
    with pytest.raises(ValueError):
        m.OpenGaussHNSWRaBitQ("angular", {})._validate_matrix(np.ones((100, 1)), fitting=True)


@pytest.mark.parametrize("key,value", [("hnsw_ef_search", 0), ("hnsw_ef_search", 1000001),
    ("rbq_query_bits", 0), ("rbq_query_bits", 9), ("rbq_query_bits", 1.5),
    ("rbq_refinek", -1), ("rbq_refinek", 1001), ("rbq_refinek", float("nan")),
    ("rbq_refinek", True)])
def test_query_bounds(key, value):
    algo = m.OpenGaussHNSWRaBitQ("euclidean", {})
    params = {**algo.query_defaults, key: value}
    with pytest.raises(ValueError):
        algo._validate_query(params)


def test_fractional_refinek_and_source_bounds():
    algo = m.OpenGaussHNSWRaBitQ("euclidean", {"m": 100, "ef_construction": 1000})
    params = {"hnsw_ef_search": 1000000, "rbq_query_bits": 1, "rbq_refinek": 2.5}
    algo._validate_query(params)
    assert params["rbq_refinek"] == 2.5
    params["rbq_refinek"] = 0
    algo._validate_query(params)
    with pytest.raises(ValueError):
        m.OpenGaussIVFFlatRaBitQ("euclidean", {"lists": 3})._validate_query(
            {"ivfflat_probes": 4, "rbq_query_bits": 8, "rbq_refinek": 5})
    assert m.OpenGaussIVFFlatRaBitQ("euclidean", {"lists": 1}).query_params["ivfflat_probes"] == 1


def fitted_mock(cls=m.OpenGaussHNSWRaBitQ):
    algo = cls("euclidean", {})
    algo.connection = MagicMock(closed=False, broken=False)
    algo._created = True
    algo._dimension = 2
    algo._count = 100
    algo._probe = np.ones(2, dtype=np.float32)
    return algo


def test_parameter_group_reset_and_validation(monkeypatch):
    algo = fitted_mock()
    monkeypatch.setattr(algo, "_apply_settings", MagicMock())
    monkeypatch.setattr(algo, "_record_settings", MagicMock())
    monkeypatch.setattr(algo, "explain", MagicMock())
    old_pool = algo.search_pool
    old_pool.close = MagicMock()
    algo.set_query_arguments({"hnsw_ef_search": 50, "rbq_query_bits": 2, "rbq_refinek": 2.5})
    old_pool.close.assert_called_once()
    algo.set_query_arguments({"hnsw_ef_search": 80})
    assert algo.query_params == {"hnsw_ef_search": 80, "rbq_query_bits": 8, "rbq_refinek": 5.0}
    before = dict(algo.query_params)
    with pytest.raises(ValueError):
        algo.set_query_arguments({"unknown": 1})
    assert algo.query_params == before


@pytest.mark.parametrize("value,k", [([1], 1), ([1, 2], 0), ([1, 2], 101),
    ([1, 2], True), ([float("nan"), 2], 1), ([[1, 2]], 1)])
def test_invalid_query_shape_and_k(value, k):
    with pytest.raises(ValueError):
        fitted_mock()._query_vector(value, k)


def test_sql_parameters_and_bad_result_detection():
    algo = fitted_mock()
    cur = algo.connection.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = [(3,), (1,)]
    assert algo.query([1, 2], 2) == [3, 1]
    call = cur.execute.call_args
    assert call.args[1] == ("[1,2]", 2)
    assert "%s::vector" in call.args[0].as_string()
    for rows in [[(3,), (3,)], [(101,), (1,)], [(1,), (2,), (3,)]]:
        cur.fetchall.return_value = rows
        with pytest.raises(RuntimeError):
            algo.query([1, 2], 2)
    for rows in [[], [(1,)]]:
        cur.fetchall.return_value = rows
        assert algo.query([1, 2], 2) == [r[0] for r in rows]


def test_normalize_query_uses_effective_ivf_default():
    algo = m.OpenGaussIVFFlatRaBitQ("euclidean", {"lists": 1})
    assert algo.normalize_query_arguments({"rbq_query_bits": 4}) == {
        "ivfflat_probes": 1, "rbq_query_bits": 4, "rbq_refinek": 5.0}
    with pytest.raises(ValueError, match="Unknown"):
        algo.normalize_query_arguments({"typo": 1})


def test_float32_vector_roundtrip():
    vector = np.array([1.23456789, -0.000000123456789, 1e30], dtype=np.float32)
    assert np.array_equal(np.asarray(json.loads(m.vector_text(vector)), dtype=np.float32), vector)


def test_connection_environment_no_hardcoded_endpoint(monkeypatch):
    for key in ("USER", "PASSWORD", "DBNAME", "HOST", "PORT"):
        monkeypatch.delenv("ANN_BENCHMARKS_OG_" + key, raising=False)
    connect = MagicMock()
    monkeypatch.setattr(m.psycopg, "connect", connect)
    algo = m.OpenGaussDiskANN("euclidean", {})
    algo._connect()
    assert "host" not in connect.call_args.kwargs
    assert "password" not in connect.call_args.kwargs
    monkeypatch.setenv("ANN_BENCHMARKS_OG_PORT", "55433")
    monkeypatch.setenv("ANN_BENCHMARKS_OG_HOST", "/tmp/test-socket")
    algo._connect()
    assert connect.call_args.kwargs["port"] == 55433
    assert connect.call_args.kwargs["host"] == "/tmp/test-socket"


def test_done_drops_only_owned_table_and_is_idempotent():
    algo = fitted_mock()
    connection = algo.connection
    cur = connection.cursor.return_value.__enter__.return_value
    algo.done()
    statement = cur.execute.call_args.args[0].as_string()
    assert algo.table_name in statement
    assert "CASCADE" not in statement and "items" not in statement
    assert not algo._created and algo.connection is None
    connection.close.assert_called_once()
    algo.done()


def test_cleanup_reconnects_if_query_connection_broken(monkeypatch):
    algo = fitted_mock()
    algo.connection.broken = True
    fresh = MagicMock(closed=False, broken=False)
    monkeypatch.setattr(algo, "_connect", MagicMock(return_value=fresh))
    algo.done()
    assert not algo._created
    fresh.close.assert_called_once()


def test_fit_error_closes_connection_and_cleans_created_object(monkeypatch):
    algo = m.OpenGaussHNSWRaBitQ("euclidean", {})
    connection = MagicMock(closed=False, broken=False)
    cur = connection.cursor.return_value.__enter__.return_value
    def execute(statement, *args, **kwargs):
        if "CREATE INDEX" in statement.as_string():
            raise RuntimeError("injected index build failure")
    cur.execute.side_effect = execute
    monkeypatch.setattr(algo, "_connect", MagicMock(return_value=connection))
    with pytest.raises(RuntimeError, match="injected index"):
        algo.fit(np.ones((100, 2), dtype=np.float32))
    statements = [c.args[0].as_string() for c in cur.execute.call_args_list]
    assert any(s.startswith("DROP TABLE") for s in statements)
    assert not algo._created
    connection.close.assert_called_once()


def test_pool_failed_prepare_closes_all_partial_connections(monkeypatch):
    algo = m.OpenGaussDiskANN("euclidean", {"concurrents": 2})
    connection = MagicMock()
    monkeypatch.setattr(algo, "_connect", MagicMock(side_effect=[connection, RuntimeError("connect failure")]))
    monkeypatch.setattr(algo, "_apply_settings", MagicMock())
    with pytest.raises(RuntimeError, match="connect failure"):
        algo.search_pool.prepare()
    connection.close.assert_called_once()
    assert not algo.search_pool.connections
    algo.search_pool.close()


def test_effective_settings_applied_to_all_workers(monkeypatch):
    algo = m.OpenGaussHNSWRaBitQ("euclidean", {"concurrents": 2, "enable_opfusion": False,
                                             "maintenance_work_mem_mb": 128})
    connections = [MagicMock(), MagicMock()]
    monkeypatch.setattr(algo, "_connect", MagicMock(side_effect=connections))
    algo.search_pool.prepare()
    for connection in connections:
        calls = connection.cursor.return_value.__enter__.return_value.execute.call_args_list
        statements = [c.args[0].as_string() for c in calls]
        assert any('"hnsw_ef_search" = 100' in s for s in statements)
        assert any('"enable_opfusion" = false' in s for s in statements)
        assert any('"rbq_refinek" = 5.0' in s for s in statements)
        assert any('"maintenance_work_mem" = \'128MB\'' in s for s in statements)
    algo.search_pool.close()
    for connection in connections:
        connection.close.assert_called_once()


def test_explain_rejects_fallback():
    algo = fitted_mock()
    cur = algo.connection.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = [("Seq Scan on another_table",)]
    with pytest.raises(RuntimeError, match="planner did not"):
        algo.explain([1, 2], 10)


def test_settings_record_includes_units():
    algo = fitted_mock()
    cur = algo.connection.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = [("maintenance_work_mem", "262144", "kB"),
                                 ("shared_buffers", "16384", "8kB"), ("enable_seqscan", "off", None)]
    algo._record_settings()
    assert json.loads(algo._metadata["effective_settings_json"])["shared_buffers"] == "16384"
    assert json.loads(algo._metadata["effective_setting_units_json"])["shared_buffers"] == "8kB"


def test_batch_empty_and_order(monkeypatch):
    algo = fitted_mock()
    monkeypatch.setattr(algo, "_connect", MagicMock(side_effect=lambda: MagicMock()))
    monkeypatch.setattr(algo, "_apply_settings", MagicMock())
    monkeypatch.setattr(algo, "_execute_query", lambda conn, v, n: [int(v[0])])
    algo.batch_query(np.empty((0, 2)), 1)
    assert algo.get_batch_results() == []
    algo.batch_query(np.array([[3, 1], [2, 1], [1, 1]], dtype=np.float32), 1)
    assert algo.get_batch_results() == [[3], [2], [1]]
    assert len(algo.get_batch_latencies()) == 3
    assert algo._batch_stats["batch_qps"] > 0
    algo.search_pool.close()


def test_batch_failure_closes_workers(monkeypatch):
    algo = fitted_mock()
    monkeypatch.setattr(algo, "_connect", MagicMock(side_effect=lambda: MagicMock()))
    monkeypatch.setattr(algo, "_apply_settings", MagicMock())
    monkeypatch.setattr(algo, "_execute_query", MagicMock(side_effect=RuntimeError("injected query failure")))
    with pytest.raises(RuntimeError, match="injected query"):
        algo.batch_query(np.ones((5, 2)), 1)
    assert algo.search_pool.executor is None and not algo.search_pool.connections


@pytest.mark.parametrize("build", ["2397a86f", "e024ee66"])
@pytest.mark.parametrize("cls,metric,refine,expected", [
    (m.OpenGaussIVFFlatRaBitQ, "euclidean", "FP32", "ivf-rabitq-refinement"),
    (m.OpenGaussIVFFlatRaBitQ, "angular", "SQ8", "ivf-rabitq-refinement"),
    (m.OpenGaussHNSWRaBitQ, "angular", "FP32", "hnsw-rabitq-cosine-refinement"),
    (m.OpenGaussHNSWRaBitQ, "angular", "SQ8", "hnsw-rabitq-cosine-refinement"),
    (m.OpenGaussHNSWRaBitQ, "euclidean", "FP32", None),
    (m.OpenGaussHNSWRaBitQ, "angular", "NONE", None),
    (m.OpenGaussIVFFlatRaBitQ, "euclidean", "NONE", None),
])
def test_known_server_risk_scope_and_no_parameter_substitution(build, cls, metric, refine, expected):
    algo = cls(metric, {"rabitq_refine_type": refine})
    algo._metadata["database_version"] = f"(openGauss 7.0.0 build {build})"
    before = (dict(algo.params), dict(algo.query_params))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        algo._record_known_server_risks()
        algo._record_known_server_risks()
    assert json.loads(algo._metadata["known_server_risks_json"]) == ([expected] if expected else [])
    assert len(caught) == (1 if expected else 0)
    if expected:
        assert "requested parameters and returned order are unchanged" in str(caught[0].message)
    assert (algo.params, algo.query_params) == before


@pytest.mark.parametrize("version", ["", "openGauss 8.0.0 build abc12345", "build 2397a86f0"])
def test_unknown_server_build_is_not_declared_affected(version):
    algo = m.OpenGaussIVFFlatRaBitQ("euclidean", {})
    algo._metadata["database_version"] = version
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        algo._record_known_server_risks()
    assert not caught
    assert json.loads(algo._metadata["known_server_risks_json"]) == []


def test_risk_warning_when_query_group_enables_refinement(monkeypatch):
    algo = fitted_mock(m.OpenGaussIVFFlatRaBitQ)
    algo._metadata["database_version"] = "(openGauss 7.0.0 build 2397a86f)"
    monkeypatch.setattr(algo, "_apply_settings", MagicMock())
    monkeypatch.setattr(algo, "_record_settings", MagicMock())
    monkeypatch.setattr(algo, "explain", MagicMock())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        algo.set_query_arguments({"rbq_refinek": 0})
    assert not caught
    assert json.loads(algo._metadata["known_server_risks_json"]) == []
    with pytest.warns(RuntimeWarning, match="independent gsql"):
        algo.set_query_arguments({"rbq_refinek": 5})
    assert algo.query_params["rbq_refinek"] == 5
    assert json.loads(algo._metadata["known_server_risks_json"]) == ["ivf-rabitq-refinement"]
    algo.set_query_arguments({"rbq_refinek": 0})
    assert json.loads(algo._metadata["known_server_risks_json"]) == []
