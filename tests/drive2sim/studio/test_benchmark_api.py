"""Kiểm tra route benchmark của Studio: ``GET /api/benchmarks`` và ``GET /api/benchmarks/{name}``.

- danh sách lấy từ ``benchmarks/*.yaml`` trong thư mục làm việc (rỗng khi chưa có thư mục);
- kết quả là ``d2s-benchresult`` do ``run_benchmark`` dựng, không chạy GPU;
- file không có là ``404``, tên có ký tự lạ là ``400``, file định nghĩa sai hoặc run không có là
  ``400`` kèm thông báo của ``run_benchmark``.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from drive2sim.core.schema import SCHEMA_VERSIONS, BenchmarkGroup, BenchmarkResult
from drive2sim.engine.workdir import Workdir
from drive2sim.scripts import make_demo
from drive2sim.studio.api import app as studio_app
from drive2sim.studio.api.routes import benchmarks

DEFINITION = (
    "schema_version: d2s-benchmark/0.2\n"
    "entries:\n"
    "  - {bench_track: C, scene_id: s_1, split_id: every8_x,\n"
    "     runs: [20261101-000000-omnire-aaaa]}\n"
)


@pytest.fixture()
def workdir(tmp_path: Path) -> Workdir:
    return Workdir(tmp_path / "work")


@pytest.fixture()
def client(workdir: Workdir) -> TestClient:
    app = studio_app.create_app(workdir, frontend_dir=None)
    with TestClient(app, base_url="http://127.0.0.1") as test_client:
        yield test_client


def write_definition(workdir: Workdir, name: str, text: str = DEFINITION) -> None:
    workdir.benchmarks_dir.mkdir(parents=True, exist_ok=True)
    (workdir.benchmarks_dir / f"{name}.yaml").write_text(text)


def test_list_is_empty_without_benchmarks_dir(client: TestClient) -> None:
    assert client.get("/api/benchmarks").json() == {"items": [], "total": 0}


def test_list_shows_definition_files_sorted(client: TestClient, workdir: Workdir) -> None:
    write_definition(workdir, "week04")
    write_definition(workdir, "smoke")
    (workdir.benchmarks_dir / "notes.txt").write_text("không phải benchmark")
    body = client.get("/api/benchmarks").json()
    assert [item["name"] for item in body["items"]] == ["smoke", "week04"]
    assert body["total"] == 2


def test_get_returns_the_result(
    client: TestClient, workdir: Workdir, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_definition(workdir, "smoke")
    seen: list[Path] = []

    def fake_run_benchmark(given: Workdir, definition: Path) -> BenchmarkResult:
        seen.append(definition)
        group = BenchmarkGroup(
            bench_track="C", scene_id="s_1", split_id="every8_x", rows=[], warnings=[]
        )
        return BenchmarkResult(
            schema_version=SCHEMA_VERSIONS["benchresult"], tie_threshold_db=0.3, groups=[group]
        )

    monkeypatch.setattr(benchmarks, "run_benchmark", fake_run_benchmark)
    body = client.get("/api/benchmarks/smoke").json()
    assert seen == [workdir.benchmarks_dir / "smoke.yaml"]
    assert body["tie_threshold_db"] == 0.3 and body["tie_threshold_source"] == "default"
    assert body["groups"][0]["scene_id"] == "s_1" and body["groups"][0]["rows"] == []


def test_unknown_definition_is_404(client: TestClient) -> None:
    response = client.get("/api/benchmarks/missing")
    assert response.status_code == 404 and response.json()["error"]["code"] == "not_found"


@pytest.mark.parametrize("name", ["..", ".hidden", "a b", "a%2Fb", "x;y"])
def test_odd_names_are_rejected(client: TestClient, name: str) -> None:
    assert client.get(f"/api/benchmarks/{name}").status_code in (400, 404)


def test_invalid_definition_is_400_with_the_message(client: TestClient, workdir: Workdir) -> None:
    write_definition(workdir, "broken", "schema_version: d2s-benchmark/0.2\nentries: []\n")
    response = client.get("/api/benchmarks/broken")
    assert response.status_code == 400 and response.json()["error"]["code"] == "invalid_request"


def test_unknown_run_is_400_with_the_message(client: TestClient, workdir: Workdir) -> None:
    write_definition(workdir, "smoke")  # cảnh và run không tồn tại trong thư mục làm việc
    response = client.get("/api/benchmarks/smoke")
    assert response.status_code == 400
    assert "20261101-000000-omnire-aaaa" in response.json()["error"]["message"]


def test_demo_workdir_has_a_working_benchmark(tmp_path: Path) -> None:
    # dữ liệu mẫu (make_demo) có benchmarks/demo.yaml để xem trang Benchmark khi chưa có số thật
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(make_demo, "write_video", lambda path, seconds=3: False)
        assert make_demo.main([str(tmp_path / "demo")]) == 0
    app = studio_app.create_app(Workdir(tmp_path / "demo"), frontend_dir=None)
    with TestClient(app, base_url="http://127.0.0.1") as demo_client:
        assert [i["name"] for i in demo_client.get("/api/benchmarks").json()["items"]] == ["demo"]
        body = demo_client.get("/api/benchmarks/demo").json()
    assert len(body["groups"]) == 2
    first = body["groups"][0]
    assert first["winner_run_id"] and first["rows"][0]["rank"] == 1
    assert {row["approach_id"] for row in first["rows"]} >= {"omnire", "streetgs"}
