# AGENTS.md

Quy định cho người và agent viết code trong repo này. Đọc hết trước khi sửa bất cứ thứ gì.

## 1. Repo này là gì

- Repo xây tool `drive2sim` (dựng cảnh lái xe thành cảnh 3D Gaussian Splatting để mô phỏng) trên nền DriveStudio.
- Toàn bộ mã nằm trong package `drive2sim/`:
  - `drive2sim/recon/`: mã tái dựng lấy từ DriveStudio (tiền xử lý, loader, model, trainer, render);
  - các thư mục còn lại (`core/`, `engine/`, `stages/`, `studio/`, `kaggle/`, `sim/`, `cli/`, ...): phần tool viết mới.
- Làm việc trên branch `drive2sim`. Branch `main` là bản cũ của teammate, không merge thẳng được vì cấu trúc thư mục đã đổi.

## 2. Đọc docs trước khi quyết định

Ba tài liệu thiết kế nằm trong `docs/drive2sim/`:

| Tài liệu | Trả lời câu hỏi |
|---|---|
| `SPEC.md` | Tool phải làm gì: mức L1–L4, luồng sử dụng, Studio, đánh giá |
| `ARCHITECTURE.md` | Xây thế nào: các quyết định thiết kế kèm lý do, cấu trúc repo, chỗ đặt code mới (mục 3.1), hiện trạng và lộ trình (mục 7) |
| `CONTRACTS.md` | Định dạng file, interface của stage và worker, quy tắc đặt tên, config, ngôn ngữ và CLI (mục 10), API của Studio (mục 11) |

Quy tắc:
- **Trước khi viết code có quyết định về thiết kế** (thêm module, đổi định dạng file, đổi interface, thêm lệnh CLI, chọn thư viện): đọc kỹ phần liên quan của cả ba tài liệu. Code phải theo docs.
- **Nếu cần làm khác docs**: không âm thầm làm khác. Sửa docs trước hoặc cùng lúc, ghi rõ lý do, ưu và nhược điểm theo mẫu ở `ARCHITECTURE.md` mục 2.
- **Khi code có quyết định mới về kiến trúc**: cập nhật cả ba tài liệu cho đồng bộ **trong cùng commit** với code. Gồm: cây thư mục (mục 3), bảng chỗ đặt code (mục 3.1), bảng tiến độ (mục 7.1), định dạng file và tên (CONTRACTS).
- `tests/drive2sim/core/test_contracts_examples.py` đọc mọi ví dụ JSON/YAML có `schema_version` trong `CONTRACTS.md` và kiểm tra theo schema. Sửa định dạng file thì sửa cả ví dụ trong docs và schema trong `drive2sim/core/schema.py`, nếu không test sẽ báo lỗi.
- Thông tin về paper, repo, số liệu: kiểm tra trong mã nguồn hoặc tài liệu gốc rồi mới ghi vào docs; chỗ nào chưa kiểm tra thì ghi rõ "chưa kiểm tra".

## 3. Đặt code ở đâu, đặt tên ra sao

- Chỗ đặt file mới: theo bảng ở `ARCHITECTURE.md` mục 3.1. Không thêm thư mục mã mới ở gốc repo; ở gốc chỉ có `docs/`, `tests/`, `third_party/`, `envs/` và file cấu hình.
- Tên định danh, file, trường, tham số, lệnh CLI: theo `CONTRACTS.md` mục 10. Config kế thừa theo mục 10.5; route, lỗi và sự kiện của API theo mục 11. Không tự nghĩ ra mã lỗi, tên sự kiện hay tuỳ chọn CLI mới; thiếu thì thêm vào bảng trong CONTRACTS trước.
- Một file một việc: mỗi stage, converter, nhóm lệnh CLI, nhóm route API là một file.
- Import theo đường dẫn đầy đủ (`from drive2sim.recon.models.nodes import RigidNodes`), không dùng import tương đối.
- File đi kèm package (assets, configs, approaches) truy cập qua `drive2sim/paths.py`, không dùng đường dẫn tương đối theo thư mục đang chạy.

## 4. Chuẩn viết code

Áp dụng cho mọi code viết mới. Với mã cũ trong `drive2sim/recon/`: chỉ áp cho phần đang sửa, không định dạng lại cả file cũ trong cùng commit (để diff còn đọc được).

### 4.1 PEP 8 và phiên bản Python

- Theo PEP 8; dòng tối đa 99 ký tự.
- Thứ tự import: thư viện chuẩn, thư viện ngoài, `drive2sim`; mỗi nhóm cách nhau một dòng trống.
- Code phải chạy được trên **Python 3.9** (phiên bản của môi trường DriveStudio):
  - dùng kiểu có sẵn cho chú thích: `list[int]`, `dict[str, float]`, `tuple[int, int]` (có từ 3.9, PEP 585); `Sequence`, `Iterable` lấy từ `collections.abc`;
  - dùng `Optional[X]`, `Union[X, Y]` từ `typing`; **không** dùng cú pháp `X | Y` (cần 3.10);
  - không dùng `match` (cần 3.10).
- Kiểm tra tự động bằng `ruff` (cấu hình trong `pyproject.toml`, áp cho code mới, bỏ qua `drive2sim/recon/` và `third_party/`):

```bash
uvx ruff format drive2sim tests      # định dạng theo PEP 8
uvx ruff check drive2sim tests       # PEP 8, đặt tên, chú thích kiểu, docstring, import
```

  Code mới phải qua `ruff check` không còn lỗi trước khi commit.

### 4.2 Đặt tên

Theo PEP 8 và phải **đọc là hiểu**, không cần đoán.

| Thứ | Quy tắc | Ví dụ đúng | Ví dụ tránh |
|---|---|---|---|
| Package, thư mục mã | chữ thường, ngắn, không gạch dưới nếu được | `core`, `stages`, `studio` | `Core`, `my_stuff` |
| Module (file `.py`) | `snake_case`, danh từ nói rõ nội dung | `identifiers.py`, `files.py`, `split.py` | `ids.py`, `utils2.py`, `io.py` |
| Lớp | `PascalCase`, danh từ | `RunRecord`, `PrepMasksSegformerStage` | `run_record`, `Mgr` |
| Hàm, phương thức | `snake_case`, bắt đầu bằng động từ | `make_cache_key`, `load_any`, `check_stage_id` | `cache_key2`, `do_it` |
| Biến, tham số | `snake_case`, danh từ đầy đủ | `camera_id`, `frame_range`, `strategy_name` | `c`, `fr`, `tmp2` |
| Boolean | tính từ hoặc `is_`, `has_`, `uses_` | `cacheable`, `uses_lidar`, `is_final` | `flag`, `check` |
| Hằng | `UPPER_CASE` ở mức module | `SCHEMA_VERSIONS`, `STAGE_GROUPS` | `schemaVersions` |
| Thành phần nội bộ | bắt đầu bằng `_` | `_read_raw`, `_validate_pairs` | |
| Hàm test | `test_<hành vi được kiểm tra>` | `test_cache_key_changes_with_params` | `test1` |

Quy tắc thêm:
- Không viết tắt khó hiểu. Chỉ dùng chữ viết tắt phổ biến trong ngành (`id`, `rgb`, `lidar`, `gpu`) hoặc đã định nghĩa trong docs (`T_a_b`).
- Biến một chữ cái chỉ dùng cho chỉ số vòng lặp ngắn (`i`) hoặc ký hiệu toán đã quen (`n` trong `every_nth_frame:N`).
- Không đặt tên trùng thư viện chuẩn hay hàm có sẵn: không dùng `io.py`, `json.py`, biến `id`, `type`, `list`, `dict`, `input`.
- Tên theo docs: định danh, trường, tham số, lệnh CLI theo `CONTRACTS.md` mục 10. Tên trường biến đổi toạ độ `T_<a>_<b>` là ngoại lệ có chủ ý của PEP 8 (ruff đã tắt các luật N802, N803, N806, N815 cho việc này).

### 4.3 Chú thích kiểu (type hints)

- Mọi hàm, phương thức có chú thích kiểu cho mọi tham số và giá trị trả về, kể cả `-> None`.
- Thuộc tính lớp có chú thích kiểu; hằng của lớp không phải field Pydantic dùng `ClassVar`.
- Tránh `Any` khi biết được kiểu cụ thể; nhận "bất kỳ giá trị nào" thì dùng `object`.

### 4.4 Docstring bằng tiếng Việt

**Mục tiêu:** người chưa đọc code, chỉ đọc docstring, vẫn hiểu được file, lớp, hàm đó làm gì, để làm gì, nhận gì, trả về gì và khi nào lỗi. Viết tiếng Việt đầy đủ câu, rõ ràng; không viết kiểu ghi chú cụt.

Tiêu đề mục `Args:`, `Returns:`, `Raises:`, `Attributes:`, `Example:` giữ nguyên tiếng Anh vì `ruff` kiểm tra định dạng Google theo các tên này; nội dung bên dưới viết tiếng Việt.

**Docstring của file (module)**, đặt ở đầu mọi file `.py`, gồm:
1. Câu đầu: file này chứa gì.
2. Danh sách các chức năng chính: mỗi lớp, hàm công khai quan trọng làm việc gì.
3. File được dùng ở đâu, ở bước nào trong luồng của tool (ai gọi nó, gọi khi nào).
4. Mục docs liên quan.

```python
"""Quản lý cache cho kết quả của các stage dùng chung giữa các approach.

File này chứa:
- ``CacheStore``: tìm thư mục cache đã có, tạo thư mục tạm ``.partial`` để ghi, và đổi tên
  thành thư mục chính thức khi ghi xong.
- ``input_key_for_scene``, ``input_key_for_split``: tạo khoá đại diện cho cảnh và split, để
  khoá cache đổi theo khi cảnh hoặc split đổi.

Engine (``drive2sim/engine/run.py``) dùng file này khi chạy một stage có ``cacheable = True``:
nếu kết quả đã có trong cache thì dùng lại, không chạy lại stage. Xem ARCHITECTURE mục 2.7,
CONTRACTS mục 5.2.
"""
```

**Docstring của lớp**, gồm: lớp đại diện cho cái gì; dùng để làm gì; ai tạo và ai dùng nó; mục `Attributes:` giải thích từng thuộc tính.

**Docstring của hàm và phương thức**, kể cả hàm nội bộ `_tên` khi hàm dài hơn vài dòng hoặc có logic không hiển nhiên, gồm:
1. Câu đầu: hàm làm gì, bắt đầu bằng động từ.
2. Đoạn giải thích: để làm gì, vì sao cần; cách làm nếu không hiển nhiên; gọi khi nào; ví dụ ngắn nếu giúp hiểu.
3. `Args:` từng tham số: ý nghĩa, dạng giá trị, đơn vị, giá trị mặc định nghĩa là gì.
4. `Returns:` trả về gì, ở dạng nào, kèm ví dụ khi dạng không hiển nhiên.
5. `Raises:` lỗi nào, trong trường hợp nào.

```python
def make_cache_key(
    scene_id: str, stage_id: str, version: str, params: dict, input_keys: Iterable[str]
) -> str:
    """Tạo khoá cache cho kết quả của một stage có cache.

    Khoá dùng để đặt tên thư mục trong ``cache/`` và để nhận ra một kết quả đã có sẵn: hai lần
    chạy cùng stage, cùng phiên bản, cùng tham số và cùng đầu vào cho ra cùng khoá, nên lần sau
    dùng lại kết quả của lần trước. Khoá chỉ băm tham số và khoá của đầu vào, không băm nội dung
    dữ liệu, vì dữ liệu chuẩn không đổi sau khi ingest (CONTRACTS mục 5.2).

    Args:
        scene_id: Định danh cảnh, dạng ``<dataset>_<sequence_id>``, ví dụ ``nuscenes_scene-0061``.
        stage_id: Định danh stage, dạng ``<nhóm>.<tên>``, ví dụ ``init.depth``.
        version: Phiên bản của stage, dạng ``MAJOR.MINOR.PATCH``.
        params: Tham số đã gộp của stage.
        input_keys: Khoá của các đầu vào; thứ tự không ảnh hưởng tới kết quả.

    Returns:
        Khoá dạng ``<scene_id>/<stage_id>@<version>-<hash8>``, ví dụ
        ``nuscenes_scene-0061/init.depth@0.1.0-9b03c2d1``.

    Raises:
        ValueError: Nếu ``scene_id``, ``stage_id`` hoặc ``version`` sai định dạng.
    """
```

**Khác:**
- Hàm test đặt tên mô tả rõ hành vi nên không bắt buộc docstring; file test vẫn có docstring nói file kiểm tra những gì.
- Tên biến, hàm, lớp giữ tiếng Anh; docstring và comment tiếng Việt; ngôn ngữ của thông báo lỗi theo `CONTRACTS.md` mục 10.6.
- Khi sửa một hàm có sẵn mà docstring chưa đạt chuẩn này, viết lại docstring trong cùng commit.

### 4.5 Comment

- Comment giải thích **vì sao** làm vậy, giả định nào đang dựa vào, chỗ nào dễ sai; không lặp lại điều code đã nói rõ.
- Đoạn xử lý khó (công thức, phép biến đổi toạ độ, quy ước của thư viện ngoài) có comment ngắn kèm nguồn, ví dụ: `# T_a_b đưa điểm từ hệ a sang hệ b (quy ước NCore, CONTRACTS mục 1)`.
- Khi viết khác cách của DriveStudio hay thư viện gốc, ghi lý do.

### 4.6 SOLID và KISS

- **S (một trách nhiệm):** mỗi lớp, hàm làm một việc. Ví dụ: stage chỉ ghi file vào `output_dir`; cache, file `done`, `artifact.json` do engine lo (CONTRACTS mục 5.1).
- **O (mở để mở rộng, đóng để sửa):** thêm tính năng bằng cách thêm stage, approach, converter, route mới qua registry; không sửa engine, Studio hay server cho từng approach.
- **L (thay thế được):** mọi lớp cài cùng một interface (`Stage`, `Renderer`, `ActorEditor`) phải dùng thay cho nhau được mà nơi gọi không cần biết lớp cụ thể.
- **I (interface nhỏ):** interface chỉ có những gì nơi gọi cần; tách interface theo mức năng lực (L2 `Renderer`, L3 `ActorEditor`) thay vì một interface lớn.
- **D (phụ thuộc vào trừu tượng):** Studio, CLI, server gọi qua interface và file theo CONTRACTS (`run.json`, `Renderer`), không import thẳng vào bên trong `recon/`.
- **KISS:** chọn cách đơn giản nhất đáp ứng docs. Không thêm lớp trừu tượng, tham số, tính năng "để sau này dùng" khi chưa có nhu cầu thật. Ví dụ đã chốt: pipeline tuyến tính thay vì đồ thị, cache chỉ cho stage dùng chung (`ARCHITECTURE.md` mục 2.5, 2.7).

## 5. Kiểm thử

- Test đặt ở `tests/drive2sim/<cùng đường dẫn với code>/test_<module>.py`, dùng `pytest`.
- Code mới có test đi kèm; sửa lỗi thì thêm test tái hiện lỗi đó.
- Các lệnh dưới đây có sẵn trong `Makefile` ở gốc repo; gõ `make` để xem danh sách.
- Chạy test: `.venv/bin/python -m pytest` hoặc `make test` (môi trường nhẹ ở local, cài bằng `pip install -e ".[dev,studio]"` hoặc `make setup`).
- Test không được gọi Kaggle hay Colab thật: `tests/drive2sim/conftest.py` chặn lệnh `kaggle` và `colab` cho mọi test; test cần Kaggle, Colab thì truyền runner hoặc client giả.
- Test không được mở cửa sổ trên máy: `conftest.py` bỏ `DISPLAY` và `WAYLAND_DISPLAY` cho mọi test, nên `/api/reveal` và `/api/pick-folder` báo máy không có giao diện; test cần giao diện thì tự đặt biến và patch lệnh chạy.
- Trước khi commit: `uvx ruff format drive2sim tests`, `uvx ruff check drive2sim tests`, rồi chạy test (`make check` làm cả ba).
- Kiểm tra Python 3.9 (`make test-py39`): `uv run --python 3.9 --no-project --with "pydantic>=2.5,<3" --with "pyyaml>=6" --with "pytest>=7" --with "fastapi>=0.110" --with "httpx>=0.24" python -m pytest -q`.
- Thay đổi hành vi ở `drive2sim/recon/` không chạy được trên máy không có GPU: ghi rõ "chưa chạy thật" trong commit và docs, rồi kiểm tra hồi quy trên Kaggle (train lại cùng cảnh, so PSNR) theo `ARCHITECTURE.md` mục 7.3 và 10.

### 5.1 Giao diện Studio

- Mã ở `drive2sim/studio/frontend/` (Vite + React + TypeScript). Cài một lần: `npm install` trong thư mục đó.
- Sửa schema trong `drive2sim/core/schema.py` thì chạy `npm run types` (cần lệnh `d2s` trên PATH; `make types` tự thêm `.venv/bin`) để sinh lại `src/api/generated/`; không sửa tay các file đó.
- Sửa mã giao diện thì chạy `npm run build` hoặc `make build` (kiểm tra kiểu rồi build) và **commit cả `dist/`** trong cùng commit, vì người dùng chạy bản đã build, không cần Node.js.
- Chuỗi giao diện đặt trong `src/locales/vi.json` và `en.json`, thêm khoá ở cả hai file (CONTRACTS mục 10.6).
- Xem thử: `make studio WORKDIR=<thư mục có dữ liệu>` (hoặc `d2s studio --workdir ...`) rồi mở địa chỉ được in ra; khi sửa giao diện, `make dev` bật cả Studio và Vite tự tải lại. Hai lệnh tự chọn cổng trống bắt đầu từ `API_PORT=8000`, `WEB_PORT=5173`; Vite chuyển `/api` sang cổng Studio qua biến `D2S_STUDIO_PORT`. `make status` xem cổng đang chạy, `make down` dừng lần chạy gần nhất.
- Môi trường GPU (train, viewer GPU, run trên máy): `conda env create -f envs/main.yml && conda activate drive2sim && bash envs/install_main.sh` (lỗi giữa chừng thì sửa rồi `--from <bước>`). Notebook Kaggle chạy đúng script này. Đổi phiên bản trong script thì sửa cả `requirements.txt` và `EXPECTED` trong `drive2sim/engine/environment.py` (test `test_expected_versions_match_env_files` kiểm). Chạy Studio bằng môi trường đó: `make studio CONDA_ENV=drive2sim`. `.venv` của `make setup` vẫn dùng cho test và lint (không cần GPU).
- Dữ liệu mẫu để xem giao diện khi chưa có kết quả thật: thêm `DEMO=1` (`make dev DEMO=1`, `make studio DEMO=1`, hay `make demo`). Dữ liệu giả do `python -m drive2sim.scripts.make_demo` tạo một lần ở `.demo/` (git-ignore) và được giữ lại; bỏ `DEMO=1` là về dữ liệu thật, `FORCE=1` để tạo lại. Studio hiện nhãn "Dữ liệu mẫu" trên thanh trên; lệnh `kaggle` và `colab` được thay bằng lệnh giả nên tab Kaggle, Colab không gửi gì đi. Thêm loại kết quả mới vào tool thì thêm cả vào dữ liệu mẫu.

## 6. Branch và commit

- Branch chính của nhóm: `drive2sim`. Branch làm việc đặt `<loại>/<việc>` với loại là `feat`, `fix`, `docs`, `refactor`, `exp` (thí nghiệm), ví dụ `feat/engine-registry`, `exp/fastgs-nuscenes`; gộp vào `drive2sim` bằng merge thường, không rebase lịch sử đã đẩy lên.
- Commit và push bằng tài khoản `awun0105` (`quanlam0105@gmail.com`).
- Dòng đầu commit: `<phạm vi>: <việc đã làm>` với phạm vi là thư mục cấp một trong package hoặc `docs`, `tests`, `envs`, ví dụ `core: add job and result schemas`, `docs: define Studio API`. Nhiều phạm vi thì tách commit.
- Mỗi commit một việc. Di chuyển file (chỉ đổi chỗ và sửa import) tách riêng khỏi thay đổi hành vi.
- Commit message tiếng Anh (mục 10.6 của CONTRACTS): dòng đầu ngắn; phần thân ghi lý do, cách kiểm tra, và những gì chưa kiểm tra.
- Không commit dữ liệu (`data/`, `cache/`, `runs/`), checkpoint, token (`kaggle.json`), `.venv/`.

## 7. License

- Branch `main` có code AGPL-3.0 (patch 3DGEER của Bosch, FastGS), nên toàn bộ package phân phối theo AGPL.
- Khi thêm thư viện hoặc mã của bên khác: kiểm tra license, ghi vào `ARCHITECTURE.md` mục 9, giữ nguyên ghi chú bản quyền gốc.
