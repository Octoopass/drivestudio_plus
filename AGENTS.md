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
| `CONTRACTS.md` | Định dạng file, interface của stage, quy tắc đặt tên và CLI (mục 10) |

Quy tắc:
- **Trước khi viết code có quyết định về thiết kế** (thêm module, đổi định dạng file, đổi interface, thêm lệnh CLI, chọn thư viện): đọc kỹ phần liên quan của cả ba tài liệu. Code phải theo docs.
- **Nếu cần làm khác docs**: không âm thầm làm khác. Sửa docs trước hoặc cùng lúc, ghi rõ lý do, ưu và nhược điểm theo mẫu ở `ARCHITECTURE.md` mục 2.
- **Khi code có quyết định mới về kiến trúc**: cập nhật cả ba tài liệu cho đồng bộ **trong cùng commit** với code. Gồm: cây thư mục (mục 3), bảng chỗ đặt code (mục 3.1), bảng tiến độ (mục 7.1), định dạng file và tên (CONTRACTS).
- `tests/drive2sim/core/test_contracts_examples.py` đọc mọi ví dụ JSON/YAML có `schema_version` trong `CONTRACTS.md` và kiểm tra theo schema. Sửa định dạng file thì sửa cả ví dụ trong docs và schema trong `drive2sim/core/schema.py`, nếu không test sẽ báo lỗi.
- Thông tin về paper, repo, số liệu: kiểm tra trong mã nguồn hoặc tài liệu gốc rồi mới ghi vào docs; chỗ nào chưa kiểm tra thì ghi rõ "chưa kiểm tra".

## 3. Đặt code ở đâu, đặt tên ra sao

- Chỗ đặt file mới: theo bảng ở `ARCHITECTURE.md` mục 3.1. Không thêm thư mục mã mới ở gốc repo; ở gốc chỉ có `docs/`, `tests/`, `third_party/`, `envs/` và file cấu hình.
- Tên định danh, file, trường, tham số, lệnh CLI: theo `CONTRACTS.md` mục 10.
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

- Mọi module, lớp, hàm và phương thức công khai có docstring **bằng tiếng Việt**, theo quy ước PEP 257: dòng đầu là một câu tóm tắt, rồi một dòng trống, rồi phần giải thích.
- Viết dễ hiểu: hàm làm gì, vì sao cần, dùng khi nào; nhắc tới mục docs liên quan nếu có (ví dụ `CONTRACTS mục 5.2`).
- Giải thích từng tham số, giá trị trả về và lỗi có thể ném ra theo mẫu:

```python
def make_cache_key(
    scene_id: str, stage_id: str, version: str, params: dict, input_keys: Iterable[str]
) -> str:
    """Tạo khoá cache cho kết quả của một stage có cache (CONTRACTS mục 5.2).

    Khoá chỉ băm tham số và khoá của các artifact đầu vào, không băm nội dung dữ liệu,
    vì dữ liệu chuẩn không đổi sau khi ingest.

    Args:
        scene_id: Định danh cảnh, dạng ``<dataset>_<sequence_id>``.
        stage_id: Định danh stage, dạng ``<nhóm>.<tên>``.
        version: Phiên bản của stage, dạng ``MAJOR.MINOR.PATCH``.
        params: Tham số đã gộp của stage.
        input_keys: Khoá cache của các artifact đầu vào; thứ tự không quan trọng.

    Returns:
        Khoá dạng ``<scene_id>/<stage_id>@<version>-<hash8>``.

    Raises:
        ValueError: Nếu một định danh sai định dạng.
    """
```

- Docstring theo quy ước Google (`Args:`, `Returns:`, `Raises:`), ruff kiểm tra định dạng.
- Hàm test đặt tên mô tả rõ hành vi nên không bắt buộc docstring; module test vẫn có docstring.
- Tên biến, hàm, lớp và thông báo lỗi trong code giữ bằng tiếng Anh; docstring và comment bằng tiếng Việt.

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
- Chạy test: `.venv/bin/python -m pytest` (môi trường nhẹ ở local, cài bằng `pip install -e ".[dev]"`).
- Trước khi commit: `uvx ruff format drive2sim tests`, `uvx ruff check drive2sim tests`, rồi chạy test.
- Kiểm tra Python 3.9: `uv run --python 3.9 --no-project --with "pydantic>=2.5,<3" --with "pyyaml>=6" --with "pytest>=7" python -m pytest -q`.
- Thay đổi hành vi ở `drive2sim/recon/` không chạy được trên máy không có GPU: ghi rõ "chưa chạy thật" trong commit và docs, rồi kiểm tra hồi quy trên Kaggle (train lại cùng cảnh, so PSNR) theo `ARCHITECTURE.md` mục 7.3 và 10.

## 6. Commit

- Commit và push bằng tài khoản `awun0105` (`quanlam0105@gmail.com`).
- Mỗi commit một việc. Di chuyển file (chỉ đổi chỗ và sửa import) tách riêng khỏi thay đổi hành vi.
- Commit message tiếng Anh: dòng đầu ngắn, mô tả việc đã làm; phần thân ghi lý do, cách kiểm tra, và những gì chưa kiểm tra.
- Không commit dữ liệu (`data/`, `cache/`, `runs/`), checkpoint, token (`kaggle.json`), `.venv/`.

## 7. License

- Branch `main` có code AGPL-3.0 (patch 3DGEER của Bosch, FastGS), nên toàn bộ package phân phối theo AGPL.
- Khi thêm thư viện hoặc mã của bên khác: kiểm tra license, ghi vào `ARCHITECTURE.md` mục 9, giữ nguyên ghi chú bản quyền gốc.
