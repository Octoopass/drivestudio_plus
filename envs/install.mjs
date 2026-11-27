#!/usr/bin/env node
// Cài và bật drive2sim bằng Node.js, dùng được trên máy chủ không màn hình (docs/install.md):
//
//   npx github:thaianh20021/drive2sim                    # cài (Linux GPU: đủ; máy khác: Studio)
//   npx github:thaianh20021/drive2sim install --studio-only --workdir /data/d2s
//   npx github:thaianh20021/drive2sim start --port 8765  # bật Studio ở nền, log ~/.drive2sim/studio.log
//   npx github:thaianh20021/drive2sim status | stop
//
// Lệnh này chỉ lấy mã nguồn (git clone, hoặc gói nén của bản phát hành khi máy không có git) vào
// ~/.drive2sim/src rồi chạy envs/install.sh (Linux, macOS) hoặc envs/install.ps1 (Windows);
// mọi bước cài nằm ở hai script đó. Cần Node.js 18 trở lên, không cần gói npm nào khác.
import { spawn, spawnSync } from "node:child_process";
import { createWriteStream, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { homedir, hostname, platform, userInfo } from "node:os";
import { dirname, join } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { fileURLToPath } from "node:url";

const REPO = "https://github.com/thaianh20021/drive2sim";
const HERE = dirname(fileURLToPath(import.meta.url));
const VERSION = JSON.parse(readFileSync(join(HERE, "..", "package.json"), "utf8")).version;
const WINDOWS = platform() === "win32";
const HOME = process.env.D2S_HOME || join(homedir(), ".drive2sim");
const PID_FILE = join(HOME, "studio.pid");
const LOG_FILE = join(HOME, "studio.log");

const HELP = `drive2sim ${VERSION}: cài và bật drive2sim Studio

Cách dùng: npx github:thaianh20021/drive2sim [lệnh] [tuỳ chọn]

Lệnh:
  install (mặc định)  lấy mã nguồn rồi cài
  start               bật Studio ở nền (máy chủ không màn hình), in cách mở qua SSH
  stop                tắt Studio đã bật bằng start
  status              Studio có đang chạy không

Tuỳ chọn:
  --full              cài đủ để train trên máy (Linux, GPU NVIDIA); mặc định tự chọn
  --studio-only       chỉ Studio + CLI (chạy run trên Kaggle, Colab)
  --no-nurec          không cài Instant NuRec
  --workdir DIR       thư mục làm việc (mặc định ~/drive2sim-work)
  --port N            cổng Studio (mặc định 8765)
  --ref REF           tag hoặc nhánh của mã nguồn (mặc định v${VERSION}; main là bản mới nhất)
  --src DIR           dùng mã nguồn có sẵn ở DIR, không tải (mặc định tải vào ~/.drive2sim/src)
  --start             bật Studio ở nền sau khi cài
  --host ADDR         Studio nghe trên ADDR thay cho 127.0.0.1 (IP mạng riêng như Tailscale;
                      Studio không có đăng nhập, không dùng IP công khai)
  --allow-host NAME   nhận thêm tên máy NAME trong Host/Origin (lặp được), ví dụ IP Tailscale khi
                      tailscaled chạy userspace-networking
`;

function parse(argv) {
  const opts = { command: "install", port: "8765", ref: `v${VERSION}`, src: join(HOME, "src"), allowHosts: [] };
  const rest = [...argv];
  if (rest[0] && !rest[0].startsWith("-")) opts.command = rest.shift();
  while (rest.length) {
    const flag = rest.shift();
    const value = () => {
      if (!rest.length) fail(`${flag} cần một giá trị`);
      return rest.shift();
    };
    if (flag === "--full") opts.mode = "full";
    else if (flag === "--studio-only") opts.mode = "studio";
    else if (flag === "--no-nurec") opts.noNurec = true;
    else if (flag === "--start") opts.start = true;
    else if (flag === "--workdir") opts.workdir = value();
    else if (flag === "--port") opts.port = value();
    else if (flag === "--ref") opts.ref = value();
    else if (flag === "--host") opts.host = value();
    else if (flag === "--allow-host") opts.allowHosts.push(value());
    else if (flag === "--src") {
      opts.src = value();
      opts.srcGiven = true;
    }
    else if (flag === "-h" || flag === "--help") opts.command = "help";
    else fail(`không hiểu tham số ${flag} (xem --help)`);
  }
  return opts;
}

function fail(message) {
  console.error(`drive2sim: ${message}`);
  process.exit(2);
}

function run(command, args, options = {}) {
  console.log(`$ ${command} ${args.join(" ")}`);
  const done = spawnSync(command, args, { stdio: "inherit", ...options });
  if (done.error) fail(`không chạy được ${command}: ${done.error.message}`);
  return done.status ?? 1;
}

function has(command) {
  const probe = spawnSync(command, ["--version"], { stdio: "ignore", shell: WINDOWS });
  return !probe.error && probe.status === 0;
}

async function download(url, file) {
  console.log(`tải ${url}`);
  const response = await fetch(url, { redirect: "follow" });
  if (!response.ok) fail(`tải ${url} lỗi: HTTP ${response.status}`);
  await pipeline(Readable.fromWeb(response.body), createWriteStream(file));
}

// Mã nguồn vào opts.src: dùng lại bản có sẵn, git clone, hoặc gói nén của bản phát hành.
// Thư mục truyền bằng --src dùng nguyên trạng (không fetch, checkout: có thể là bản clone của người
// dùng có thay đổi chưa commit); chỉ thư mục mặc định ~/.drive2sim/src do lệnh này quản lý.
async function fetchSource(opts) {
  if (opts.srcGiven) {
    if (!existsSync(join(opts.src, "envs", "install.sh"))) fail(`${opts.src} không có envs/install.sh`);
    console.log(`dùng mã nguồn ở ${opts.src}`);
    return;
  }
  if (existsSync(join(opts.src, "envs", "install.sh")) && !existsSync(join(opts.src, ".git"))) {
    console.log(`dùng mã nguồn có sẵn ở ${opts.src} (gói nén; xoá thư mục này để tải bản khác)`);
    return;
  }
  if (has("git")) {
    if (existsSync(join(opts.src, ".git"))) {
      run("git", ["-C", opts.src, "fetch", "--depth", "1", "origin", opts.ref]);
      if (run("git", ["-C", opts.src, "checkout", "--force", "FETCH_HEAD"])) fail("checkout lỗi");
      run("git", ["-C", opts.src, "submodule", "update", "--init", "--depth", "1"]);
      return;
    }
    mkdirSync(dirname(opts.src), { recursive: true });
    const code = run("git", [
      "clone", "--depth", "1", "--branch", opts.ref, "--recurse-submodules", "--shallow-submodules",
      `${REPO}.git`, opts.src,
    ]);
    if (code === 0) return;
    rmSync(opts.src, { recursive: true, force: true });
    console.log("git clone lỗi: thử gói nén của bản phát hành");
  }
  // tên file như bản phát hành (docs/releases): Windows là .zip; tar của Windows 10+ giải được zip
  const os = WINDOWS ? "windows" : platform() === "darwin" ? "macos" : "linux";
  const name = `drive2sim-${opts.ref}-${os}${WINDOWS ? ".zip" : ".tar.gz"}`;
  mkdirSync(opts.src, { recursive: true });
  const archive = join(HOME, name);
  await download(`${REPO}/releases/download/${opts.ref}/${name}`, archive);
  if (run("tar", ["-xf", archive, "-C", opts.src, "--strip-components", "1"])) fail("giải nén lỗi");
  rmSync(archive, { force: true });
}

async function install(opts) {
  if (opts.mode === "full" && WINDOWS) {
    fail("train trên máy cần Linux có GPU NVIDIA: trên Windows cài WSL2 rồi chạy lệnh này trong WSL, hoặc bỏ --full");
  }
  mkdirSync(HOME, { recursive: true });
  await fetchSource(opts);
  let code;
  if (WINDOWS) {
    const args = ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", join(opts.src, "envs", "install.ps1"), "-Port", opts.port];
    if (opts.workdir) args.push("-Workdir", opts.workdir);
    code = run("powershell", args);
  } else {
    const args = [join(opts.src, "envs", "install.sh"), "--port", opts.port];
    if (opts.mode === "full") args.push("--full");
    if (opts.mode === "studio") args.push("--studio-only");
    if (opts.noNurec) args.push("--no-nurec");
    if (opts.workdir) args.push("--workdir", opts.workdir);
    code = run("bash", args);
  }
  if (code !== 0) process.exit(code);
  if (opts.start) await start(opts);
}

function launcher() {
  return WINDOWS ? join(HOME, "bin", "drive2sim-studio.cmd") : join(HOME, "bin", "drive2sim-studio");
}

function running() {
  if (!existsSync(PID_FILE)) return null;
  const pid = Number(readFileSync(PID_FILE, "utf8").trim());
  try {
    process.kill(pid, 0);
    return pid;
  } catch {
    return null;
  }
}

async function start(opts) {
  if (!existsSync(launcher())) fail("chưa cài: chạy install trước");
  const pid = running();
  if (pid) {
    console.log(`Studio đang chạy (pid ${pid}); log ${LOG_FILE}`);
    return;
  }
  const env = { ...process.env, D2S_PORT: opts.port };
  if (opts.workdir) env.D2S_WORKDIR = opts.workdir;
  // tham số thêm của launcher đi thẳng vào d2s studio
  const extra = [...(opts.host ? ["--host", opts.host] : []), ...opts.allowHosts.flatMap((name) => ["--allow-host", name])];
  const log = createWriteStream(LOG_FILE, { flags: "a" });
  await new Promise((resolve) => log.on("open", resolve));
  const child = WINDOWS
    ? spawn("cmd.exe", ["/c", launcher(), ...extra], { detached: true, stdio: ["ignore", log, log], env, windowsHide: true })
    : spawn(launcher(), extra, { detached: true, stdio: ["ignore", log, log], env });
  child.unref();
  writeFileSync(PID_FILE, String(child.pid));
  const url = studioUrl(opts);
  for (let i = 0; i < 30; i++) {
    await new Promise((resolve) => setTimeout(resolve, 1000));
    if (await alive(url)) break;
  }
  const ok = await alive(url);
  console.log(ok ? `Studio chạy ở ${url} (pid ${child.pid}); log ${LOG_FILE}` : `Studio chưa trả lời ở ${url}: xem log ${LOG_FILE}`);
  if (opts.host) console.log(`Máy trong cùng mạng riêng mở thẳng ${url}`);
  else
    console.log(`Máy chủ không màn hình: từ máy của bạn chạy
  ssh -N -L ${opts.port}:127.0.0.1:${opts.port} ${userInfo().username}@${hostname()}
rồi mở ${url}`);
  if (!ok) process.exit(1);
}

function studioUrl(opts) {
  const host = !opts.host || opts.host === "0.0.0.0" ? "127.0.0.1" : opts.host;
  return `http://${host.includes(":") ? `[${host}]` : host}:${opts.port}`;
}

async function alive(url) {
  try {
    const response = await fetch(url, { signal: AbortSignal.timeout(2000) });
    return response.ok;
  } catch {
    return false;
  }
}

function stop() {
  const pid = running();
  if (!pid) {
    console.log("Studio không chạy");
    return;
  }
  try {
    if (WINDOWS) spawnSync("taskkill", ["/pid", String(pid), "/t", "/f"], { stdio: "ignore" });
    else process.kill(-pid, "SIGTERM");
  } catch {
    process.kill(pid, "SIGTERM");
  }
  rmSync(PID_FILE, { force: true });
  console.log(`đã tắt Studio (pid ${pid})`);
}

async function status(opts) {
  const pid = running();
  const url = studioUrl(opts);
  const ok = await alive(url);
  console.log(pid ? `Studio: pid ${pid}, ${ok ? `trả lời ở ${url}` : `chưa trả lời ở ${url}`}` : "Studio không chạy (theo start)");
  console.log(`log: ${LOG_FILE}`);
}

const opts = parse(process.argv.slice(2));
if (opts.command === "help") console.log(HELP);
else if (opts.command === "install") await install(opts);
else if (opts.command === "start") await start(opts);
else if (opts.command === "stop") stop();
else if (opts.command === "status") await status(opts);
else fail(`không có lệnh ${opts.command} (xem --help)`);
