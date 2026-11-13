// Tab Artifact của trang run: nơi lưu kết quả của từng stage (thư mục run, hoặc cache dùng chung
// giữa các run) và các file ở đó, lấy từ GET /api/runs/{id}/artifacts. Nút thư mục mở đúng chỗ đó
// trong trình quản lý file của máy chạy Studio (POST /api/reveal); nút chép đưa đường dẫn đầy đủ vào
// clipboard, dùng khi Studio chạy trên máy khác (SSH) hoặc máy không có giao diện.
import { Copy, ExternalLink, File, Folder, FolderOpen, RefreshCw } from "lucide-react";
import type { ReactNode } from "react";
import { asApiError, post, seg, useApi } from "../api/client";
import { useI18n } from "../i18n";
import { bytes, Empty, ErrorBox, InfoTip, Loading, Status, useToast } from "./ui";

interface Entry {
  name: string;
  is_dir: boolean;
  size_bytes: number;
  path: string;
}

interface StageArtifacts {
  stage_id: string;
  status: string;
  location: "run" | "cache";
  path: string | null;
  path_abs: string | null;
  size_bytes: number;
  files: Entry[];
}

interface Artifacts {
  run_dir: string;
  run_dir_abs: string;
  files: Entry[];
  stages: StageArtifacts[];
}

// File xem được ngay trong trình duyệt qua /files/.
const VIEWABLE = /\.(mp4|png|jpe?g|json|txt|yaml|log)$/i;

// Stage export → loại xuất của POST /api/runs/{id}/export.
const REEXPORT: Record<string, "ply" | "preview"> = { "export.ply_world": "ply", "export.preview": "preview" };

export function RunArtifacts({ runId, onReexport }: { runId: string; onReexport?: (kind: "ply" | "preview") => void }) {
  const { t } = useI18n();
  const toast = useToast();
  const data = useApi<Artifacts>(`/api/runs/${seg(runId)}/artifacts`);

  const reveal = async (path: string) => {
    try {
      const result = await post<{ path: string }>("/api/reveal", { path });
      toast({ kind: "ok", title: t("artifacts.opened"), detail: result.path });
    } catch (err) {
      toast({ kind: "bad", title: t("artifacts.open_failed"), detail: asApiError(err).message });
    }
  };
  const copy = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      toast({ kind: "ok", title: t("artifacts.copied"), detail: text });
    } catch {
      toast({ kind: "bad", title: t("artifacts.copy_failed"), detail: text });
    }
  };
  const actions = (path: string, absolute: string, label: string) => (
    <span className="row" style={{ gap: 2 }}>
      <button className="btn sm ghost icon" title={t("artifacts.open")} aria-label={`${t("artifacts.open")} ${label}`} onClick={() => void reveal(path)}>
        <FolderOpen size={17} />
      </button>
      <button className="btn sm ghost icon" title={t("artifacts.copy")} aria-label={`${t("artifacts.copy")} ${label}`} onClick={() => void copy(absolute)}>
        <Copy size={16} />
      </button>
    </span>
  );

  if (data.error) return <ErrorBox error={data.error} />;
  if (!data.data) return <Loading rows={4} />;
  const root = data.data;
  const base = root.run_dir_abs.slice(0, root.run_dir_abs.length - root.run_dir.length);
  return (
    <div className="stack" style={{ gap: 14 }}>
      <div className="artifact-root">
        <Folder size={18} />
        <code className="grow ellipsis" title={root.run_dir_abs}>
          {root.run_dir_abs}
        </code>
        <InfoTip>{t("artifacts.layout")}</InfoTip>
        {actions(root.run_dir, root.run_dir_abs, root.run_dir)}
      </div>
      <FileList title={t("artifacts.tool_files")} files={root.files} base={base} actions={actions} />
      {root.stages.map((stage) => (
        <section key={stage.stage_id} className="artifact-stage">
          <header className="row wrap" style={{ gap: 8 }}>
            <b>{stage.stage_id}</b>
            <Status value={stage.status} />
            <span className={`tag ${stage.location === "cache" ? "info" : ""}`}>{t(`artifacts.location.${stage.location}`)}</span>
            {stage.location === "cache" && <InfoTip>{t("artifacts.cache_hint")}</InfoTip>}
            <span className="grow" />
            {stage.path && <span className="small muted">{bytes(stage.size_bytes)}</span>}
            {onReexport && REEXPORT[stage.stage_id] && (
              <span className="row" style={{ gap: 2 }}>
                <button className="btn sm ghost" onClick={() => onReexport(REEXPORT[stage.stage_id])}>
                  <RefreshCw size={15} />
                  {t("artifacts.reexport")}
                </button>
                <InfoTip>{t("run.action.reexport_hint")}</InfoTip>
              </span>
            )}
            {stage.path && stage.path_abs && actions(stage.path, stage.path_abs, stage.stage_id)}
          </header>
          {stage.path ? (
            <>
              <code className="small muted ellipsis artifact-path" title={stage.path_abs ?? ""}>
                {stage.path}
              </code>
              <FileList files={stage.files} base={base} actions={actions} />
            </>
          ) : (
            <p className="small muted">{t("artifacts.none")}</p>
          )}
        </section>
      ))}
      {root.stages.length === 0 && <Empty title={t("artifacts.none")} />}
    </div>
  );
}

function FileList({
  title,
  files,
  base,
  actions,
}: {
  title?: string;
  files: Entry[];
  base: string;
  actions: (path: string, absolute: string, label: string) => ReactNode;
}) {
  const { t } = useI18n();
  if (!files.length) return null;
  return (
    <div className="artifact-files">
      {title && <div className="small muted" style={{ marginBottom: 4 }}>{title}</div>}
      <ul>
        {files.map((file) => (
          <li key={file.path}>
            {file.is_dir ? <Folder size={16} className="muted" /> : <File size={16} className="muted" />}
            <span className="grow ellipsis">{file.name}{file.is_dir ? "/" : ""}</span>
            <span className="small muted nowrap">{bytes(file.size_bytes)}</span>
            {!file.is_dir && VIEWABLE.test(file.name) ? (
              <a className="btn sm ghost icon" href={`/files/${file.path}`} target="_blank" rel="noreferrer" title={t("artifacts.view")} aria-label={`${t("artifacts.view")} ${file.name}`}>
                <ExternalLink size={16} />
              </a>
            ) : (
              <span className="btn sm ghost icon" aria-hidden="true" style={{ visibility: "hidden" }} />
            )}
            {actions(file.path, base + file.path, file.name)}
          </li>
        ))}
      </ul>
    </div>
  );
}
