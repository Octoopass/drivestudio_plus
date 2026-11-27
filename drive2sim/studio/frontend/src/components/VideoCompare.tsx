// Trình phát so sánh video đánh giá của một run (SPEC mục 8C): ảnh gốc, ảnh render, bản đồ sai
// khác, độ sâu và các lớp (nền, xe, người...) của cùng khung hình, phát đồng bộ; xem cả khung ghép
// hoặc từng camera; so hai kênh bằng thanh trượt hoặc đặt nhiều kênh cạnh nhau.
//
// Video do đánh giá cuối của stage train ghi: mỗi kênh một file, mọi camera ghép chung một khung
// (engine/videos.py). Khung của từng camera lấy từ layout.json; video bị imageio co giãn cho kích
// thước chia hết 16, nên mọi phép cắt tính theo tỉ lệ của khung ghép, không theo pixel của video.
import { Pause, Play, SkipBack, SkipForward } from "lucide-react";
import { type CSSProperties, type KeyboardEvent, type PointerEvent, type ReactNode, type SyntheticEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { seg, useApi } from "../api/client";
import type { List, VideoLayout, VideoSet, VideoTile } from "../api/types";
import { useI18n } from "../i18n";
import { Empty, ErrorBox, Loading, Note, Segmented, turbo } from "./ui";

type Mode = "wipe" | "grid";
const ALL = "__all__";
const RATES = [0.25, 0.5, 1];
// Lệch quá ngần này (giây) so với video chính thì kéo video phụ về cho khớp.
const MAX_DRIFT_S = 0.04;
// Mốc màu inferno (matplotlib) để vẽ chú thích bản đồ sai khác.
const INFERNO = ["#000004", "#1b0c41", "#4a0c6b", "#781c6d", "#a52c60", "#cf4446", "#ed6925", "#fb9b06", "#f7d13d", "#fcffa4"];

function channelLabel(t: (key: string) => string, channel: string): string {
  const key = `video.channel.${channel}`;
  const text = t(key);
  return text === key ? channel : text;
}

/** Một ô video: cả khung ghép, hoặc chỉ khung của một camera (cắt theo tỉ lệ khung ghép). */
function Tile({
  src,
  layout,
  box,
  register,
  onMeta,
  style,
}: {
  src: string;
  layout: VideoLayout | null;
  box: VideoTile | null;
  register: (el: HTMLVideoElement | null) => void;
  onMeta: (video: HTMLVideoElement) => void;
  style?: CSSProperties;
}) {
  const meta = (event: SyntheticEvent<HTMLVideoElement>) => onMeta(event.currentTarget);
  if (!layout || !box) {
    return (
      <div className="vc-tile" style={{ ...style, aspectRatio: layout ? `${layout.width} / ${layout.height}` : undefined }}>
        <video ref={register} src={src} muted playsInline preload="auto" className="vc-full" onLoadedMetadata={meta} />
      </div>
    );
  }
  const videoStyle: CSSProperties = {
    width: `${(layout.width / box.width) * 100}%`,
    height: `${(layout.height / box.height) * 100}%`,
    left: `${(-box.x / box.width) * 100}%`,
    top: `${(-box.y / box.height) * 100}%`,
  };
  return (
    <div className="vc-tile" style={{ ...style, aspectRatio: `${box.width} / ${box.height}` }}>
      <video ref={register} src={src} muted playsInline preload="auto" className="vc-crop" style={videoStyle} onLoadedMetadata={meta} />
    </div>
  );
}

function Legend({ layout, channels }: { layout: VideoLayout | null; channels: string[] }) {
  const { t } = useI18n();
  const items: ReactNode[] = [];
  if (channels.includes("rgb_error_maps") && layout?.error_map) {
    const grey = Math.round(layout.error_map.invalid_grey * 255);
    items.push(
      <div key="error" className="vc-legend">
        <span>{t("video.legend.error")}</span>
        <span className="vc-bar" style={{ background: `linear-gradient(90deg, ${INFERNO.join(", ")})` }} />
        <span className="small muted">{`0 – ${Math.round(layout.error_map.max * 255)}/255`}</span>
        <span className="vc-swatch" style={{ background: `rgb(${grey}, ${grey}, ${grey})` }} />
        <span className="small muted">{t("video.legend.invalid")}</span>
      </div>,
    );
  }
  if (channels.some((c) => c.endsWith("depths")) && layout?.depth) {
    // depth_visualizer: gần thì màu cuối của turbo (đỏ), xa thì màu đầu (xanh), theo log.
    const stops = Array.from({ length: 9 }, (_, i) => turbo(1 - i / 8));
    items.push(
      <div key="depth" className="vc-legend">
        <span>{t("video.legend.depth")}</span>
        <span className="vc-bar" style={{ background: `linear-gradient(90deg, ${stops.join(", ")})` }} />
        <span className="small muted">{`${layout.depth.min_m} – ${layout.depth.max_m} m (log)`}</span>
      </div>,
    );
  }
  return items.length ? <div className="vc-legends">{items}</div> : null;
}

export function VideoCompare({ runId }: { runId: string }) {
  const { t } = useI18n();
  const sets = useApi<List<VideoSet>>(`/api/runs/${seg(runId)}/video_sets`);
  const [setId, setSetId] = useState<string | null>(null);
  const [mode, setMode] = useState<Mode>("wipe");
  const [left, setLeft] = useState("gt_rgbs");
  const [right, setRight] = useState("rgbs");
  const [gridChannels, setGridChannels] = useState<string[]>(["gt_rgbs", "rgbs", "rgb_error_maps"]);
  const [camera, setCamera] = useState(ALL);
  const [split, setSplit] = useState(50);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [rate, setRate] = useState(1);
  const videos = useRef(new Map<string, HTMLVideoElement>());

  const items = sets.data?.items ?? [];
  const current = items.find((s) => s.id === setId) ?? items[0] ?? null;
  const channels = useMemo(() => Object.keys(current?.channels ?? {}), [current]);
  const layout = current?.layout ?? null;
  const fps = layout?.fps ?? null;
  const box = camera === ALL ? null : (layout?.cameras.find((c) => c.name === camera) ?? null);
  const shown = useMemo(
    () => (mode === "wipe" ? [left, right] : gridChannels).filter((c) => channels.includes(c)),
    [mode, left, right, gridChannels, channels],
  );

  // Kênh đang chọn không có trong bộ mới thì về kênh có sẵn.
  useEffect(() => {
    if (!channels.length) return;
    if (!channels.includes(left)) setLeft(channels.includes("gt_rgbs") ? "gt_rgbs" : channels[0]);
    if (!channels.includes(right)) setRight(channels.includes("rgbs") ? "rgbs" : channels[Math.min(1, channels.length - 1)]);
    if (camera !== ALL && !layout?.cameras.some((c) => c.name === camera)) setCamera(ALL);
  }, [channels, left, right, camera, layout]);

  const register = useCallback(
    (key: string) => (el: HTMLVideoElement | null) => {
      if (el) {
        videos.current.set(key, el);
        el.playbackRate = rate;
      } else videos.current.delete(key);
    },
    [rate],
  );
  const all = () => [...videos.current.values()];
  const master = () => all()[0] ?? null;

  const seek = (seconds: number) => {
    const target = Math.max(0, Math.min(duration || seconds, seconds));
    all().forEach((v) => (v.currentTime = target));
    setTime(target);
  };
  const toggle = () => {
    if (playing) {
      all().forEach((v) => v.pause());
      setPlaying(false);
    } else {
      const start = master()?.ended ? 0 : time;
      all().forEach((v) => {
        v.currentTime = start;
        void v.play().catch(() => undefined);
      });
      setPlaying(true);
    }
  };
  const step = (frames: number) => {
    all().forEach((v) => v.pause());
    setPlaying(false);
    // Đặt vào giữa khung để trình duyệt hiện đúng khung đó, không phải khung kề bên.
    const frameTime = fps ? 1 / fps : 0.1;
    const index = Math.round(time / frameTime) + frames;
    seek(Math.max(0, index) * frameTime + frameTime / 2);
  };

  // Video chính dẫn nhịp; video phụ lệch quá MAX_DRIFT_S thì kéo về.
  useEffect(() => {
    if (!playing) return;
    let handle = 0;
    const tick = () => {
      const lead = master();
      if (lead) {
        setTime(lead.currentTime);
        all().forEach((v) => {
          if (v !== lead && Math.abs(v.currentTime - lead.currentTime) > MAX_DRIFT_S) v.currentTime = lead.currentTime;
        });
        if (lead.ended) setPlaying(false);
      }
      handle = requestAnimationFrame(tick);
    };
    handle = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(handle);
  }, [playing]);

  useEffect(() => all().forEach((v) => (v.playbackRate = rate)), [rate]);
  // Đổi bộ, kênh hay camera thì dừng; video mới nạp xong tự nhảy tới vị trí đang xem (onMeta).
  useEffect(() => setPlaying(false), [current?.id, shown.join(","), camera]);

  const onMeta = (video: HTMLVideoElement) => {
    if (Number.isFinite(video.duration)) setDuration(video.duration);
    video.playbackRate = rate;
    video.currentTime = time;
  };
  const onKey = (event: KeyboardEvent) => {
    if (event.key === " ") {
      event.preventDefault();
      toggle();
    } else if (event.key === "ArrowRight") step(1);
    else if (event.key === "ArrowLeft") step(-1);
  };
  const dragSplit = (event: PointerEvent<HTMLDivElement>) => {
    if (event.type === "pointermove" && event.buttons !== 1) return;
    const rect = event.currentTarget.getBoundingClientRect();
    setSplit(Math.max(0, Math.min(100, ((event.clientX - rect.left) / rect.width) * 100)));
  };

  if (sets.error) return <ErrorBox error={sets.error} />;
  if (!sets.data) return <Loading rows={3} />;
  if (!current) return <Empty title={t("run.videos.none")} />;

  const frameCount = fps && duration ? Math.round(duration * fps) : null;
  const frameIndex = fps ? Math.min(frameCount ?? Infinity, Math.floor(time * fps) + 1) : null;
  const options = channels.map((c) => ({ value: c, label: channelLabel(t, c) }));
  const missing = ["rgb_error_maps", "depths"].filter((c) => !channels.includes(c));
  const caption = (c: string) => <div className="vc-caption">{channelLabel(t, c)}</div>;
  // Khung cao quá 70% màn hình thì thu hẹp chiều rộng, để thấy được cả thanh điều khiển.
  const aspect = box ? box.width / box.height : layout ? layout.width / layout.height : null;
  const fit: CSSProperties | undefined = aspect ? { maxWidth: `min(100%, calc(70vh * ${aspect.toFixed(4)}))` } : undefined;

  return (
    <div className="vc" tabIndex={0} onKeyDown={onKey}>
      <div className="vc-toolbar">
        {items.length > 1 && (
          <label className="vc-field">
            <span className="small muted">{t("video.set")}</span>
            <select value={current.id} onChange={(e) => setSetId(e.target.value)}>
              {items.map((s) => (
                <option key={s.id} value={s.id}>
                  {`${t(`video.set.${s.set}`)} · ${t("video.step", { step: s.step })}${s.postfix ? ` · ${s.postfix}` : ""}`}
                </option>
              ))}
            </select>
          </label>
        )}
        <Segmented
          label={t("video.mode")}
          value={mode}
          onChange={setMode}
          options={[
            { value: "wipe", label: t("video.mode.wipe") },
            { value: "grid", label: t("video.mode.grid") },
          ]}
        />
        <label className="vc-field">
          <span className="small muted">{t("common.camera")}</span>
          <select value={camera} onChange={(e) => setCamera(e.target.value)} disabled={!layout}>
            <option value={ALL}>{t("video.camera.all")}</option>
            {(layout?.cameras ?? []).map((c) => (
              <option key={c.name} value={c.name}>
                {c.name}
              </option>
            ))}
          </select>
        </label>
        {mode === "wipe" ? (
          <>
            <label className="vc-field">
              <span className="small muted">{t("video.left")}</span>
              <select value={left} onChange={(e) => setLeft(e.target.value)}>
                {options.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="vc-field">
              <span className="small muted">{t("video.right")}</span>
              <select value={right} onChange={(e) => setRight(e.target.value)}>
                {options.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
            </label>
          </>
        ) : (
          <div className="vc-chips" role="group" aria-label={t("video.channels")}>
            {options.map((o) => (
              <button
                key={o.value}
                type="button"
                aria-pressed={gridChannels.includes(o.value)}
                onClick={() =>
                  setGridChannels((list) => (list.includes(o.value) ? list.filter((c) => c !== o.value) : [...list, o.value]))
                }
              >
                {o.label}
              </button>
            ))}
          </div>
        )}
      </div>

      {!layout && <Note>{t("video.no_layout")}</Note>}
      {missing.length > 0 && <Note>{t("video.missing", { channels: missing.map((c) => channelLabel(t, c)).join(", ") })}</Note>}

      {mode === "wipe" ? (
        <div className="vc-stage" style={fit}>
          <div className="vc-wipe" onPointerDown={dragSplit} onPointerMove={dragSplit}>
            {channels.includes(right) && <Tile key={`r-${right}`} src={current.channels[right]} layout={layout} box={box} register={register(`r:${right}`)} onMeta={onMeta} />}
            {channels.includes(left) && (
              <Tile
                key={`l-${left}`}
                src={current.channels[left]}
                layout={layout}
                box={box}
                register={register(`l:${left}`)}
                onMeta={onMeta}
                style={{ position: "absolute", inset: 0, clipPath: `inset(0 ${100 - split}% 0 0)` }}
              />
            )}
            <div className="vc-divider" style={{ left: `${split}%` }} />
            <span className="vc-tag vc-tag-left">{channelLabel(t, left)}</span>
            <span className="vc-tag vc-tag-right">{channelLabel(t, right)}</span>
          </div>
        </div>
      ) : shown.length === 0 ? (
        <Empty title={t("video.pick")} />
      ) : (
        <div className="vc-grid" style={{ ...(shown.length === 1 ? fit : undefined), gridTemplateColumns: `repeat(${Math.min(shown.length, box ? 3 : 2)}, minmax(0, 1fr))` }}>
          {shown.map((c) => (
            <figure key={c}>
              {caption(c)}
              <Tile src={current.channels[c]} layout={layout} box={box} register={register(`g:${c}`)} onMeta={onMeta} />
            </figure>
          ))}
        </div>
      )}

      <div className="vc-transport">
        <button type="button" className="btn sm" onClick={() => step(-1)} title={t("video.prev")} aria-label={t("video.prev")}>
          <SkipBack size={15} />
        </button>
        <button type="button" className="btn sm primary" onClick={toggle} aria-label={playing ? t("video.pause") : t("video.play")}>
          {playing ? <Pause size={15} /> : <Play size={15} />}
        </button>
        <button type="button" className="btn sm" onClick={() => step(1)} title={t("video.next")} aria-label={t("video.next")}>
          <SkipForward size={15} />
        </button>
        <input
          type="range"
          className="vc-seek"
          min={0}
          max={duration || 0}
          step={fps ? 1 / fps : 0.01}
          value={Math.min(time, duration || 0)}
          onChange={(e) => {
            all().forEach((v) => v.pause());
            setPlaying(false);
            seek(Number(e.target.value));
          }}
          aria-label={t("video.seek")}
        />
        <span className="num small">
          {frameIndex !== null && frameCount ? t("video.frame", { index: frameIndex, total: frameCount }) : `${time.toFixed(1)} s`}
        </span>
        <Segmented label={t("video.rate")} value={rate} onChange={setRate} options={RATES.map((r) => ({ value: r, label: `${r}×` }))} />
      </div>
      <Legend layout={layout} channels={shown} />
      <p className="small muted">{t("video.keys")}</p>
    </div>
  );
}
