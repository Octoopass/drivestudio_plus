// Màn hình chạy mới (SPEC mục 5 bước 4–5): chọn cảnh, split, approach, preset và các ô chọn. Mỗi
// lần đổi lựa chọn, gọi /check để biết lựa chọn nào bị khoá (làm mờ kèm lý do) và vẽ trước các
// stage sẽ chạy, stage nào dùng lại cache. Bấm Chạy thì tạo run và mở màn hình của run.
import { useEffect, useState } from "react";
import { ApiError, post, useApi } from "../api/client";
import type { Approach, CheckResult, List, Manifest, OptionValue, RunRecord, SceneDetail } from "../api/types";
import { ErrorBox } from "../components/common";
import { StageRoad } from "../components/StageRoad";
import { useI18n } from "../i18n";
import { go, href } from "../router";

const show = (value: OptionValue) => (typeof value === "boolean" ? (value ? "true" : "false") : String(value));

export function NewRunPage({ sceneId: initialScene, splitId: initialSplit }: { sceneId?: string; splitId?: string }) {
  const { t } = useI18n();
  const scenes = useApi<List<Manifest>>("/api/scenes");
  const approaches = useApi<List<Approach>>("/api/approaches");
  const [sceneId, setSceneId] = useState(initialScene ?? "");
  const scene = useApi<SceneDetail>(sceneId ? `/api/scenes/${encodeURIComponent(sceneId)}` : null);
  const [splitId, setSplitId] = useState(initialSplit ?? "");
  const [approachId, setApproachId] = useState("");
  const [preset, setPreset] = useState("");
  const [options, setOptions] = useState<Record<string, OptionValue>>({});
  const [check, setCheck] = useState<CheckResult | null>(null);
  const [checkError, setCheckError] = useState<ApiError | null>(null);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<ApiError | null>(null);

  const approach = approaches.data?.items.find((a) => a.approach_id === approachId);

  // Chọn sẵn split đầu tiên của cảnh, approach đầu tiên trong danh sách.
  useEffect(() => {
    if (scene.data && !scene.data.splits.includes(splitId)) setSplitId(scene.data.splits[0] ?? "");
  }, [scene.data]);
  useEffect(() => {
    if (!approachId && approaches.data?.items[0]) setApproachId(approaches.data.items[0].approach_id);
  }, [approaches.data, approachId]);
  useEffect(() => setOptions({}), [approachId, preset]);

  // Kiểm tra lại mỗi khi lựa chọn đổi.
  useEffect(() => {
    if (!sceneId || !splitId || !approachId) return;
    let cancelled = false;
    post<CheckResult>(`/api/approaches/${encodeURIComponent(approachId)}/check`, {
      scene_id: sceneId,
      split_id: splitId,
      preset: preset || null,
      options,
    })
      .then((result) => !cancelled && (setCheck(result), setCheckError(null)))
      .catch((err) => !cancelled && (setCheck(null), setCheckError(err as ApiError)));
    return () => {
      cancelled = true;
    };
  }, [sceneId, splitId, approachId, preset, options]);

  const start = async () => {
    setStarting(true);
    setStartError(null);
    try {
      const run = await post<RunRecord>("/api/runs", {
        approach_id: approachId,
        scene_id: sceneId,
        split_id: splitId,
        preset: preset || null,
        options,
      });
      go({ page: "run", runId: run.run_id });
    } catch (err) {
      setStartError(err as ApiError);
      setStarting(false);
    }
  };

  const blockedReason = (option: string, choice: OptionValue) =>
    check?.blocked_choices[option]?.find((b) => b.choice === choice)?.message;
  const decided = check?.options ?? {};

  return (
    <>
      <h1>{t("new.page.title")}</h1>
      <p className="lede">{t("new.page.lede")}</p>

      <div className="form-row">
        <label className="field">
          {t("new.field.scene")}
          <select value={sceneId} onChange={(e) => setSceneId(e.target.value)}>
            <option value="">{t("new.field.choose")}</option>
            {(scenes.data?.items ?? []).map((m) => (
              <option key={m.scene_id}>{m.scene_id}</option>
            ))}
          </select>
        </label>
        <label className="field">
          {t("new.field.split")}
          <select value={splitId} onChange={(e) => setSplitId(e.target.value)} disabled={!scene.data}>
            {(scene.data?.splits ?? []).map((id) => (
              <option key={id}>{id}</option>
            ))}
          </select>
        </label>
        <label className="field">
          {t("new.field.approach")}
          <select value={approachId} onChange={(e) => setApproachId(e.target.value)}>
            {(approaches.data?.items ?? []).map((a) => (
              <option key={a.approach_id} value={a.approach_id}>
                {a.name}
              </option>
            ))}
          </select>
        </label>
        {approach && Object.keys(approach.presets ?? {}).length > 0 && (
          <label className="field">
            {t("new.field.preset")}
            <select value={preset} onChange={(e) => setPreset(e.target.value)}>
              <option value="">{t("new.field.no_preset")}</option>
              {Object.keys(approach.presets ?? {}).map((name) => (
                <option key={name}>{name}</option>
              ))}
            </select>
          </label>
        )}
      </div>

      {scene.data && scene.data.splits.length === 0 && (
        <p>
          {t("new.need.split")} <a href={href({ page: "scene", sceneId })}>{sceneId}</a>
        </p>
      )}

      {approach && Object.keys(approach.options ?? {}).length > 0 && (
        <section>
          <h2>{t("new.options.title")}</h2>
          {Object.entries(approach.options ?? {}).map(([name, spec]) => {
            const reasons = (spec.choices as OptionValue[]).map((c) => blockedReason(name, c)).filter(Boolean);
            return (
              <div className="option" key={name}>
                <div className="name">{name}</div>
                <div className="segmented" role="group" aria-label={name}>
                  {(spec.choices as OptionValue[]).map((choice) => (
                    <button
                      key={show(choice)}
                      aria-pressed={decided[name] === choice}
                      disabled={Boolean(blockedReason(name, choice))}
                      title={blockedReason(name, choice)}
                      onClick={() => setOptions({ ...options, [name]: choice })}
                    >
                      {show(choice)}
                    </button>
                  ))}
                </div>
                {reasons.map((reason) => (
                  <div className="why" key={reason}>
                    {reason}
                  </div>
                ))}
              </div>
            );
          })}
        </section>
      )}

      <ErrorBox error={checkError} />

      {check && (
        <section>
          <h2>{t("new.plan.title")}</h2>
          <StageRoad
            stages={check.stages.map((s) => ({
              stage_id: s.stage_id,
              status: "pending",
              cached: s.cache_hit,
              note: t(s.cache_hit ? "new.plan.cache" : "new.plan.run"),
            }))}
          />
          <p className="muted small">
            {check.lidar_usage.length ? t("new.plan.lidar", { stages: check.lidar_usage.join(", ") }) : t("new.plan.no_lidar")}
          </p>
          <button className="button primary" onClick={start} disabled={starting}>
            {t(starting ? "new.action.starting" : "new.action.start")}
          </button>
          <ErrorBox error={startError} />
        </section>
      )}
    </>
  );
}
