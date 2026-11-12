// Chạy mới (SPEC mục 5 bước 4–5): chọn cảnh, split, approach, preset và các ô chọn; chọn chạy trên
// máy hay trên Kaggle. Mỗi lần đổi lựa chọn, gọi /check để biết lựa chọn nào bị khoá (gạch và ghi
// lý do) và vẽ trước làn stage sẽ chạy, stage nào dùng lại cache. Bấm chạy thì mở màn hình run.
import { Cloud, Cpu, Database, Play } from "lucide-react";
import { useEffect, useState } from "react";
import { asApiError, post, seg, useApi, type ApiError } from "../api/client";
import type { Approach, CheckResult, List, Manifest, OptionValue, RunRecord, SceneDetail } from "../api/types";
import { StageRoad } from "../components/StageRoad";
import { ErrorBox, Note, PageHead, Panel, Segmented, Skeleton } from "../components/ui";
import { useI18n } from "../i18n";
import { go, href } from "../router";

const show = (value: OptionValue) => (typeof value === "boolean" ? (value ? "true" : "false") : String(value));

export function NewRunPage({ sceneId: initialScene, splitId: initialSplit }: { sceneId?: string; splitId?: string }) {
  const { t } = useI18n();
  const scenes = useApi<List<Manifest>>("/api/scenes");
  const approaches = useApi<List<Approach>>("/api/approaches");
  const [sceneId, setSceneId] = useState(initialScene ?? "");
  const scene = useApi<SceneDetail>(sceneId ? `/api/scenes/${seg(sceneId)}` : null);
  const [splitId, setSplitId] = useState(initialSplit ?? "");
  const [approachId, setApproachId] = useState("");
  const [preset, setPreset] = useState("");
  const [options, setOptions] = useState<Record<string, OptionValue>>({});
  const [executor, setExecutor] = useState<"local" | "kaggle">("local");
  const [check, setCheck] = useState<CheckResult | null>(null);
  const [checkError, setCheckError] = useState<ApiError | null>(null);
  const [checking, setChecking] = useState(false);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<ApiError | null>(null);

  const approach = approaches.data?.items.find((a) => a.approach_id === approachId);

  // Chọn sẵn cảnh đầu tiên, split đầu tiên của cảnh, approach đầu tiên.
  useEffect(() => {
    if (!sceneId && scenes.data?.items[0]) setSceneId(scenes.data.items[0].scene_id);
  }, [scenes.data, sceneId]);
  useEffect(() => {
    if (scene.data && !scene.data.splits.includes(splitId)) setSplitId(scene.data.splits[0] ?? "");
  }, [scene.data]);
  useEffect(() => {
    if (!approachId && approaches.data?.items[0]) setApproachId(approaches.data.items[0].approach_id);
  }, [approaches.data, approachId]);
  useEffect(() => setOptions({}), [approachId, preset]);

  // Kiểm tra lại mỗi khi lựa chọn đổi.
  useEffect(() => {
    if (!sceneId || !splitId || !approachId) {
      setCheck(null);
      return;
    }
    let cancelled = false;
    setChecking(true);
    post<CheckResult>(`/api/approaches/${seg(approachId)}/check`, { scene_id: sceneId, split_id: splitId, preset: preset || null, options })
      .then((result) => !cancelled && (setCheck(result), setCheckError(null)))
      .catch((err) => !cancelled && (setCheck(null), setCheckError(asApiError(err))))
      .finally(() => !cancelled && setChecking(false));
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
        executor,
      });
      go({ page: "run", runId: run.run_id });
    } catch (err) {
      setStartError(asApiError(err));
      setStarting(false);
    }
  };

  const blockedReason = (option: string, choice: OptionValue) => check?.blocked_choices[option]?.find((b) => b.choice === choice)?.message;
  const decided = check?.options ?? {};
  const cached = check?.stages.filter((s) => s.cache_hit).length ?? 0;
  const presets = Object.keys(approach?.presets ?? {});

  return (
    <div className="page">
      <PageHead title={t("new.page.title")} lede={t("new.page.lede")} />
      <div className="grid-main">
        <div className="stack">
          <Panel title={t("new.section.data")}>
            <div className="form-grid">
              <label className="field">
                <span>{t("new.field.scene")}</span>
                <select value={sceneId} onChange={(e) => setSceneId(e.target.value)}>
                  {!sceneId && <option value="">{t("new.field.choose")}</option>}
                  {(scenes.data?.items ?? []).map((m) => (
                    <option key={m.scene_id}>{m.scene_id}</option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>{t("new.field.split")}</span>
                <select value={splitId} onChange={(e) => setSplitId(e.target.value)} disabled={!scene.data?.splits.length}>
                  {(scene.data?.splits ?? []).map((id) => (
                    <option key={id}>{id}</option>
                  ))}
                </select>
              </label>
            </div>
            {scenes.data?.total === 0 && (
              <div style={{ marginTop: 12 }}>
                <Note>
                  {t("new.need.scene")} <a href={href({ page: "scenes" })}>{t("nav.menu.scenes")}</a>
                </Note>
              </div>
            )}
            {scene.data && scene.data.splits.length === 0 && (
              <div style={{ marginTop: 12 }}>
                <Note kind="warn">
                  {t("new.need.split")} <a href={href({ page: "scene", sceneId })}>{sceneId}</a>
                </Note>
              </div>
            )}
          </Panel>

          <Panel title={t("new.section.approach")}>
            {!approaches.data && <Skeleton height={64} />}
            <div className="choice-list">
              {(approaches.data?.items ?? []).map((a) => (
                <button key={a.approach_id} className="choice" aria-pressed={a.approach_id === approachId} onClick={() => setApproachId(a.approach_id)}>
                  <span className="name">{a.name}</span>
                  <span className="desc">{t("new.approach.meta", { codebase: a.codebase, level: a.level, stages: a.stages.length })}</span>
                </button>
              ))}
            </div>
            {approach && (presets.length > 0 || Object.keys(approach.options ?? {}).length > 0) && (
              <div style={{ marginTop: 14 }}>
                {presets.length > 0 && (
                  <div className="option-row">
                    <div className="name">{t("new.field.preset")}</div>
                    <Segmented<string>
                      label={t("new.field.preset")}
                      value={preset}
                      onChange={setPreset}
                      options={[{ value: "", label: t("new.field.no_preset") }, ...presets.map((p) => ({ value: p, label: p }))]}
                    />
                  </div>
                )}
                {Object.entries(approach.options ?? {}).map(([name, spec]) => {
                  const choices = spec.choices as OptionValue[];
                  const reasons = [...new Set(choices.map((c) => blockedReason(name, c)).filter(Boolean))];
                  return (
                    <div className="option-row" key={name}>
                      <div className="name">{name}</div>
                      <div>
                        <Segmented<OptionValue>
                          label={name}
                          value={decided[name]}
                          onChange={(choice) => setOptions({ ...options, [name]: choice })}
                          options={choices.map((choice) => ({
                            value: choice,
                            label: show(choice),
                            disabled: Boolean(blockedReason(name, choice)),
                            title: blockedReason(name, choice),
                          }))}
                        />
                        {reasons.map((reason) => (
                          <div className="why" key={reason}>
                            {reason}
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </Panel>

          <Panel title={t("new.plan.title")} sub={check ? t("new.plan.summary", { stages: check.stages.length, cached }) : undefined}>
            {checking && !check && <Skeleton height={58} />}
            <ErrorBox error={checkError} />
            {check && (
              <div className="stack" style={{ gap: 10 }}>
                <StageRoad
                  stages={check.stages.map((s) => ({
                    stage_id: s.stage_id,
                    status: s.cache_hit ? "succeeded" : "pending",
                    cached: s.cache_hit,
                    note: t(s.cache_hit ? "new.plan.cache" : "new.plan.run"),
                  }))}
                />
                <p className="muted small">
                  {check.lidar_usage.length ? t("new.plan.lidar", { stages: check.lidar_usage.join(", ") }) : t("new.plan.no_lidar")}
                </p>
              </div>
            )}
            {!check && !checking && !checkError && <p className="muted">{t("new.plan.waiting")}</p>}
          </Panel>
        </div>

        <aside className="sticky stack">
          <Panel title={t("new.section.launch")}>
            <div className="stack" style={{ gap: 14 }}>
              <dl className="facts" style={{ flexDirection: "column", gap: 8 }}>
                <div>
                  <dt>{t("new.field.scene")}</dt>
                  <dd>{sceneId || "–"}</dd>
                </div>
                <div>
                  <dt>{t("new.field.split")}</dt>
                  <dd>{splitId || "–"}</dd>
                </div>
                <div>
                  <dt>{t("new.field.approach")}</dt>
                  <dd>{approach?.name ?? "–"}</dd>
                </div>
                {check && Object.keys(check.options).length > 0 && (
                  <div>
                    <dt>{t("new.options.title")}</dt>
                    <dd className="small">
                      {Object.entries(check.options).map(([k, v]) => (
                        <div key={k}>
                          {k}: <b>{show(v)}</b>
                        </div>
                      ))}
                    </dd>
                  </div>
                )}
              </dl>
              <div className="field">
                <span>{t("new.field.executor")}</span>
                <Segmented<"local" | "kaggle">
                  label={t("new.field.executor")}
                  value={executor}
                  onChange={setExecutor}
                  options={[
                    { value: "local", label: <span className="row" style={{ gap: 5 }}><Cpu size={13} />{t("new.executor.local")}</span> },
                    { value: "kaggle", label: <span className="row" style={{ gap: 5 }}><Cloud size={13} />Kaggle</span> },
                  ]}
                />
                <span className="hint">{t(`new.executor.${executor}_hint`)}</span>
              </div>
              {cached > 0 && (
                <div className="row small muted">
                  <Database size={13} />
                  {t("new.plan.reuse", { count: cached })}
                </div>
              )}
              <button className="btn primary" onClick={start} disabled={!check || starting} style={{ height: 38 }}>
                <Play size={15} />
                {t(starting ? "new.action.starting" : executor === "kaggle" ? "new.action.start_kaggle" : "new.action.start")}
              </button>
              <ErrorBox error={startError} />
            </div>
          </Panel>
        </aside>
      </div>
    </div>
  );
}
