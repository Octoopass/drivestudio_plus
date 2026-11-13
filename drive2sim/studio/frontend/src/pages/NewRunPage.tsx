// Chạy mới (SPEC mục 5 bước 4–5): chọn cảnh, split, approach, preset và các mục cấu hình; chọn chạy trên
// máy hay trên Kaggle, train đầy đủ hay thử nhanh. Mỗi lần đổi lựa chọn, gọi /check để biết lựa chọn nào bị khoá (gạch và ghi
// lý do) và vẽ trước làn stage sẽ chạy, stage nào dùng lại cache. Bấm chạy thì mở màn hình run.
import { AlertTriangle, CheckCircle2, Cloud, Cpu, Database, Play, XCircle, Zap } from "lucide-react";
import { useEffect, useState } from "react";
import { asApiError, post, seg, useApi, type ApiError } from "../api/client";
import type { Approach, CheckResult, List, Manifest, OptionValue, RunRecord, SceneDetail } from "../api/types";
import { KaggleSetup, type KaggleSetupStatus } from "../components/KaggleSetup";
import { StageRoad } from "../components/StageRoad";
import { ErrorBox, InfoTip, Note, PageHead, Panel, Segmented, Skeleton } from "../components/ui";
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
  // Chỉ các mục cấu hình người dùng đã chọn khác đi; mục không có ở đây lấy giá trị mặc định của approach.
  // Preset không lưu riêng: bấm preset là điền các mục cấu hình, còn preset đang dùng được suy ra từ các mục cấu hình.
  const [options, setOptions] = useState<Record<string, OptionValue>>({});
  const [executor, setExecutor] = useState<"local" | "kaggle">("local");
  const [quick, setQuick] = useState(false);
  const [kaggleStatus, setKaggleStatus] = useState<KaggleSetupStatus | null>(null);
  const [check, setCheck] = useState<CheckResult | null>(null);
  const [checkError, setCheckError] = useState<ApiError | null>(null);
  const [checking, setChecking] = useState(false);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<ApiError | null>(null);

  const approach = approaches.data?.items.find((a) => a.approach_id === approachId);
  const specs = approach?.options ?? {};
  const raw = (name: string): OptionValue => (name in options ? options[name] : (specs[name]?.default as OptionValue));
  // Preset khớp khi mọi mục cấu hình đúng bằng giá trị preset (mục preset không ghi thì là mặc định).
  const matches = (values: Record<string, OptionValue>) =>
    Object.keys(specs).every((name) => raw(name) === (name in values ? values[name] : specs[name].default));
  const presetNames = Object.keys(approach?.presets ?? {});
  const activePreset: string | null = matches({})
    ? ""
    : (presetNames.find((name) => matches((approach?.presets ?? {})[name] as Record<string, OptionValue>)) ?? null);
  const preset = activePreset || null;

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
  useEffect(() => setOptions({}), [approachId]);

  // Kiểm tra lại mỗi khi lựa chọn đổi.
  useEffect(() => {
    if (!sceneId || !splitId || !approachId) {
      setCheck(null);
      return;
    }
    let cancelled = false;
    setChecking(true);
    post<CheckResult>(`/api/approaches/${seg(approachId)}/check`, { scene_id: sceneId, split_id: splitId, preset, options, quick })
      .then((result) => !cancelled && (setCheck(result), setCheckError(null)))
      .catch((err) => !cancelled && (setCheck(null), setCheckError(asApiError(err))))
      .finally(() => !cancelled && setChecking(false));
    return () => {
      cancelled = true;
    };
  }, [sceneId, splitId, approachId, preset, options, quick]);

  const start = async () => {
    setStarting(true);
    setStartError(null);
    try {
      const run = await post<RunRecord>("/api/runs", {
        approach_id: approachId,
        scene_id: sceneId,
        split_id: splitId,
        preset,
        options,
        quick,
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
  // Số iteration của stage train; không có khi lấy theo file config của method (thường 30000).
  const trainIters = check?.stages.map((s) => s.params.num_iters).find((n) => typeof n === "number") as number | undefined;
  // Giá trị hiện tại của một mục cấu hình: lấy từ /check (đã quyết định cả "auto"), chưa có thì lấy tạm.
  const current = (name: string): OptionValue => (name in decided ? decided[name] : raw(name));
  // Lựa chọn xung đột với các mục cấu hình khác: cùng quy tắc với engine (mọi cặp when và forbid đều khớp).
  const conflict = (name: string, choice: OptionValue): string | undefined => {
    for (const constraint of approach?.constraints ?? []) {
      const pairs = { ...constraint.when, ...constraint.forbid } as Record<string, OptionValue>;
      if (!(name in pairs)) continue;
      if (Object.entries(pairs).every(([key, value]) => (key === name ? choice : current(key)) === value)) {
        const others = Object.entries(pairs)
          .filter(([key]) => key !== name)
          .map(([key, value]) => `${optionLabel(key)} ${choiceLabel(key, value)}`)
          .join(", ");
        return t("new.option.conflict", { others, reason: constraint.reason });
      }
    }
    return undefined;
  };
  const optionLabel = (name: string) => specs[name]?.label ?? name;
  const choiceLabel = (name: string, choice: OptionValue) => {
    const spec = specs[name];
    const index = (spec?.choices as OptionValue[] | undefined)?.indexOf(choice) ?? -1;
    return spec?.labels?.[index] ?? show(choice);
  };
  const presetHelp = (name: string) => approach?.preset_help?.[name];
  const optionsSummary = (values: Record<string, OptionValue>) =>
    Object.entries(values)
      .map(([key, value]) => `${optionLabel(key)} ${choiceLabel(key, value)}`)
      .join(", ");
  // Mục không thuộc nhóm nào hiện trước, không có tiêu đề; sau đó từng nhóm theo thứ tự khai báo.
  const groupMeta = approach?.groups ?? {};
  const optionGroups = [
    { id: "", label: "", help: undefined as string | undefined, names: Object.keys(specs).filter((n) => !specs[n].group || !(specs[n].group! in groupMeta)) },
    ...Object.entries(groupMeta).map(([id, meta]) => ({
      id,
      label: meta.label,
      help: meta.help ?? undefined,
      names: Object.keys(specs).filter((n) => specs[n].group === id),
    })),
  ].filter((group) => group.names.length > 0);

  return (
    <div className="page">
      <PageHead title={t("new.page.title")} info={t("new.page.lede")} />
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
              <div style={{ marginTop: 15 }}>
                <Note>
                  {t("new.need.scene")} <a href={href({ page: "scenes" })}>{t("nav.menu.scenes")}</a>
                </Note>
              </div>
            )}
            {scene.data && scene.data.splits.length === 0 && (
              <div style={{ marginTop: 15 }}>
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
            {approach && (presetNames.length > 0 || Object.keys(specs).length > 0) && (
              <div style={{ marginTop: 18 }}>
                {presetNames.length > 0 && (
                  <div className="option-row">
                    <div className="name">
                      {t("new.field.preset")}
                      <InfoTip>
                        <span className="stack" style={{ gap: 6 }}>
                          <span>{t("new.preset.info")}</span>
                          <span>
                            <b>{t("new.field.no_preset")}</b>: {t("new.preset.default_help")}
                          </span>
                          {presetNames.map((name) => (
                            <span key={name}>
                              <b>{name}</b>: {presetHelp(name) ?? optionsSummary((approach.presets ?? {})[name] as Record<string, OptionValue>)}
                            </span>
                          ))}
                        </span>
                      </InfoTip>
                    </div>
                    <div className="row wrap" style={{ gap: 10 }}>
                      <Segmented<string>
                        label={t("new.field.preset")}
                        value={activePreset ?? "\u0000custom"}
                        onChange={(name) => setOptions(name ? { ...((approach.presets ?? {})[name] as Record<string, OptionValue>) } : {})}
                        options={[
                          { value: "", label: t("new.field.no_preset") },
                          ...presetNames.map((name) => ({ value: name, label: name })),
                        ]}
                      />
                      {activePreset === null && <span className="tag warn">{t("new.preset.custom")}</span>}
                    </div>
                  </div>
                )}
                {optionGroups.map((group) => (
                  <div key={group.id} className="option-group">
                    {group.label && (
                      <h3 className="option-group-title">
                        {group.label}
                        {group.help && <InfoTip>{group.help}</InfoTip>}
                      </h3>
                    )}
                    {group.names.map((name) => {
                      const spec = specs[name];
                      const choices = spec.choices as OptionValue[];
                      const reasonOf = (choice: OptionValue) => blockedReason(name, choice) ?? conflict(name, choice);
                      const blocked = choices
                        .map((choice) => ({ choice, reason: reasonOf(choice) }))
                        .filter((item): item is { choice: OptionValue; reason: string } => Boolean(item.reason));
                      return (
                        <div className="option-row" key={name}>
                          <div className="name">
                            {optionLabel(name)}
                            <InfoTip>
                              <span className="stack" style={{ gap: 6 }}>
                                {spec.help && <span>{spec.help}</span>}
                                {choices.map((choice, index) =>
                                  spec.descriptions?.[index] ? (
                                    <span key={String(choice)}>
                                      <b>{choiceLabel(name, choice)}</b>: {spec.descriptions[index]}
                                    </span>
                                  ) : null,
                                )}
                                {blocked.map(({ choice, reason }) => (
                                  <span key={`blocked-${String(choice)}`} className="muted">
                                    {t("new.option.blocked", { choice: choiceLabel(name, choice), reason })}
                                  </span>
                                ))}
                              </span>
                            </InfoTip>
                          </div>
                          <Segmented<OptionValue>
                            label={optionLabel(name)}
                            value={current(name)}
                            onChange={(choice) => setOptions({ ...options, [name]: choice })}
                            options={choices.map((choice) => ({
                              value: choice,
                              label: choiceLabel(name, choice),
                              disabled: Boolean(reasonOf(choice)),
                              title: reasonOf(choice),
                            }))}
                          />
                        </div>
                      );
                    })}
                  </div>
                ))}
              </div>
            )}
          </Panel>

          <Panel title={t("new.plan.title")} sub={check ? t("new.plan.summary", { stages: check.stages.length, cached }) : undefined}>
            {checking && !check && <Skeleton height={58} />}
            <ErrorBox error={checkError} />
            {check && (
              <div className="stack" style={{ gap: 12 }}>
                <StageRoad
                  stages={check.stages.map((s) => ({
                    stage_id: s.stage_id,
                    status: s.cache_hit ? "succeeded" : "pending",
                    cached: s.cache_hit,
                    lidar: check.lidar_usage.includes(s.stage_id),
                    note: t(s.cache_hit ? "new.plan.cache" : "new.plan.run"),
                  }))}
                />
                {check.lidar_usage.length === 0 && <p className="muted small">{t("new.plan.no_lidar")}</p>}
              </div>
            )}
            {!check && !checking && !checkError && <p className="muted">{t("new.plan.waiting")}</p>}
          </Panel>
        </div>

        <aside className="sticky stack">
          <Panel title={t("new.section.launch")}>
            <div className="stack" style={{ gap: 18 }}>
              <dl className="facts" style={{ flexDirection: "column", gap: 10 }}>
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
                {approach && presetNames.length > 0 && (
                  <div>
                    <dt>{t("new.field.preset")}</dt>
                    <dd>{activePreset === null ? t("new.preset.custom") : activePreset || t("new.field.no_preset")}</dd>
                  </div>
                )}
                {check && Object.keys(check.options).length > 0 && (
                  <div>
                    <dt>{t("new.options.title")}</dt>
                    <dd className="small">
                      {Object.entries(check.options).map(([k, v]) => (
                        <div key={k}>
                          {optionLabel(k)}: <b>{choiceLabel(k, v)}</b>
                        </div>
                      ))}
                    </dd>
                  </div>
                )}
              </dl>
              <div className="field">
                <span>
                  {t("new.field.training")}
                  <InfoTip>{t("new.training.info", { iters: trainIters ?? "–" })}</InfoTip>
                </span>
                <Segmented<"full" | "quick">
                  label={t("new.field.training")}
                  value={quick ? "quick" : "full"}
                  onChange={(value) => setQuick(value === "quick")}
                  options={[
                    { value: "full", label: t("new.training.full") },
                    { value: "quick", label: <span className="row" style={{ gap: 7 }}><Zap size={16} />{t("new.training.quick")}</span> },
                  ]}
                />
              </div>
              <div className="field">
                <span>
                  {t("new.field.executor")}
                  <InfoTip>{t("new.executor.info")}</InfoTip>
                </span>
                <Segmented<"local" | "kaggle">
                  label={t("new.field.executor")}
                  value={executor}
                  onChange={setExecutor}
                  options={[
                    { value: "local", label: <span className="row" style={{ gap: 7 }}><Cpu size={16} />{t("new.executor.local")}</span> },
                    { value: "kaggle", label: <span className="row" style={{ gap: 7 }}><Cloud size={16} />Kaggle</span> },
                  ]}
                />
              </div>
              {executor === "local" && approachId && <LocalEnvironment approachId={approachId} />}
              {executor === "kaggle" && <KaggleSetup onStatus={setKaggleStatus} />}
              {executor === "kaggle" && kaggleStatus && !kaggleStatus.ready && <Note kind="warn">{t("kaggle.setup.start_warning")}</Note>}
              {cached > 0 && (
                <div className="row small muted">
                  <Database size={16} />
                  {t("new.plan.reuse", { count: cached })}
                </div>
              )}
              <button className="btn primary" onClick={start} disabled={!check || starting} style={{ height: 47 }}>
                <Play size={19} />
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

interface EnvironmentCheck {
  env: string | null;
  file: string | null;
  install: string | null;
  level: "ok" | "warn" | "missing";
  checks: { id: string; level: "ok" | "warn" | "missing"; found: string | null; expected: string | null }[];
}

/** Chạy trên máy này: môi trường Python có đúng phiên bản approach cần không (GET .../environment). */
function LocalEnvironment({ approachId }: { approachId: string }) {
  const { t } = useI18n();
  const env = useApi<EnvironmentCheck>(`/api/approaches/${seg(approachId)}/environment`);
  if (!env.data || !env.data.env) return null;
  const data = env.data;
  if (data.level === "ok") {
    return (
      <div className="row small muted" style={{ gap: 6 }}>
        <CheckCircle2 size={16} className="ok-ink" />
        {t("new.env.ok", { env: data.env ?? "" })}
      </div>
    );
  }
  return (
    <Note kind="warn">
      <div className="stack" style={{ gap: 6 }}>
        <b>{t(data.level === "missing" ? "new.env.missing" : "new.env.warn", { env: data.env ?? "" })}</b>
        <ul className="env-list">
          {data.checks
            .filter((c) => c.level !== "ok")
            .map((c) => (
              <li key={c.id}>
                {c.level === "missing" ? <XCircle size={15} className="bad-ink" /> : <AlertTriangle size={15} />}
                <span>
                  {t(`new.env.check.${c.id}`)}: {c.found ?? t("new.env.none")}
                  {c.expected && <span className="muted"> ({t("new.env.expected", { version: c.expected })})</span>}
                </span>
              </li>
            ))}
        </ul>
        <span className="small">
          {t("new.env.install")} <code>{data.install}</code>
          <InfoTip>{t("new.env.install_hint", { file: data.file ?? "" })}</InfoTip>
        </span>
      </div>
    </Note>
  );
}
