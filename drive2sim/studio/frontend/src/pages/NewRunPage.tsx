// Chạy mới (SPEC mục 5 bước 4–5): chọn cảnh, split, approach, preset và các mục cấu hình; chọn chạy trên
// máy, trên Kaggle hay trên Colab, train đầy đủ hay thử nhanh. Mỗi lần đổi lựa chọn, gọi /check để biết lựa chọn nào bị khoá (gạch và ghi
// lý do) và vẽ trước làn stage sẽ chạy, stage nào dùng lại cache. Bấm chạy thì mở màn hình run.
import { AlertTriangle, CheckCircle2, Cloud, Cpu, Database, Play, XCircle, Zap } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { asApiError, post, seg, useApi, type ApiError } from "../api/client";
import type { Approach, CheckResult, List, Manifest, OptionValue, RunRecord, SceneDetail } from "../api/types";
import { ColabSetup, type ColabSetupStatus } from "../components/ColabSetup";
import { KaggleSetup, type KaggleSetupStatus } from "../components/KaggleSetup";
import { StageRoad } from "../components/StageRoad";
import { ErrorBox, InfoTip, Note, PageHead, Panel, Segmented, Skeleton } from "../components/ui";
import { useI18n } from "../i18n";
import { lensGroup, lensOf, lensSummary } from "../lens";
import { go, href } from "../router";
import { EnvInstall } from "../components/EnvInstall";

// Pinhole lý tưởng của NCore được coi là pinhole OpenCV, như engine (CONTRACTS 4.2).
const lensModel = (model: string) => (model === "ideal_pinhole" ? "opencv_pinhole" : model);

// Thứ tự nhóm approach ở Chạy mới (CONTRACTS 4.1, trường `kind`).
const APPROACH_KINDS = ["per_scene", "feed_forward", "slam"] as const;

const show = (value: OptionValue) => (typeof value === "boolean" ? (value ? "true" : "false") : String(value));

export function NewRunPage({
  sceneId: initialScene,
  splitId: initialSplit,
  approachId: initialApproach,
  preset: initialPreset,
  executor: initialExecutor,
  quick: initialQuick,
}: {
  sceneId?: string;
  splitId?: string;
  approachId?: string;
  preset?: string;
  executor?: "local" | "kaggle" | "colab";
  quick?: boolean;
}) {
  const { t } = useI18n();
  const scenes = useApi<List<Manifest>>("/api/scenes");
  const approaches = useApi<List<Approach>>("/api/approaches");
  const [sceneId, setSceneId] = useState(initialScene ?? "");
  const scene = useApi<SceneDetail>(sceneId ? `/api/scenes/${seg(sceneId)}` : null);
  const [splitId, setSplitId] = useState(initialSplit ?? "");
  const [approachId, setApproachId] = useState(initialApproach ?? "");
  const autoApproach = useRef(!initialApproach); // approach hiện tại do trang tự chọn
  // preset đề xuất (Trợ lý): điền sau khi approach, cảnh và split đã nạp, sau bước về mặc định
  const [pendingPreset, setPendingPreset] = useState(initialPreset);
  // Chỉ các mục cấu hình người dùng đã chọn khác đi; mục không có ở đây lấy giá trị mặc định của approach.
  // Preset không lưu riêng: bấm preset là điền các mục cấu hình, còn preset đang dùng được suy ra từ các mục cấu hình.
  const [options, setOptions] = useState<Record<string, OptionValue>>({});
  const [executor, setExecutor] = useState<"local" | "kaggle" | "colab">(initialExecutor ?? "local");
  const [quick, setQuick] = useState(Boolean(initialQuick));
  const [kaggleStatus, setKaggleStatus] = useState<KaggleSetupStatus | null>(null);
  const [colabStatus, setColabStatus] = useState<ColabSetupStatus | null>(null);
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
  // Tự chọn approach đầu tiên dùng được trên cảnh (không bị khoá vì thiếu box 3D). Cảnh tải xong sau
  // danh sách approach thì chọn lại nếu approach tự chọn trước đó bị khoá; approach người dùng chọn
  // hoặc có trong URL thì giữ nguyên.
  useEffect(() => {
    const items = approaches.data?.items ?? [];
    if (items.length === 0) return;
    const current = items.find((a) => a.approach_id === approachId);
    if (approachId && !(autoApproach.current && current && boxReason(current))) return;
    const pick = items.find((a) => !boxReason(a)) ?? items[0];
    if (pick.approach_id !== approachId) setApproachId(pick.approach_id);
  }, [approaches.data, approachId, scene.data]);
  // Đổi approach, cảnh hay split thì về cấu hình mặc định; nếu mặc định không vẽ được camera của
  // split (ví dụ 3DGS trên split có camera mắt cá) thì chọn sẵn preset đầu tiên vẽ được, thay vì để
  // người dùng gặp lỗi bị khoá.
  useEffect(() => {
    const support = approach?.camera_support;
    if (!support || !scene.data || !cameraReason(support.default)) return setOptions({});
    const name = Object.keys(approach.presets ?? {}).find((n) => !cameraReason(support.presets[n]));
    setOptions(name ? { ...((approach.presets ?? {})[name] as Record<string, OptionValue>) } : {});
  }, [approachId, scene.data, approaches.data, splitId]);
  useEffect(() => {
    if (!pendingPreset || !approach || !scene.data || !scene.data.splits.includes(splitId)) return;
    const values = (approach.presets ?? {})[pendingPreset];
    if (values) setOptions({ ...(values as Record<string, OptionValue>) });
    setPendingPreset(undefined);
  }, [pendingPreset, approach, scene.data, splitId]);

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

  // Camera của split đang chọn (chưa có split thì mọi camera của cảnh): làm mờ approach, preset và
  // lựa chọn không vẽ được các camera đó trước cả khi /check trả lời. Phải theo camera của split như
  // engine (CONTRACTS mục 4.2): split KITTI-360 chỉ gồm camera pinhole thì 3DGS vẫn chạy được dù cảnh
  // có camera mắt cá.
  const sceneCameras = scene.data?.manifest.cameras ?? [];
  const splitCameraIds = (splitId && scene.data?.split_cameras?.[splitId]) || null;
  const usedCameras = splitCameraIds ? sceneCameras.filter((c) => splitCameraIds.includes(c.camera_id)) : sceneCameras;
  const usedModels = [...new Set(usedCameras.map((c) => lensModel(c.model)))];
  const usedLenses = new Set(usedCameras.map((c) => lensOf(c.model)));
  const usedGroup = lensGroup(usedCameras.map((c) => c.model));
  const sceneHasPinhole = sceneCameras.some((c) => lensOf(c.model) === "pinhole");
  const splitLabel = (id: string) => {
    const ids = scene.data?.split_cameras?.[id];
    if (!ids) return id;
    return `${id} · ${lensSummary(sceneCameras.filter((c) => ids.includes(c.camera_id)).map((c) => c.model), t)}`;
  };
  const fits = (models: string[]) => usedModels.every((m) => models.includes(m));
  const trackText = (fit: { tracks: string[] }) => fit.tracks.map((track) => t(`new.track.${track}`)).join(", ");
  const cameraReason = (fit: { camera_models: string[] }) =>
    scene.data && !fits(fit.camera_models) ? t("new.camera.mismatch", { models: usedModels.join(", ") }) : undefined;
  // Cùng quy tắc với kiểm tra lúc tạo run (engine/resolve.py): approach bắt buộc box 3D mà cảnh không có box.
  const boxReason = (a: Approach) =>
    scene.data && a.requires.cuboids === "required" && scene.data.manifest.available.cuboids === "none"
      ? t("new.boxes.missing.hint")
      : undefined;
  const choiceCameraReason = (name: string, choice: OptionValue) => {
    const spec = specs[name];
    const index = (spec?.choices as OptionValue[] | undefined)?.indexOf(choice) ?? -1;
    const models = spec?.camera_models?.[index];
    return models ? cameraReason({ camera_models: models }) : undefined;
  };
  const presetFit = (name: string) => (name ? approach?.camera_support.presets[name] : approach?.camera_support.default);

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
            {/* auto-fit: hai ô giãn hết chiều rộng, để tên cảnh và split (kèm loại ống kính) không bị cắt */}
            <div className="form-grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))" }}>
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
                    <option key={id} value={id}>
                      {splitLabel(id)}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            {usedGroup && (
              <p className="small muted row wrap" style={{ marginTop: 12, gap: 6 }}>
                {splitCameraIds ? t("new.field.split_cameras") : t("new.field.camera")}:{" "}
                <span className="tag">{t(`lens.${usedGroup}`)}</span>
                {lensSummary(usedCameras.map((c) => c.model), t)}
                <code>{usedCameras.map((c) => c.camera_id).join(", ")}</code>
                <InfoTip>{t("lens.hint")}</InfoTip>
              </p>
            )}
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
            {APPROACH_KINDS.map((kind) => {
              const items = (approaches.data?.items ?? []).filter((a) => (a.kind ?? "per_scene") === kind);
              if (items.length === 0) return null;
              return (
                <section key={kind} className="approach-group" data-kind={kind}>
                  <h4 className="approach-group-title">
                    {t(`new.kind.${kind}`)}
                    <InfoTip>{t(`new.kind.${kind}.hint`)}</InfoTip>
                  </h4>
                  <div className="choice-list">
                  {items.map((a) => (
                    <button
                      key={a.approach_id}
                      className="choice"
                      data-kind={kind}
                      aria-pressed={a.approach_id === approachId}
                      onClick={() => {
                        autoApproach.current = false;
                        setApproachId(a.approach_id);
                      }}
                      disabled={Boolean(boxReason(a)) && a.approach_id !== approachId}
                      title={boxReason(a) ?? cameraReason(a.camera_support)}
                    >
                      <span className="name">
                        {a.name}
                        {a.experimental && (
                          <span className="tag warn" title={t("new.approach.experimental.hint")} style={{ marginLeft: 8 }}>
                            {t("new.approach.experimental")}
                          </span>
                        )}
                      </span>
                      <span className="desc">{t("new.approach.meta", { codebase: a.codebase, level: a.level, stages: a.stages.length })}</span>
                      <span className="row wrap" style={{ gap: 6 }}>
                        {a.camera_support.tracks.map((track) => (
                          <span key={track} className={`tag${usedLenses.has(track as "pinhole" | "fisheye") ? " go" : ""}`}>
                            {t(`new.track.${track}`)}
                          </span>
                        ))}
                        {a.dynamics && (
                          <span className="tag" title={t(`new.dynamics.${a.dynamics}.hint`)}>
                            {t(`new.dynamics.${a.dynamics}`)}
                          </span>
                        )}
                        {cameraReason(a.camera_support) && <span className="tag warn">{t("new.camera.unfit")}</span>}
                        {boxReason(a) && <span className="tag warn">{t("new.boxes.missing")}</span>}
                      </span>
                    </button>
                  ))}
                  </div>
                </section>
              );
            })}
            {approach && usedLenses.has("fisheye") && sceneHasPinhole && cameraReason(approach.camera_support.default) && (
              <div style={{ marginTop: 12 }}>
                <Note>
                  {t("new.camera.make_pinhole")}: <a href={href({ page: "scene", sceneId })}>{sceneId}</a>
                </Note>
              </div>
            )}
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
                            <b>{t("new.field.no_preset")}</b>: {t("new.preset.default_help")} ({trackText(approach.camera_support.default)})
                          </span>
                          {presetNames.map((name) => (
                            <span key={name}>
                              <b>{name}</b>: {presetHelp(name) ?? optionsSummary((approach.presets ?? {})[name] as Record<string, OptionValue>)} (
                              {trackText(approach.camera_support.presets[name])})
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
                        options={["", ...presetNames].map((name) => {
                          const fit = presetFit(name);
                          const reason = fit && cameraReason(fit);
                          return {
                            value: name,
                            label: name || t("new.field.no_preset"),
                            disabled: Boolean(reason),
                            title: reason ?? (fit ? trackText(fit) : undefined),
                          };
                        })}
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
                      const reasonOf = (choice: OptionValue) =>
                        blockedReason(name, choice) ?? choiceCameraReason(name, choice) ?? conflict(name, choice);
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
                          <div className="row wrap" style={{ gap: 10 }}>
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
                            {spec.custom && (
                              <CustomValue
                                key={`${approachId}-${name}`}
                                range={spec.custom}
                                value={current(name)}
                                isChoice={choices.includes(current(name))}
                                onChange={(value) => setOptions({ ...options, [name]: value })}
                              />
                            )}
                          </div>
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
                {(check.data_warnings ?? []).map((w) => (
                  <p key={w} className="small" style={{ color: "var(--warn, #d97706)" }}>
                    ⚠ {w}
                  </p>
                ))}
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
                <Segmented<"local" | "kaggle" | "colab">
                  label={t("new.field.executor")}
                  value={executor}
                  onChange={setExecutor}
                  options={[
                    { value: "local", label: <span className="row" style={{ gap: 7 }}><Cpu size={16} />{t("new.executor.local")}</span> },
                    { value: "kaggle", label: <span className="row" style={{ gap: 7 }}><Cloud size={16} />Kaggle</span> },
                    { value: "colab", label: <span className="row" style={{ gap: 7 }}><Cloud size={16} />Colab</span> },
                  ]}
                />
              </div>
              {executor === "local" && approachId && (
                <LocalEnvironment approachId={approachId} stages={check?.approach_id === approachId ? check.stages.map((s) => s.stage_id) : null} />
              )}
              {executor === "kaggle" && <KaggleSetup onStatus={setKaggleStatus} />}
              {executor === "kaggle" && kaggleStatus && !kaggleStatus.ready && <Note kind="warn">{t("kaggle.setup.start_warning")}</Note>}
              {executor === "colab" && <ColabSetup onStatus={setColabStatus} needsSmpl={check?.stages.some((s) => s.stage_id === "prep.smpl")} />}
              {executor === "colab" && colabStatus && !colabStatus.ready && <Note kind="warn">{t("colab.setup.start_warning")}</Note>}
              {cached > 0 && (
                <div className="row small muted">
                  <Database size={16} />
                  {t("new.plan.reuse", { count: cached })}
                </div>
              )}
              <button className="btn primary" onClick={start} disabled={!check || starting} style={{ height: 47 }}>
                <Play size={19} />
                {t(starting ? "new.action.starting" : executor === "kaggle" ? "new.action.start_kaggle" : executor === "colab" ? "new.action.start_colab" : "new.action.start")}
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

/** Chạy trên máy này: môi trường Python có đúng phiên bản approach cần không (GET .../environment).
 * ``stages``: stage của run đã giải (kết quả kiểm tra), để file SMPL chỉ bị đòi khi run thật sự chạy
 * bước SMPL; chưa có thì server coi SMPL là có thể cần (chỉ cảnh báo). */
function LocalEnvironment({ approachId, stages }: { approachId: string; stages: string[] | null }) {
  const { t } = useI18n();
  const query = stages ? `?stages=${encodeURIComponent(stages.join(","))}` : "";
  const env = useApi<EnvironmentCheck>(`/api/approaches/${seg(approachId)}/environment${query}`);
  if (!env.data || !env.data.env) return null;
  const data = env.data;
  // chỉ thiếu file SMPL: script cài không sửa được, nói đúng nguyên nhân và chỗ đặt file
  const notOk = data.checks.filter((c) => c.level !== "ok");
  const smplOnly = notOk.length > 0 && notOk.every((c) => c.id === "smpl_model");
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
        <b>
          {smplOnly
            ? t(data.level === "missing" ? "new.env.smpl_missing" : "new.env.smpl_maybe")
            : t(data.level === "missing" ? "new.env.missing" : "new.env.warn", { env: data.env ?? "" })}
        </b>
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
        {smplOnly ? (
          <span className="small">{t("new.env.smpl_fix")}</span>
        ) : (
        <span className="small">
          {t("new.env.install")} <code>{data.install}</code>
          <InfoTip>{t(data.env === "nurec" || data.env === "xsim" ? `new.env.install_hint_${data.env}` : "new.env.install_hint", { file: data.file ?? "" })}</InfoTip>
        </span>
        )}
        {data.env === "nurec" && (
          <span className="small">
            {t("new.env.nurec_notebook")}{" "}
            <a href="/api/environment/nurec/notebook/colab" download>
              Colab
            </a>
            {" · "}
            <a href="/api/environment/nurec/notebook/kaggle" download>
              Kaggle
            </a>
            <InfoTip>{t("new.env.nurec_notebook_hint")}</InfoTip>
          </span>
        )}
        {/* cài chỉ sửa được gói Python: không hiện khi thiếu GPU, hay khi chỉ thiếu file SMPL */}
        {data.env &&
          !data.checks.some((c) => c.id === "gpu" && c.level === "missing") &&
          data.checks.some((c) => c.level !== "ok" && c.id !== "gpu" && c.id !== "smpl_model") && (
          <EnvInstall env={data.env} onDone={env.reload} />
        )}
      </div>
    </Note>
  );
}

/** Ô nhập giá trị số tự chọn của một mục cấu hình có ``custom`` (ví dụ số Gaussian của NuRec). */
function CustomValue({
  range,
  value,
  isChoice,
  onChange,
}: {
  range: { min: number; max: number; integer?: boolean };
  value: OptionValue;
  isChoice: boolean;
  onChange: (value: number) => void;
}) {
  const { t } = useI18n();
  const [text, setText] = useState(isChoice ? "" : String(value));
  // Chọn lại một lựa chọn sẵn có thì xoá ô nhập.
  useEffect(() => {
    if (isChoice) setText("");
  }, [isChoice, value]);
  const number = Number(text.replace(/[\s.,_]/g, ""));
  const valid = text.trim() !== "" && Number.isFinite(number) && number >= range.min && number <= range.max && (!range.integer || Number.isInteger(number));
  const hint = t("new.option.custom_range", { min: range.min.toLocaleString(), max: range.max.toLocaleString() });
  return (
    <label className={`custom-value${!isChoice && text ? " active" : ""}${text && !valid ? " invalid" : ""}`} title={hint}>
      <span className="small muted">{t("new.option.custom")}</span>
      <input
        type="text"
        inputMode="numeric"
        value={text}
        placeholder={hint}
        aria-invalid={Boolean(text) && !valid}
        onChange={(e) => {
          setText(e.target.value);
          const next = Number(e.target.value.replace(/[\s.,_]/g, ""));
          if (e.target.value.trim() && Number.isFinite(next) && next >= range.min && next <= range.max && (!range.integer || Number.isInteger(next))) onChange(next);
        }}
      />
      {text && valid && <span className="small dense">{number.toLocaleString()}</span>}
    </label>
  );
}
