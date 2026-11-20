// Ngăn "Trợ lý": hỏi đáp với một mô hình ngôn ngữ, câu trả lời dựa trên dữ kiện thật của Studio lúc
// hỏi (CONTRACTS mục 11.1, /api/assistant). Cài đặt: nhà cung cấp chính thức hoặc endpoint tự đặt,
// model, API key (nhập ở đây, lưu trên máy chạy Studio, không hiện lại). Trợ lý đề xuất cấu hình thì
// có nút mở trang tạo run với cảnh, split, approach, preset, nơi chạy đó. Cuộc hỏi đáp giữ trong
// sessionStorage của tab, đóng ngăn rồi mở lại vẫn còn.
import { Eraser, ListRestart, Rocket, Send, Settings } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { ApiError, api, asApiError, post } from "../api/client";
import { useI18n } from "../i18n";
import { href } from "../router";
import { Drawer, ErrorBox } from "./ui";

interface Provider {
  id: string;
  label: string;
  base_url: string | null;
  needs_url: boolean;
}

interface Config {
  provider: string;
  base_url: string;
  model: string;
  has_key: boolean;
  key_hint: string;
  providers: Provider[];
}

interface Apply {
  scene_id?: string;
  split_id?: string;
  approach_id?: string;
  preset?: string;
  executor?: string;
  quick?: boolean;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  apply?: Apply | null;
}

const STORE = "d2s.assistant.messages";

function loadMessages(): Message[] {
  try {
    const value = JSON.parse(sessionStorage.getItem(STORE) ?? "[]");
    return Array.isArray(value) ? value : [];
  } catch {
    return [];
  }
}

/** Bỏ khối d2s-apply khỏi phần chữ hiện ra; khối đó thành nút. */
const visible = (text: string) => text.replace(/```d2s-apply[\s\S]*?```/g, "").trim();

function Setup({ config, onSaved }: { config: Config; onSaved: (config: Config) => void }) {
  const { t } = useI18n();
  const [provider, setProvider] = useState(config.provider);
  const [baseUrl, setBaseUrl] = useState(config.base_url);
  const [model, setModel] = useState(config.model);
  const [key, setKey] = useState("");
  const [models, setModels] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const current = config.providers.find((p) => p.id === provider);

  const run = async (action: () => Promise<void>) => {
    setBusy(true);
    setError(null);
    try {
      await action();
    } catch (err) {
      setError(asApiError(err));
    } finally {
      setBusy(false);
    }
  };
  const choose = (id: string) =>
    run(async () => {
      const saved = await post<Config>("/api/assistant/config", { provider: id });
      setProvider(saved.provider);
      setBaseUrl(saved.base_url);
      setModel(saved.model);
      setModels([]);
      onSaved(saved);
    });
  const save = () =>
    run(async () => {
      const body: Record<string, string> = { model };
      if (current?.needs_url) body.base_url = baseUrl;
      if (key.trim()) body.api_key = key.trim();
      onSaved(await post<Config>("/api/assistant/config", body));
      setKey("");
    });
  const clearKey = () => run(async () => onSaved(await post<Config>("/api/assistant/config", { clear_key: true })));
  const listModels = () => run(async () => setModels((await api<{ models: string[] }>("/api/assistant/models")).models));

  return (
    <div className="stack" style={{ gap: 8 }}>
      <label className="field">
        <span>{t("assistant.provider")}</span>
        <select value={provider} onChange={(e) => void choose(e.target.value)} disabled={busy}>
          {config.providers.map((p) => (
            <option key={p.id} value={p.id}>
              {p.label}
            </option>
          ))}
        </select>
      </label>
      {current?.needs_url && (
        <label className="field">
          <span>{t("assistant.base_url")}</span>
          <input value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} placeholder="https://host/v1" spellCheck={false} />
        </label>
      )}
      <div className="row" style={{ gap: 8, alignItems: "flex-end" }}>
        <label className="field grow">
          <span>{t("assistant.model")}</span>
          <input value={model} onChange={(e) => setModel(e.target.value)} list="d2s-assistant-models" spellCheck={false} />
          <datalist id="d2s-assistant-models">
            {models.map((m) => (
              <option key={m} value={m} />
            ))}
          </datalist>
        </label>
        <button className="btn sm" type="button" onClick={() => void listModels()} disabled={busy || !config.has_key} title={t("assistant.list_models")}>
          <ListRestart size={15} />
        </button>
      </div>
      <label className="field">
        <span>
          {t("assistant.key")} {config.has_key && <span className="muted">({t("assistant.key_saved", { hint: config.key_hint })})</span>}
        </span>
        <input type="password" value={key} onChange={(e) => setKey(e.target.value)} autoComplete="off" placeholder={config.has_key ? t("assistant.key_keep") : ""} />
      </label>
      <div className="row" style={{ gap: 8 }}>
        <button className="btn sm primary" type="button" onClick={() => void save()} disabled={busy}>
          {t("assistant.save")}
        </button>
        {config.has_key && (
          <button className="btn sm" type="button" onClick={() => void clearKey()} disabled={busy}>
            {t("assistant.clear_key")}
          </button>
        )}
      </div>
      <span className="small muted">{t("assistant.privacy")}</span>
      <ErrorBox error={error} />
    </div>
  );
}

export function AssistantDrawer({ page, onClose }: { page: string; onClose: () => void }) {
  const { t } = useI18n();
  const [config, setConfig] = useState<Config | null>(null);
  const [setup, setSetup] = useState(false);
  const [messages, setMessages] = useState<Message[]>(loadMessages);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const end = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api<Config>("/api/assistant/config")
      .then((result) => {
        setConfig(result);
        if (!result.has_key || !result.model) setSetup(true);
      })
      .catch((err) => setError(asApiError(err)));
  }, []);
  useEffect(() => {
    sessionStorage.setItem(STORE, JSON.stringify(messages.slice(-40)));
    end.current?.scrollIntoView({ block: "end" });
  }, [messages]);

  const send = async () => {
    const question = draft.trim();
    if (!question || busy) return;
    const next: Message[] = [...messages, { role: "user", content: question }];
    setMessages(next);
    setDraft("");
    setBusy(true);
    setError(null);
    try {
      const payload = next.map(({ role, content }) => ({ role, content }));
      const result = await post<{ reply: string; apply: Apply | null }>("/api/assistant/chat", { messages: payload, page });
      setMessages([...next, { role: "assistant", content: result.reply, apply: result.apply }]);
    } catch (err) {
      setError(asApiError(err));
    } finally {
      setBusy(false);
    }
  };

  const open = (apply: Apply) => {
    const executor = ["local", "kaggle", "colab"].includes(apply.executor ?? "") ? (apply.executor as "local" | "kaggle" | "colab") : undefined;
    window.location.hash = href({
      page: "new",
      sceneId: apply.scene_id,
      splitId: apply.split_id,
      approachId: apply.approach_id,
      preset: apply.preset || undefined,
      executor,
      quick: apply.quick || undefined,
    });
    onClose();
  };

  return (
    <Drawer
      title={t("assistant.title")}
      info={t("assistant.info")}
      onClose={onClose}
      foot={
        <form
          className="row"
          style={{ gap: 8, width: "100%" }}
          onSubmit={(e) => {
            e.preventDefault();
            void send();
          }}
        >
          <textarea
            className="grow"
            rows={2}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void send();
              }
            }}
            placeholder={t("assistant.placeholder")}
            disabled={busy || !config?.has_key}
          />
          <button className="btn primary" type="submit" disabled={busy || !draft.trim() || !config?.has_key} aria-label={t("assistant.send")}>
            <Send size={17} />
          </button>
        </form>
      }
    >
      <div className="row" style={{ gap: 8 }}>
        <span className="small muted grow">
          {config ? `${config.providers.find((p) => p.id === config.provider)?.label ?? config.provider} · ${config.model || "–"}` : ""}
        </span>
        <button className="btn sm ghost" type="button" onClick={() => setSetup(!setup)}>
          <Settings size={15} />
          {t("assistant.settings")}
        </button>
        {messages.length > 0 && (
          <button className="btn sm ghost" type="button" onClick={() => setMessages([])} disabled={busy}>
            <Eraser size={15} />
            {t("assistant.clear")}
          </button>
        )}
      </div>
      {setup && config && <Setup config={config} onSaved={setConfig} />}
      <div className="stack" style={{ gap: 10 }}>
        {messages.length === 0 && <p className="small muted">{t("assistant.empty")}</p>}
        {messages.map((m, i) => (
          <div key={i} className="panel" style={{ padding: "10px 13px", background: m.role === "user" ? "var(--surface-2, transparent)" : undefined }}>
            <div className="small faint">{m.role === "user" ? t("assistant.you") : t("assistant.bot")}</div>
            <div style={{ whiteSpace: "pre-wrap" }}>{m.role === "assistant" ? visible(m.content) : m.content}</div>
            {m.apply && (
              <button className="btn sm primary" type="button" style={{ marginTop: 8 }} onClick={() => open(m.apply as Apply)}>
                <Rocket size={15} />
                {t("assistant.apply", { approach: m.apply.approach_id ?? "", preset: m.apply.preset || t("assistant.default_preset") })}
              </button>
            )}
          </div>
        ))}
        {busy && (
          <div className="progress indeterminate">
            <i />
          </div>
        )}
        <div ref={end} />
      </div>
      <ErrorBox error={error} />
    </Drawer>
  );
}
