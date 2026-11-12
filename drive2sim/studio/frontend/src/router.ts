// Điều hướng bằng hash URL (#/scenes/<id>, #/runs/<id>...), không cần thư viện router: FastAPI
// phục vụ một file index.html duy nhất, và hash giữ được trang khi tải lại.
import { useEffect, useState } from "react";

export type Route =
  | { page: "overview" }
  | { page: "scenes" }
  | { page: "scene"; sceneId: string }
  | { page: "new"; sceneId?: string; splitId?: string }
  | { page: "runs"; sceneId?: string; status?: string }
  | { page: "run"; runId: string }
  | { page: "compare"; sceneId?: string; splitId?: string }
  | { page: "viewer"; runId?: string }
  | { page: "storage" };

export type Page = Route["page"];

export function parseRoute(hash: string): Route {
  const [path, query = ""] = hash.replace(/^#/, "").split("?");
  const params = new URLSearchParams(query);
  const get = (name: string) => params.get(name) ?? undefined;
  const parts = path.split("/").filter(Boolean).map(decodeURIComponent);
  switch (parts[0]) {
    case "scenes":
      return parts[1] ? { page: "scene", sceneId: parts[1] } : { page: "scenes" };
    case "new":
      return { page: "new", sceneId: get("scene"), splitId: get("split") };
    case "runs":
      return parts[1] ? { page: "run", runId: parts[1] } : { page: "runs", sceneId: get("scene"), status: get("status") };
    case "compare":
      return { page: "compare", sceneId: get("scene"), splitId: get("split") };
    case "viewer":
      return { page: "viewer", runId: get("run") };
    case "storage":
      return { page: "storage" };
    default:
      return { page: "overview" };
  }
}

function withQuery(path: string, values: Record<string, string | undefined>): string {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(values)) if (value) params.set(key, value);
  const query = params.toString();
  return `#/${path}${query ? `?${query}` : ""}`;
}

export function href(route: Route): string {
  const enc = encodeURIComponent;
  switch (route.page) {
    case "scenes":
      return "#/scenes";
    case "scene":
      return `#/scenes/${enc(route.sceneId)}`;
    case "new":
      return withQuery("new", { scene: route.sceneId, split: route.splitId });
    case "runs":
      return withQuery("runs", { scene: route.sceneId, status: route.status });
    case "run":
      return `#/runs/${enc(route.runId)}`;
    case "compare":
      return withQuery("compare", { scene: route.sceneId, split: route.splitId });
    case "viewer":
      return withQuery("viewer", { run: route.runId });
    case "storage":
      return "#/storage";
    default:
      return "#/";
  }
}

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parseRoute(window.location.hash));
  useEffect(() => {
    const onChange = () => setRoute(parseRoute(window.location.hash));
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return route;
}

export function go(route: Route): void {
  window.location.hash = href(route);
}
