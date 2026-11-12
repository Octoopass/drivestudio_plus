// Điều hướng bằng hash URL (#/scenes/<id>, #/runs/<id>...), không cần thư viện router: Studio
// chỉ có sáu màn hình và FastAPI phục vụ một file index.html duy nhất.
import { useEffect, useState } from "react";

export type Route =
  | { page: "scenes" }
  | { page: "scene"; sceneId: string }
  | { page: "new"; sceneId?: string; splitId?: string }
  | { page: "runs" }
  | { page: "run"; runId: string }
  | { page: "compare"; sceneId?: string; splitId?: string };

export function parseRoute(hash: string): Route {
  const [path, query = ""] = hash.replace(/^#/, "").split("?");
  const params = new URLSearchParams(query);
  const parts = path.split("/").filter(Boolean).map(decodeURIComponent);
  if (parts[0] === "scenes" && parts[1]) return { page: "scene", sceneId: parts[1] };
  if (parts[0] === "new") {
    return { page: "new", sceneId: params.get("scene") ?? undefined, splitId: params.get("split") ?? undefined };
  }
  if (parts[0] === "runs" && parts[1]) return { page: "run", runId: parts[1] };
  if (parts[0] === "runs") return { page: "runs" };
  if (parts[0] === "compare") {
    return { page: "compare", sceneId: params.get("scene") ?? undefined, splitId: params.get("split") ?? undefined };
  }
  return { page: "scenes" };
}

export function href(route: Route): string {
  const enc = encodeURIComponent;
  switch (route.page) {
    case "scene":
      return `#/scenes/${enc(route.sceneId)}`;
    case "run":
      return `#/runs/${enc(route.runId)}`;
    case "runs":
      return "#/runs";
    case "new":
    case "compare": {
      const params = new URLSearchParams();
      if (route.sceneId) params.set("scene", route.sceneId);
      if (route.splitId) params.set("split", route.splitId);
      const query = params.toString();
      return `#/${route.page}${query ? `?${query}` : ""}`;
    }
    default:
      return "#/scenes";
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
