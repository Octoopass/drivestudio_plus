// Gốc của giao diện: chọn màn hình theo hash URL, bọc trong khung trang.
import { Layout } from "./components/Layout";
import { ComparePage } from "./pages/ComparePage";
import { NewRunPage } from "./pages/NewRunPage";
import { RunPage } from "./pages/RunPage";
import { RunsPage } from "./pages/RunsPage";
import { ScenePage } from "./pages/ScenePage";
import { ScenesPage } from "./pages/ScenesPage";
import { useRoute } from "./router";

export function App() {
  const route = useRoute();
  return (
    <Layout route={route}>
      {route.page === "scenes" && <ScenesPage />}
      {route.page === "scene" && <ScenePage key={route.sceneId} sceneId={route.sceneId} />}
      {route.page === "new" && <NewRunPage sceneId={route.sceneId} splitId={route.splitId} />}
      {route.page === "runs" && <RunsPage />}
      {route.page === "run" && <RunPage key={route.runId} runId={route.runId} />}
      {route.page === "compare" && <ComparePage sceneId={route.sceneId} splitId={route.splitId} />}
    </Layout>
  );
}
