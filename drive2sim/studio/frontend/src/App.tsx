// Gốc của giao diện: chọn màn hình theo hash URL, bọc trong khung trang. Trang xem 3D nạp lười
// (three.js và Spark nặng khoảng 3 MB), nên các trang khác mở nhanh.
import { lazy, Suspense } from "react";
import { Layout } from "./components/Layout";
import { Loading } from "./components/ui";
import { ComparePage } from "./pages/ComparePage";
import { NewRunPage } from "./pages/NewRunPage";
import { OverviewPage } from "./pages/OverviewPage";
import { RunPage } from "./pages/RunPage";
import { RunsPage } from "./pages/RunsPage";
import { ScenePage } from "./pages/ScenePage";
import { ScenesPage } from "./pages/ScenesPage";
import { LibrariesPage } from "./pages/LibrariesPage";
import { StoragePage } from "./pages/StoragePage";
import { useRoute } from "./router";

const ViewerPage = lazy(() => import("./pages/ViewerPage").then((m) => ({ default: m.ViewerPage })));

export function App() {
  const route = useRoute();
  return (
    <Layout route={route}>
      {route.page === "overview" && <OverviewPage />}
      {route.page === "scenes" && <ScenesPage />}
      {route.page === "scene" && <ScenePage key={route.sceneId} sceneId={route.sceneId} />}
      {route.page === "new" && (
        <NewRunPage
          key={JSON.stringify(route)}
          sceneId={route.sceneId}
          splitId={route.splitId}
          approachId={route.approachId}
          preset={route.preset}
          executor={route.executor}
          quick={route.quick}
        />
      )}
      {route.page === "runs" && <RunsPage sceneId={route.sceneId} status={route.status} />}
      {route.page === "run" && <RunPage key={route.runId} runId={route.runId} />}
      {route.page === "compare" && <ComparePage sceneId={route.sceneId} splitId={route.splitId} />}
      {route.page === "storage" && <StoragePage focus={route.focus} />}
      {route.page === "libraries" && <LibrariesPage />}
      {route.page === "viewer" && (
        <Suspense fallback={<Loading />}>
          <ViewerPage runId={route.runId} />
        </Suspense>
      )}
    </Layout>
  );
}
