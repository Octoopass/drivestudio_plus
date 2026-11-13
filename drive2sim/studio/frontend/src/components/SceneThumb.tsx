// Ảnh đại diện của một cảnh: ảnh của camera đầu tiên ở khung 0.
import { ImageOff } from "lucide-react";
import { seg, useApi } from "../api/client";
import type { FrameImages } from "../api/types";

export function SceneThumb({ sceneId, frame = 0 }: { sceneId: string; frame?: number }) {
  const images = useApi<FrameImages>(`/api/scenes/${seg(sceneId)}/frames/${frame}`);
  const url = images.data?.images.find((image) => image.url)?.url;
  if (url) return <img src={url} alt="" loading="lazy" />;
  return (
    <div style={{ display: "grid", placeItems: "center", height: "100%", color: "var(--faint)" }}>
      {images.data && <ImageOff size={25} />}
    </div>
  );
}
