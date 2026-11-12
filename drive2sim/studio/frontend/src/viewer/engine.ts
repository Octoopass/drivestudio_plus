// Lõi của trang xem 3D: một cảnh three.js vẽ Gaussian splat bằng Spark (World Labs, MIT), điều khiển
// bằng chuột (OrbitControls). Tách khỏi React để trang chỉ gọi vài hàm: nạp file, đặt trục, đặt pose
// của xe theo khung, căn lại góc nhìn. Máy xem chỉ cần WebGL2 (ARCHITECTURE mục 2.13).
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

/** Trục "lên" của dữ liệu: DriveStudio dùng hệ world z lên; 3DGS gốc từ COLMAP thường y xuống. */
export type UpAxis = "z" | "-y" | "y";

export interface SplatSource {
  name: string;
  url?: string;
  bytes?: ArrayBuffer;
}

export class SplatViewer {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private root = new THREE.Group();
  private meshes: SplatMesh[] = [];
  private actors = new Map<string, SplatMesh>();
  private observer: ResizeObserver;
  private generation = 0;

  constructor(private host: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x0a0d10);
    host.appendChild(this.renderer.domElement);
    this.camera = new THREE.PerspectiveCamera(60, 1, 0.05, 5000);
    this.camera.position.set(0, 4, 12);
    this.scene.add(new SparkRenderer({ renderer: this.renderer }));
    this.scene.add(this.root);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.controls.screenSpacePanning = true;
    this.setUpAxis("z");
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(host);
    this.resize();
    this.renderer.setAnimationLoop(() => {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    });
  }

  private resize(): void {
    const { clientWidth: w, clientHeight: h } = this.host;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  /** Xoay cả cảnh sao cho trục "lên" của dữ liệu thành trục y của three.js. */
  setUpAxis(axis: UpAxis): void {
    this.root.rotation.set(axis === "z" ? -Math.PI / 2 : axis === "-y" ? Math.PI : 0, 0, 0);
    this.root.updateMatrixWorld(true);
  }

  /** Bậc spherical harmonics tối đa khi vẽ (0–3); thấp hơn thì nhẹ hơn cho card yếu. */
  setMaxSh(level: number): void {
    for (const mesh of this.meshes) mesh.maxSh = level;
  }

  clear(): void {
    this.generation += 1;
    for (const mesh of this.meshes) {
      this.root.remove(mesh);
      mesh.dispose();
    }
    this.meshes = [];
    this.actors.clear();
  }

  /**
   * Nạp một hoặc nhiều nguồn splat (thay những gì đang có). Nguồn có tên trong `actorIds` được
   * đặt theo pose từng khung bằng `setPoses`; còn lại đứng yên trong hệ world.
   * Trả về tổng số splat, hoặc null nếu một lần nạp mới hơn đã thay thế lần này.
   */
  async load(sources: SplatSource[], actorIds: string[] = [], onProgress?: (fraction: number) => void): Promise<number | null> {
    this.clear();
    const generation = this.generation;
    const loaded = new Array(sources.length).fill(0);
    const meshes = sources.map((source, index) => {
      const mesh = new SplatMesh({
        url: source.url,
        fileBytes: source.bytes,
        fileName: source.name,
        onProgress: (event) => {
          if (event.lengthComputable) loaded[index] = event.loaded / event.total;
          onProgress?.(loaded.reduce((a, b) => a + b, 0) / sources.length);
        },
      });
      if (actorIds.includes(source.name)) {
        mesh.matrixAutoUpdate = false;
        mesh.visible = false;
        this.actors.set(source.name, mesh);
      }
      this.root.add(mesh);
      return mesh;
    });
    this.meshes = meshes;
    await Promise.all(meshes.map((mesh) => mesh.initialized));
    if (generation !== this.generation) return null;
    this.frameAll();
    return meshes.reduce((sum, mesh) => sum + (mesh.splats?.getNumSplats() ?? 0), 0);
  }

  /** Đặt pose của từng xe ở một khung: ma trận 4x4 theo hàng, null là xe không có mặt. */
  setPoses(trackIds: string[], poses: (number[] | null)[]): void {
    trackIds.forEach((trackId, index) => {
      const mesh = this.actors.get(trackId);
      if (!mesh) return;
      const pose = poses[index];
      mesh.visible = pose != null;
      if (pose) {
        mesh.matrix.set(...(pose as [number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number]));
        mesh.matrixWorldNeedsUpdate = true;
      }
    });
  }

  /** Đưa camera tới nhìn bao quát mọi splat đứng yên. Dùng hộp bao tâm splat; bán kính giới hạn
   * 400 m để vài splat bầu trời ở xa không đẩy camera ra quá xa cảnh. */
  frameAll(): void {
    const box = new THREE.Box3();
    this.root.updateMatrixWorld(true);
    for (const mesh of this.meshes) {
      if (this.actors.size && [...this.actors.values()].includes(mesh)) continue;
      const local = mesh.getBoundingBox(true);
      if (!local.isEmpty()) box.union(local.clone().applyMatrix4(mesh.matrixWorld));
    }
    if (box.isEmpty()) return;
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const radius = Math.max(1, Math.min(size.length() / 2, 400));
    this.controls.target.copy(center);
    // Nhìn chếch từ trên xuống khoảng 50 độ: thấy được mặt đường giữa các nhà hai bên.
    this.camera.position.copy(center).add(new THREE.Vector3(0, radius * 0.75, radius * 0.65));
    this.camera.near = Math.max(0.01, radius / 2000);
    this.camera.far = radius * 40;
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  dispose(): void {
    this.clear();
    this.observer.disconnect();
    this.renderer.setAnimationLoop(null);
    this.controls.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
