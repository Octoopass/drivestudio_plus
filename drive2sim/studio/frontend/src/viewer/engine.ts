// Lõi của trang xem 3D: một cảnh three.js vẽ Gaussian splat bằng Spark (World Labs, MIT), điều khiển
// bằng chuột (OrbitControls). Tách khỏi React để trang chỉ gọi vài hàm: nạp file, đặt trục, đặt pose
// của xe theo khung, căn lại góc nhìn. Máy xem chỉ cần WebGL2 (ARCHITECTURE mục 2.13).
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

/** Trục "lên" của dữ liệu: DriveStudio dùng hệ world z lên; 3DGS gốc từ COLMAP thường y xuống. */
export type UpAxis = "z" | "-z" | "y" | "-y" | "x" | "-x";

const UP_VECTORS: Record<UpAxis, THREE.Vector3> = {
  z: new THREE.Vector3(0, 0, 1),
  "-z": new THREE.Vector3(0, 0, -1),
  y: new THREE.Vector3(0, 1, 0),
  "-y": new THREE.Vector3(0, -1, 0),
  x: new THREE.Vector3(1, 0, 0),
  "-x": new THREE.Vector3(-1, 0, 0),
};

/** Một camera của cảnh: pose camera→world (OpenCV, 16 số theo hàng) và ma trận K theo từng khung. */
export interface SceneCamera {
  name: string;
  width: number;
  height: number;
  c2w: number[][];
  K: number[][];
}

/** Giá trị ở phân vị q (0–1) của một mảng đã sắp. */
function quantile(sorted: Float32Array, q: number): number {
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor(q * (sorted.length - 1))))];
}

/** Vector riêng ứng với trị riêng nhỏ nhất của ma trận đối xứng 3x3 (phép quay Jacobi). */
function smallestEigenvector(m: number[][]): THREE.Vector3 {
  const a = m.map((row) => [...row]);
  const v = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  for (let sweep = 0; sweep < 30; sweep++) {
    for (const [p, q] of [
      [0, 1],
      [0, 2],
      [1, 2],
    ]) {
      if (Math.abs(a[p][q]) < 1e-12) continue;
      const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1);
      const s = t * c;
      for (let k = 0; k < 3; k++) {
        const akp = a[k][p];
        const akq = a[k][q];
        a[k][p] = c * akp - s * akq;
        a[k][q] = s * akp + c * akq;
      }
      for (let k = 0; k < 3; k++) {
        const apk = a[p][k];
        const aqk = a[q][k];
        a[p][k] = c * apk - s * aqk;
        a[q][k] = s * apk + c * aqk;
      }
      for (let k = 0; k < 3; k++) {
        const vkp = v[k][p];
        const vkq = v[k][q];
        v[k][p] = c * vkp - s * vkq;
        v[k][q] = s * vkp + c * vkq;
      }
    }
  }
  const index = [0, 1, 2].reduce((best, i) => (a[i][i] < a[best][best] ? i : best), 0);
  return new THREE.Vector3(v[0][index], v[1][index], v[2][index]);
}

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
  private spark: SparkRenderer;
  private upAxis: UpAxis | null = null;

  constructor(private host: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x0a0d10);
    host.appendChild(this.renderer.domElement);
    this.camera = new THREE.PerspectiveCamera(60, 1, 0.05, 5000);
    this.camera.position.set(0, 4, 12);
    this.spark = new SparkRenderer({ renderer: this.renderer });
    this.scene.add(this.spark);
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

  /** Xoay cả cảnh sao cho trục "lên" của dữ liệu thành trục y của three.js, rồi căn lại góc nhìn. */
  setUpAxis(axis: UpAxis): void {
    // Cùng trục thì không làm gì: không căn lại góc nhìn (giữ góc camera gốc vừa đặt).
    if (axis === this.upAxis) return;
    this.upAxis = axis;
    this.root.quaternion.setFromUnitVectors(UP_VECTORS[axis], new THREE.Vector3(0, 1, 0));
    this.root.updateMatrixWorld(true);
    if (this.meshes.length) this.frameAll();
  }

  /** Tâm của các splat đứng yên, lấy mẫu tối đa `limit` điểm (toạ độ của dữ liệu, chưa xoay). */
  private sampleCenters(limit = 200_000): Float32Array[] {
    const statics = this.meshes.filter((mesh) => ![...this.actors.values()].includes(mesh));
    const total = statics.reduce((sum, mesh) => sum + (mesh.splats?.getNumSplats() ?? 0), 0);
    const stride = Math.max(1, Math.ceil(total / limit));
    const xs: number[] = [];
    const ys: number[] = [];
    const zs: number[] = [];
    for (const mesh of statics) {
      mesh.forEachSplat((index, center) => {
        if (index % stride) return;
        xs.push(center.x);
        ys.push(center.y);
        zs.push(center.z);
      });
    }
    return [Float32Array.from(xs), Float32Array.from(ys), Float32Array.from(zs)];
  }

  /**
   * Đoán trục "lên" của một file splat lạ. Cảnh ngoài trời có mặt đất là lớp phẳng dày nhất: trục
   * ít biến thiên nhất của phần lõi (bỏ 10% điểm xa nhất) là pháp tuyến mặt đất. Chiều lên là chiều
   * từ lớp mặt đất (đỉnh mật độ) về phía có đuôi dài hơn: nhà, cây, cột mọc lên khỏi mặt đất, còn
   * dưới mặt đất gần như không có gì. Trả `null` nếu không đủ điểm.
   */
  estimateUpAxis(): UpAxis | null {
    const [xs, ys, zs] = this.sampleCenters(60_000);
    const n = xs.length;
    if (n < 100) return null;
    const median = [xs, ys, zs].map((values) => quantile(Float32Array.from(values).sort(), 0.5));
    const distances = new Float32Array(n);
    for (let i = 0; i < n; i++) distances[i] = Math.hypot(xs[i] - median[0], ys[i] - median[1], zs[i] - median[2]);
    const limit = quantile(Float32Array.from(distances).sort(), 0.9);
    const core: number[] = [];
    for (let i = 0; i < n; i++) if (distances[i] <= limit) core.push(i);
    const mean = [xs, ys, zs].map((values) => core.reduce((sum, i) => sum + values[i], 0) / core.length);
    const cov = [0, 1, 2].map((r) =>
      [0, 1, 2].map((c) => {
        const a = [xs, ys, zs][r];
        const b = [xs, ys, zs][c];
        return core.reduce((sum, i) => sum + (a[i] - mean[r]) * (b[i] - mean[c]), 0) / core.length;
      }),
    );
    const normal = smallestEigenvector(cov);
    const axisIndex = [Math.abs(normal.x), Math.abs(normal.y), Math.abs(normal.z)].reduce((best, v, i, all) => (v > all[best] ? i : best), 0);
    const values = Float32Array.from(core.map((i) => [xs, ys, zs][axisIndex][i])).sort();
    const low = quantile(values, 0.02);
    const high = quantile(values, 0.98);
    const bins = new Array(60).fill(0);
    for (const value of values) if (value >= low && value <= high) bins[Math.min(59, Math.floor(((value - low) / (high - low || 1)) * 60))]++;
    const peak = low + ((bins.indexOf(Math.max(...bins)) + 0.5) / 60) * (high - low);
    const positive = high - peak >= peak - low;
    const name = (["x", "y", "z"] as const)[axisIndex];
    return (positive ? name : `-${name}`) as UpAxis;
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

  /**
   * Đưa camera tới nhìn bao quát phần lõi của cảnh: hộp bao từ phân vị 2–98% của tâm splat trên mỗi
   * trục, không phải hộp bao mọi splat. Cảnh lái xe có điểm khởi tạo rải xa hàng trăm mét (phông nền,
   * bầu trời); lấy hộp bao toàn bộ thì camera lùi ra rất xa và con đường chỉ còn một chấm.
   */
  frameAll(): void {
    this.root.updateMatrixWorld(true);
    const [xs, ys, zs] = this.sampleCenters();
    if (!xs.length) return;
    const [sx, sy, sz] = [xs, ys, zs].map((values) => Float32Array.from(values).sort());
    const local = new THREE.Box3(
      new THREE.Vector3(quantile(sx, 0.02), quantile(sy, 0.02), quantile(sz, 0.02)),
      new THREE.Vector3(quantile(sx, 0.98), quantile(sy, 0.98), quantile(sz, 0.98)),
    );
    const box = local.applyMatrix4(this.root.matrixWorld);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const radius = Math.max(1, size.length() / 2);
    this.controls.target.copy(center);
    // Nhìn chếch từ trên xuống khoảng 50 độ: thấy được mặt đường giữa các nhà hai bên.
    this.camera.position.copy(center).add(new THREE.Vector3(0, radius * 0.75, radius * 0.65));
    this.camera.near = Math.max(0.01, radius / 2000);
    this.camera.far = radius * 40;
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  /**
   * Đặt góc nhìn đúng như một camera gốc ở một khung: pose camera→world theo quy ước OpenCV (x
   * phải, y xuống, z nhìn tới), trong toạ độ của dữ liệu; góc nhìn dọc lấy từ fy và chiều cao ảnh.
   */
  setCameraPose(c2w: number[], fy: number, height: number): void {
    this.root.updateMatrixWorld(true);
    const pose = new THREE.Matrix4().set(...(c2w as [number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number]));
    // OpenCV nhìn theo +z, y xuống; camera three.js nhìn theo −z, y lên: lật trục y và z.
    const flip = new THREE.Matrix4().makeScale(1, -1, -1);
    const world = this.root.matrixWorld.clone().multiply(pose).multiply(flip);
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    world.decompose(position, quaternion, new THREE.Vector3());
    this.camera.position.copy(position);
    this.camera.quaternion.copy(quaternion);
    this.camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(height / (2 * fy)));
    this.camera.near = 0.05;
    this.camera.far = 5000;
    this.camera.updateProjectionMatrix();
    // Tâm xoay đặt trước mặt camera 10 m, để kéo chuột thì xoay quanh cảnh trước mặt.
    this.controls.target.copy(position).add(new THREE.Vector3(0, 0, -10).applyQuaternion(quaternion));
    this.controls.update();
  }

  dispose(): void {
    this.clear();
    this.observer.disconnect();
    this.renderer.setAnimationLoop(null);
    this.controls.dispose();
    // Giải phóng tài nguyên GPU của Spark và trả hẳn context WebGL: trình duyệt chỉ cho khoảng 16
    // context cùng lúc, mỗi lần vào trang Xem 3D lại tạo một cái mới.
    this.spark.dispose();
    this.renderer.dispose();
    this.renderer.forceContextLoss();
    this.renderer.domElement.remove();
  }
}
