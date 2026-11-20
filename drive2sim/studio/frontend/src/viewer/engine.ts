// Lõi của trang xem 3D: một cảnh three.js vẽ Gaussian splat bằng Spark (World Labs, MIT), điều khiển
// bằng chuột (OrbitControls). Tách khỏi React để trang chỉ gọi vài hàm: nạp file, đặt trục, đặt pose
// của xe theo khung, căn lại góc nhìn. Máy xem chỉ cần WebGL2 (ARCHITECTURE mục 2.13).
import { SparkRenderer, SplatMesh, SplatSkinning, SplatSkinningMode } from "@sparkjsdev/spark";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

/** Trục "lên" của dữ liệu: DriveStudio dùng hệ world z lên; 3DGS gốc từ COLMAP thường y xuống. */
export type UpAxis = "z" | "-z" | "y" | "-y" | "x" | "-x";

/** K của camera gốc (điểm ảnh) và cỡ ảnh gốc. */
type Intrinsics = { fx: number; fy: number; cx: number; cy: number; width: number; height: number };

/** Góc nhìn dọc (độ) của khung nhìn bao quát. */
export const DEFAULT_FOV = 60;

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

/**
 * Trục "lên" của hệ world suy từ pose camera (như ``up_axis_from_cameras`` khi xuất PLY). Hệ world
 * của DriveStudio lấy theo camera trước ở khung đầu, nên trục lên tuỳ dataset (nuScenes, KITTI:
 * -y; Waymo: z). Camera gắn trên xe gần thẳng đứng: cộng hướng lên (-y của camera OpenCV, cột 2 của
 * c2w) của mọi camera, mọi khung rồi chọn trục gần nhất. `null` nếu không có camera.
 */
export function upAxisFromCameras(cameras: SceneCamera[]): UpAxis | null {
  const total = [0, 0, 0];
  for (const camera of cameras)
    for (const m of camera.c2w) for (let r = 0; r < 3; r++) total[r] -= m[r * 4 + 1];
  const index = [0, 1, 2].reduce((best, i) => (Math.abs(total[i]) > Math.abs(total[best]) ? i : best), 0);
  if (!total[index]) return null;
  const name = (["x", "y", "z"] as const)[index];
  return (total[index] > 0 ? name : `-${name}`) as UpAxis;
}

/**
 * Đường camera "novel" giống video `novel/front_center_interp.mp4` của DriveStudio
 * (`front_center_interp` và `interpolate_poses` trong `drive2sim/recon/utils/camera.py`): lấy pose
 * của camera đầu tiên ở các khung `[::F // 4]` làm điểm mốc, rồi nội suy ra `F` khung trải đều
 * từ mốc đầu tới mốc cuối, vị trí theo đường thẳng, hướng bằng slerp. Khung `i` của đường này ứng
 * với thời điểm `i` của cảnh, như lúc DriveStudio render video novel.
 *
 * @param c2w Pose camera→world (16 số theo hàng) của camera đầu tiên ở từng khung.
 * @returns `F` pose cùng dạng.
 */
export function novelTrajectory(c2w: number[][]): number[][] {
  const frames = c2w.length;
  if (frames < 2) return c2w.slice();
  const step = Math.max(1, Math.floor(frames / 4));
  const keys: number[][] = [];
  for (let index = 0; index < frames; index += step) keys.push(c2w[index]);
  if (keys.length < 2) return c2w.slice();
  const pose = new THREE.Matrix4();
  const out: number[][] = [];
  for (let i = 0; i < frames; i++) {
    const t = (i / (frames - 1)) * (keys.length - 1);
    const segment = Math.min(Math.floor(t), keys.length - 2);
    blendPose(keys[segment], keys[segment + 1], t - segment, pose);
    out.push(pose.clone().transpose().toArray());
  }
  return out;
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

type Matrix16 = [number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number];

const _posA = new THREE.Vector3();
const _posB = new THREE.Vector3();
const _rotA = new THREE.Quaternion();
const _rotB = new THREE.Quaternion();
const _scale = new THREE.Vector3();

/**
 * Pose nằm giữa hai pose cứng (ma trận 4x4 theo hàng): vị trí nội suy tuyến tính, hướng nội suy
 * cầu (slerp), `alpha` 0 là `a`, 1 là `b`. Ghi kết quả vào `out`.
 */
export function blendPose(a: number[], b: number[], alpha: number, out: THREE.Matrix4): THREE.Matrix4 {
  out.set(...(a as Matrix16));
  if (alpha <= 0 || a === b) return out;
  out.decompose(_posA, _rotA, _scale);
  new THREE.Matrix4().set(...(b as Matrix16)).decompose(_posB, _rotB, new THREE.Vector3());
  return out.compose(_posA.lerp(_posB, alpha), _rotA.slerp(_rotB, alpha), _scale);
}

const _boneMatrix = new THREE.Matrix4();

/** Đọc xương ở vị trí `offset` của `bones` (3 cột phần xoay, rồi phần dời) thành hướng và vị trí. */
function boneQuatPos(bones: Float32Array, offset: number, position: THREE.Vector3, rotation: THREE.Quaternion): void {
  const o = offset;
  _boneMatrix.set(
    bones[o], bones[o + 3], bones[o + 6], bones[o + 9],
    bones[o + 1], bones[o + 4], bones[o + 7], bones[o + 10],
    bones[o + 2], bones[o + 5], bones[o + 8], bones[o + 11],
    0, 0, 0, 1,
  );
  _boneMatrix.decompose(position, rotation, _scale);
}

/** Đường bay của camera khi nhấp đúp: vị trí và tâm xoay đi từ `from*` tới `to*` trong `duration` ms. */
interface Flight {
  fromPosition: THREE.Vector3;
  toPosition: THREE.Vector3;
  fromTarget: THREE.Vector3;
  toTarget: THREE.Vector3;
  start: number;
  duration: number;
}

export interface SplatSource {
  name: string;
  url?: string;
  bytes?: ArrayBuffer;
}

/** Một người của gói xem trước: Gaussian ở dáng gốc và skin (4 xương mỗi Gaussian). */
export interface HumanSource {
  name: string;
  url: string;
  skinUrl: string;
}

/** Xương và vị trí của mọi người theo khung (CONTRACTS mục 7). */
export interface HumanMotion {
  /** float32 (F, H, 24, 12): mỗi xương 3 cột phần xoay rồi phần dời. */
  bones: Float32Array;
  track_ids: string[];
  /** [F][H]: ma trận 4x4 theo hàng (chỉ dời), null khi người không có mặt. */
  T_object_world: (number[] | null)[][];
}

const NUM_BONES = 24;
const HUMAN_SH_DEGREE = 1;
const BONE_FLOATS = 12;

/**
 * Người uốn theo xương bằng `SplatSkinning` của Spark, chế độ dual quaternion (chạy trên GPU,
 * chỉ cập nhật 24 xương mỗi khung). OmniRe dùng linear blend với trọng số không chuẩn hoá; dual
 * quaternion chuẩn hoá trọng số, lệch trung bình 4 mm, 99% điểm dưới 3,3 cm so với OmniRe (đo trên
 * run f8a6). Chế độ linear blend của Spark 2.4.0 cần covSplats, mà đường đó thiếu bộ sắp xếp độ
 * sâu nên không vẽ được.
 */
interface Human {
  mesh: SplatMesh;
  skinning: SplatSkinning;
  /** Chỉ số của người trong `HumanMotion.track_ids`. */
  index: number;
}

const SKY_VERTEX = /* glsl */ `
  varying vec3 vWorld;
  void main() {
    vec4 world = modelMatrix * vec4(position, 1.0);
    vWorld = world.xyz;
    gl_Position = projectionMatrix * viewMatrix * world;
    gl_Position.z = gl_Position.w;  // luôn ở mặt xa nhất
  }`;

// Tra cubemap EnvLight của OmniRe: hướng nhìn đổi sang hệ dữ liệu, nhân to_opengl, rồi chọn mặt và
// (s, t) theo bảng OpenGL như nvdiffrast; 6 mặt xếp 3 cột × 2 hàng trong một ảnh.
const SKY_FRAGMENT = /* glsl */ `
  uniform sampler2D atlas;
  uniform mat3 worldToData;
  uniform float faceSize;
  varying vec3 vWorld;
  void main() {
    vec3 l = worldToData * normalize(vWorld - cameraPosition);
    vec3 d = vec3(l.x, l.z, -l.y);
    vec3 a = abs(d);
    float face; float sc; float tc; float ma;
    if (a.x >= a.y && a.x >= a.z) { face = d.x > 0.0 ? 0.0 : 1.0; sc = d.x > 0.0 ? -d.z : d.z; tc = -d.y; ma = a.x; }
    else if (a.y >= a.z) { face = d.y > 0.0 ? 2.0 : 3.0; sc = d.x; tc = d.y > 0.0 ? d.z : -d.z; ma = a.y; }
    else { face = d.z > 0.0 ? 4.0 : 5.0; sc = d.z > 0.0 ? d.x : -d.x; tc = -d.y; ma = a.z; }
    float margin = 0.5 / faceSize;
    vec2 st = clamp(vec2(sc, tc) / ma * 0.5 + 0.5, margin, 1.0 - margin);
    vec2 uv = (vec2(mod(face, 3.0), floor(face / 3.0)) + st) / vec2(3.0, 2.0);
    gl_FragColor = vec4(texture2D(atlas, uv).rgb, 1.0);
  }`;

export class SplatViewer {
  private renderer: THREE.WebGLRenderer;
  private fovOverride: number | null = null;
  // Góc nhìn và K của camera gốc gần nhất (setCameraPose), để bỏ góc tự chọn thì quay lại được.
  private native: { fov: number; intrinsics: Intrinsics } | null = null;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private root = new THREE.Group();
  private meshes: SplatMesh[] = [];
  private actors = new Map<string, SplatMesh>();
  private humans: Human[] = [];
  private shLevel = 3;
  private motion: HumanMotion | null = null;
  private sky: THREE.Mesh<THREE.SphereGeometry, THREE.ShaderMaterial> | null = null;
  private observer: ResizeObserver;
  private generation = 0;
  private spark: SparkRenderer;
  private upAxis: UpAxis | null = null;
  private flight: Flight | null = null;
  // Vẽ theo nhu cầu: chỉ vẽ tới mốc `renderUntil` sau mỗi thay đổi (camera, pose, nạp), rồi nghỉ.
  // Spark sắp xếp splat không đồng bộ sau khi camera đổi, nên giữ vẽ thêm một lúc để bản sắp
  // xếp mới kịp hiện. Đứng yên thì không tốn GPU, trang không bị giật.
  private renderUntil = 0;
  /** Bước chỉnh màu sau khi vẽ (setColorAffine); null là vẽ thẳng ra màn hình. */
  private affine: { scene: THREE.Scene; camera: THREE.OrthographicCamera; material: THREE.ShaderMaterial } | null = null;
  private colorTarget: THREE.FramebufferTexture | null = null;
  /** K của camera gốc đang dùng (setCameraPose), null khi nhìn tự do bao quát. */
  private intrinsics: Intrinsics | null = null;
  /** Gọi mỗi khi camera đổi (vị trí, hướng, phép chiếu), để trang ẩn ảnh chính xác cũ. */
  onCameraChange?: () => void;
  private lastView = new THREE.Matrix4();
  /** Độ nghiêng của camera gốc quanh trục nhìn, áp sau OrbitControls (xem `frame`). */
  private roll = new THREE.Quaternion();
  private lastProjection = new THREE.Matrix4();
  // Luôn vẽ đủ độ phân giải màn hình: ưu tiên nét, không hạ độ phân giải lúc chuyển động.
  private basePixelRatio = Math.min(window.devicePixelRatio, 2);
  private raycaster = new THREE.Raycaster();
  /** Gọi khi người xem nhấp đúp bay tới một điểm (trang tắt "bám camera" để camera không bị kéo về). */
  onFly?: () => void;

  constructor(private host: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x0a0d10);
    host.appendChild(this.renderer.domElement);
    this.camera = new THREE.PerspectiveCamera(DEFAULT_FOV, 1, 0.05, 5000);
    this.camera.position.set(0, 4, 12);
    this.spark = new SparkRenderer({ renderer: this.renderer });
    this.renderer.setPixelRatio(this.basePixelRatio);
    this.scene.add(this.spark);
    this.scene.add(this.root);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.controls.screenSpacePanning = true;
    // Người xem kéo chuột giữa lúc đang bay thì dừng bay, nhường cho chuột.
    this.controls.addEventListener("start", () => {
      this.flight = null;
    });
    this.controls.addEventListener("change", () => this.invalidate());
    this.setUpAxis("z");
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(host);
    this.resize();
    this.renderer.domElement.addEventListener("dblclick", this.onDoubleClick);
    this.renderer.setAnimationLoop(() => this.frame());
  }

  // Mốc thời gian của các lần vẽ trong khoảng 1 giây gần nhất, để đếm FPS thật (``renderRate``).
  private renderTimes: number[] = [];

  /** Số khung đã vẽ trong 1 giây gần nhất; ``idle`` khi viewer đang không vẽ lại (cảnh đứng yên). */
  renderRate(): { fps: number; idle: boolean } {
    const now = performance.now();
    while (this.renderTimes.length && now - this.renderTimes[0] > 1000) this.renderTimes.shift();
    return { fps: this.renderTimes.length, idle: now > this.renderUntil };
  }

  /** Yêu cầu vẽ lại trong 0,8 giây tới. */
  invalidate(): void {
    this.renderUntil = performance.now() + 800;
  }

  private frame(): void {
    if (this.flight) this.invalidate();
    if (this.controls.update()) this.invalidate();
    // OrbitControls luôn dựng camera thẳng đứng (lookAt theo trục lên); trả lại độ nghiêng của
    // camera gốc (xe nghiêng, đường dốc), nếu không ảnh lệch vài điểm ảnh so với render gốc.
    this.camera.quaternion.multiply(this.roll);
    this.stepFlight();
    const now = performance.now();
    if (now > this.renderUntil) return;
    this.renderTimes.push(now);
    this.camera.updateMatrixWorld();
    if (!this.lastView.equals(this.camera.matrixWorld) || !this.lastProjection.equals(this.camera.projectionMatrix)) {
      this.lastView.copy(this.camera.matrixWorld);
      this.lastProjection.copy(this.camera.projectionMatrix);
      this.onCameraChange?.();
    }
    if (this.sky) {
      // Trời ở vô cực: quả cầu đi theo camera, bán kính trong tầm nhìn xa.
      this.sky.position.copy(this.camera.position);
      this.sky.scale.setScalar(this.camera.far * 0.5);
      const toData = new THREE.Matrix3().setFromMatrix4(this.root.matrixWorld).invert();
      this.sky.material.uniforms.worldToData.value.copy(toData);
    }
    if (!this.affine) {
      this.renderer.render(this.scene, this.camera);
      return;
    }
    // Chỉnh màu của model (AffineTransform): rgb' = A·rgb + b trên ảnh đã ghép (splat + trời). Phép
    // biến đổi tuyến tính nên áp sau khi ghép cho đúng kết quả như áp lên từng Gaussian và trời.
    // Vẽ thẳng ra màn hình như khi không chỉnh rồi chép lại ảnh màn hình: vẽ vào bộ đệm riêng thì
    // Spark trộn các splat trong màu tuyến tính, sáng hơn ảnh gốc (đo: +4 mức với ma trận đơn vị).
    const size = this.renderer.getDrawingBufferSize(new THREE.Vector2());
    if (!this.colorTarget || this.colorTarget.image.width !== size.x || this.colorTarget.image.height !== size.y) {
      this.colorTarget?.dispose();
      this.colorTarget = new THREE.FramebufferTexture(size.x, size.y);
      this.affine.material.uniforms.image.value = this.colorTarget;
    }
    this.renderer.render(this.scene, this.camera);
    this.renderer.copyFramebufferToTexture(this.colorTarget);
    this.renderer.render(this.affine.scene, this.affine.camera);
  }

  /**
   * Ma trận chỉnh màu 3×4 theo hàng (`rgb' = A·rgb + b`) của ảnh (camera, khung) đang xem, như
   * renderer gốc áp; null là không chỉnh.
   */
  setColorAffine(matrix: ArrayLike<number> | null): void {
    if (!matrix) {
      this.affine = null;
      this.invalidate();
      return;
    }
    if (!this.affine) {
      const material = new THREE.ShaderMaterial({
        uniforms: { image: { value: this.colorTarget }, A: { value: new THREE.Matrix3() }, b: { value: new THREE.Vector3() } },
        vertexShader: "varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }",
        fragmentShader:
          // Ảnh màn hình đã ở màu hiển thị, cùng không gian với ảnh train nơi model học ma trận.
          "uniform sampler2D image; uniform mat3 A; uniform vec3 b; varying vec2 vUv;" +
          "void main() { vec3 c = texture2D(image, vUv).rgb; gl_FragColor = vec4(clamp(A * c + b, 0.0, 1.0), 1.0); }",
        depthTest: false,
        depthWrite: false,
      });
      const scene = new THREE.Scene();
      scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material));
      this.affine = { scene, camera: new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1), material };
    }
    const m = matrix;
    this.affine.material.uniforms.A.value.set(m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]);
    this.affine.material.uniforms.b.value.set(m[3], m[7], m[11]);
    this.invalidate();
  }

  /**
   * Cách làm mờ splat theo cách model được train: gsplat không chống răng cưa thì cộng 0,3 vào
   * kích thước đốm, không bù độ đục (`preBlurAmount`); có chống răng cưa thì cộng và bù độ đục
   * (`blurAmount`, mặc định của Spark).
   */
  setAntialiased(antialiased: boolean): void {
    this.spark.preBlurAmount = antialiased ? 0 : 0.3;
    this.spark.blurAmount = antialiased ? 0.3 : 0;
    this.invalidate();
  }

  /** Đặt ảnh trời (cubemap EnvLight của OmniRe, 3 cột × 2 hàng); null là bỏ trời. */
  async setSky(url: string | null): Promise<void> {
    if (this.sky) {
      this.scene.remove(this.sky);
      this.sky.geometry.dispose();
      this.sky.material.uniforms.atlas.value?.dispose();
      this.sky.material.dispose();
      this.sky = null;
    }
    if (!url) return this.invalidate();
    const generation = this.generation;
    const texture = await new THREE.TextureLoader().loadAsync(url);
    if (generation !== this.generation) return texture.dispose();
    texture.flipY = false;
    texture.colorSpace = THREE.NoColorSpace;  // màu như ảnh train, không đổi không gian màu
    texture.generateMipmaps = false;
    texture.minFilter = THREE.LinearFilter;
    texture.needsUpdate = true;
    const material = new THREE.ShaderMaterial({
      vertexShader: SKY_VERTEX,
      fragmentShader: SKY_FRAGMENT,
      uniforms: {
        atlas: { value: texture },
        worldToData: { value: new THREE.Matrix3() },
        faceSize: { value: texture.image.height / 2 },
      },
      side: THREE.BackSide,
      depthWrite: false,
      depthTest: false,
    });
    this.sky = new THREE.Mesh(new THREE.SphereGeometry(1, 64, 32), material);
    this.sky.renderOrder = -1;
    this.sky.frustumCulled = false;
    this.scene.add(this.sky);
    this.invalidate();
  }

  /**
   * Nạp người (Gaussian dáng gốc, skin) và chuyển động của họ. Trả về số splat của người, hoặc
   * null nếu một lần nạp mới hơn đã thay thế lần này.
   */
  async loadHumans(sources: HumanSource[], motion: HumanMotion): Promise<number | null> {
    const generation = this.generation;
    const loaded = await Promise.all(
      sources.map(async (source) => {
        const skin = new Uint16Array(await (await fetch(source.skinUrl)).arrayBuffer());
        const mesh = new SplatMesh({ url: source.url, fileName: `${source.name}.spz` });
        await mesh.initialized;
        return { source, skin, mesh };
      }),
    );
    if (generation !== this.generation) {
      for (const { mesh } of loaded) mesh.dispose();
      return null;
    }
    this.motion = motion;
    for (const { source, skin, mesh } of loaded) {
      const skinning = new SplatSkinning({ mesh, numSplats: mesh.numSplats, numBones: NUM_BONES, mode: SplatSkinningMode.DUAL_QUATERNION });
      skinning.skinData.set(skin.subarray(0, mesh.numSplats * 4));
      skinning.skinTexture.needsUpdate = true;
      mesh.skinning = skinning;
      mesh.updateGenerator();
      mesh.matrixAutoUpdate = false;
      mesh.visible = false;
      this.root.add(mesh);
      this.meshes.push(mesh);
      this.humans.push({ mesh, skinning, index: motion.track_ids.indexOf(source.name) });
    }
    this.setMaxSh(this.shLevel);
    this.setHumanFrame(0);
    return loaded.reduce((sum, { mesh }) => sum + mesh.numSplats, 0);
  }

  /** Đặt dáng và vị trí của người ở khung `index` (nội suy tới `next` theo `alpha` khi phát). */
  setHumanFrame(index: number, next = index, alpha = 0): void {
    const motion = this.motion;
    if (!motion) return;
    const count = motion.track_ids.length;
    const frames = motion.T_object_world.length;
    const a = Math.min(index, frames - 1);
    const b = Math.min(next, frames - 1);
    for (const human of this.humans) {
      const pose = motion.T_object_world[a]?.[human.index];
      const after = motion.T_object_world[b]?.[human.index];
      human.mesh.visible = pose != null;
      if (!pose) continue;
      // Người chỉ có mặt ở khung này thì không nội suy.
      const t = after ? alpha : 0;
      for (let bone = 0; bone < NUM_BONES; bone++) {
        boneQuatPos(motion.bones, ((a * count + human.index) * NUM_BONES + bone) * BONE_FLOATS, _posA, _rotA);
        if (t > 0) {
          boneQuatPos(motion.bones, ((b * count + human.index) * NUM_BONES + bone) * BONE_FLOATS, _posB, _rotB);
          _posA.lerp(_posB, t);
          _rotA.slerp(_rotB, t);
        }
        human.skinning.setBoneQuatPos(bone, _rotA, _posA);
      }
      human.skinning.updateBones();
      const end = after ?? pose;
      human.mesh.matrix.makeTranslation(
        pose[3] + (end[3] - pose[3]) * t,
        pose[7] + (end[7] - pose[7]) * t,
        pose[11] + (end[11] - pose[11]) * t,
      );
      human.mesh.matrixWorldNeedsUpdate = true;
    }
    this.invalidate();
  }

  /** Nhấp đúp vào cảnh: tìm splat trúng tia từ chuột rồi bay tới gần điểm đó. */
  private onDoubleClick = (event: MouseEvent): void => {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const pointer = new THREE.Vector2(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
    this.raycaster.setFromCamera(pointer, this.camera);
    const hit = this.raycaster.intersectObjects(this.meshes.filter((mesh) => mesh.visible), false)[0];
    if (!hit) return;
    this.flyTo(hit.point);
    this.onFly?.();
  };

  /**
   * Bay camera tới gần `point` (toạ độ three.js), giữ hướng nhìn hiện tại: dừng ở 40% quãng
   * đường, trong khoảng 2–30 m trước điểm đó; tâm xoay chuyển sang điểm đó.
   */
  flyTo(point: THREE.Vector3, duration = 700): void {
    const direction = point.clone().sub(this.camera.position);
    const distance = direction.length();
    if (distance < 1e-6) return;
    const stop = THREE.MathUtils.clamp(distance * 0.4, Math.min(2, distance), 30);
    this.flight = {
      fromPosition: this.camera.position.clone(),
      toPosition: point.clone().sub(direction.normalize().multiplyScalar(stop)),
      fromTarget: this.controls.target.clone(),
      toTarget: point.clone(),
      start: performance.now(),
      duration,
    };
  }

  private stepFlight(): void {
    const flight = this.flight;
    if (!flight) return;
    const linear = Math.min(1, (performance.now() - flight.start) / flight.duration);
    const eased = linear < 0.5 ? 2 * linear * linear : 1 - (-2 * linear + 2) ** 2 / 2;
    this.camera.position.lerpVectors(flight.fromPosition, flight.toPosition, eased);
    this.controls.target.lerpVectors(flight.fromTarget, flight.toTarget, eased);
    if (linear >= 1) this.flight = null;
  }

  private resize(): void {
    const { clientWidth: w, clientHeight: h } = this.host;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.updateProjection();
    this.invalidate();
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
    const moving = new Set([...this.actors.values(), ...this.humans.map((human) => human.mesh)]);
    const statics = this.meshes.filter((mesh) => !moving.has(mesh));
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
    this.shLevel = level;
    // Người chỉ có SH bậc 1 (SMPLNodes); đặt cao hơn thì Spark đọc phần bậc 2–3 không có trong file
    // và cả cảnh lấm tấm nhiễu (thấy ở run f8a6).
    const humans = new Set(this.humans.map((human) => human.mesh));
    for (const mesh of this.meshes) mesh.maxSh = humans.has(mesh) ? Math.min(level, HUMAN_SH_DEGREE) : level;
    this.invalidate();
  }

  clear(): void {
    this.generation += 1;
    for (const mesh of this.meshes) {
      this.root.remove(mesh);
      mesh.dispose();
    }
    this.meshes = [];
    this.actors.clear();
    this.humans = [];
    this.motion = null;
    void this.setSky(null);
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
    // Nạp xong vẫn còn giải mã, sắp xếp trong nền một lúc: giữ vẽ 3 giây.
    this.renderUntil = performance.now() + 3000;
    return meshes.reduce((sum, mesh) => sum + (mesh.splats?.getNumSplats() ?? 0), 0);
  }

  /**
   * Đặt pose của từng xe: ma trận 4x4 theo hàng, null là xe không có mặt. Có `next` và `alpha` thì
   * đặt pose nằm giữa khung này và khung sau (phát mượt); xe chỉ có mặt ở một trong hai khung thì
   * lấy đúng pose của khung này, không nội suy.
   */
  setPoses(trackIds: string[], poses: (number[] | null)[], next?: (number[] | null)[], alpha = 0): void {
    trackIds.forEach((trackId, index) => {
      const mesh = this.actors.get(trackId);
      if (!mesh) return;
      const pose = poses[index];
      mesh.visible = pose != null;
      if (pose) {
        blendPose(pose, next?.[index] ?? pose, alpha, mesh.matrix);
        mesh.matrixWorldNeedsUpdate = true;
      }
    });
    this.invalidate();
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
    this.invalidate();
    // Nhìn chếch từ trên xuống khoảng 50 độ: thấy được mặt đường giữa các nhà hai bên.
    this.camera.position.copy(center).add(new THREE.Vector3(0, radius * 0.75, radius * 0.65));
    this.camera.near = Math.max(0.01, radius / 2000);
    this.camera.far = radius * 40;
    this.camera.fov = this.fovOverride ?? DEFAULT_FOV;
    this.intrinsics = null;
    this.native = null;
    this.roll.identity();
    this.updateProjection();
    this.controls.update();
  }

  /**
   * Góc nhìn dọc do người xem chọn (độ), dùng thay K của camera gốc; ``null`` là theo camera gốc
   * (khung nhìn bao quát thì 60°). Thử nghiệm cho camera fisheye: K của fisheye đưa vào phép chiếu
   * pinhole cho góc nhìn khoảng 119° (ARCHITECTURE mục 2.13).
   */
  setFovOverride(deg: number | null): void {
    this.fovOverride = deg;
    if (deg !== null) {
      this.camera.fov = deg;
      this.intrinsics = null;
    } else {
      // Về theo camera gốc: K của camera gốc gần nhất, hoặc 60° ở khung nhìn bao quát.
      this.camera.fov = this.native?.fov ?? DEFAULT_FOV;
      this.intrinsics = this.native?.intrinsics ?? null;
    }
    this.updateProjection();
    this.invalidate();
  }

  /** Góc nhìn dọc đang dùng (độ). */
  verticalFov(): number {
    return this.camera.fov;
  }

  /**
   * Đặt góc nhìn đúng như một camera gốc ở một khung: pose camera→world theo quy ước OpenCV (x
   * phải, y xuống, z nhìn tới), trong toạ độ của dữ liệu; góc nhìn dọc lấy từ fy và chiều cao ảnh.
   */
  setCameraPose(c2w: number[], K: number[], width: number, height: number, next?: number[], alpha = 0): void {
    this.flight = null;
    this.root.updateMatrixWorld(true);
    const pose = blendPose(c2w, next ?? c2w, alpha, new THREE.Matrix4());
    // OpenCV nhìn theo +z, y xuống; camera three.js nhìn theo −z, y lên: lật trục y và z.
    const flip = new THREE.Matrix4().makeScale(1, -1, -1);
    const world = this.root.matrixWorld.clone().multiply(pose).multiply(flip);
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    world.decompose(position, quaternion, new THREE.Vector3());
    this.camera.position.copy(position);
    this.camera.quaternion.copy(quaternion);
    this.camera.near = 0.05;
    this.camera.far = 5000;
    this.native = {
      fov: THREE.MathUtils.radToDeg(2 * Math.atan(height / (2 * K[4]))),
      intrinsics: { fx: K[0], fy: K[4], cx: K[2], cy: K[5], width, height },
    };
    if (this.fovOverride === null) {
      this.camera.fov = this.native.fov;
      this.intrinsics = this.native.intrinsics;
    } else {
      // Góc nhìn người xem chọn: pinhole đối xứng, giữ pose của camera gốc, bỏ K của nó.
      this.camera.fov = this.fovOverride;
      this.intrinsics = null;
    }
    this.updateProjection();
    // Tâm xoay đặt trước mặt camera 10 m, để kéo chuột thì xoay quanh cảnh trước mặt.
    this.controls.target.copy(position).add(new THREE.Vector3(0, 0, -10).applyQuaternion(quaternion));
    this.controls.update();
    // Phần nghiêng (quanh trục nhìn) mà OrbitControls bỏ đi, áp lại ở mỗi khung hình.
    this.roll.copy(this.camera.quaternion).invert().multiply(quaternion);
    this.camera.quaternion.copy(quaternion);
    this.invalidate();
  }

  /**
   * Phép chiếu theo đúng K của camera gốc (fx, fy, cx, cy), như renderer của DriveStudio: ảnh gốc
   * được co theo chiều cao canvas và đặt giữa, tâm ảnh lệch đúng (cx − W/2, cy − H/2) điểm ảnh.
   * Không có K (nhìn bao quát, file kéo thả) thì là phép chiếu đối xứng của three.js. Canvas rộng
   * hơn ảnh gốc thì thấy thêm hai bên.
   */
  private updateProjection(): void {
    this.camera.updateProjectionMatrix();
    const k = this.intrinsics;
    if (!k) return;
    const { clientWidth: cw, clientHeight: ch } = this.host;
    if (!cw || !ch) return;
    const scale = ch / k.height;
    const e = this.camera.projectionMatrix.elements;
    e[0] = (2 * k.fx * scale) / cw;
    e[5] = (2 * k.fy) / k.height;
    e[8] = (-2 * (k.cx - k.width / 2) * scale) / cw;
    e[9] = (2 * k.cy) / k.height - 1;
    this.camera.projectionMatrixInverse.copy(this.camera.projectionMatrix).invert();
  }

  /**
   * Pose camera hiện tại trong hệ dữ liệu (OpenCV, 16 số theo hàng) và K cho ảnh `width` × `height`
   * cùng khung nhìn với canvas: để renderer gốc vẽ đúng ảnh đang thấy (chế độ Kết hợp).
   */
  renderRequest(width: number, height: number): { c2w: number[]; K: number[] } {
    this.root.updateMatrixWorld(true);
    this.camera.updateMatrixWorld(true);
    const flip = new THREE.Matrix4().makeScale(1, -1, -1);
    const pose = this.root.matrixWorld.clone().invert().multiply(this.camera.matrixWorld).multiply(flip);
    const e = this.camera.projectionMatrix.elements;
    const K = [(e[0] * width) / 2, 0, ((1 - e[8]) * width) / 2, 0, (e[5] * height) / 2, ((1 + e[9]) * height) / 2, 0, 0, 1];
    return { c2w: pose.clone().transpose().toArray(), K };
  }

  /** Kích thước canvas đang hiển thị (điểm ảnh CSS). */
  viewSize(): { width: number; height: number } {
    return { width: this.host.clientWidth, height: this.host.clientHeight };
  }

  dispose(): void {
    this.clear();
    this.renderer.domElement.removeEventListener("dblclick", this.onDoubleClick);
    this.observer.disconnect();
    this.renderer.setAnimationLoop(null);
    this.controls.dispose();
    // Giải phóng tài nguyên GPU của Spark và trả hẳn context WebGL: trình duyệt chỉ cho khoảng 16
    // context cùng lúc, mỗi lần vào trang Xem 3D lại tạo một cái mới.
    this.colorTarget?.dispose();
    this.spark.dispose();
    this.renderer.dispose();
    this.renderer.forceContextLoss();
    this.renderer.domElement.remove();
  }
}
