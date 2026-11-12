// Sinh kiểu TypeScript từ JSON Schema của drive2sim (CONTRACTS mục 11.2).
// Chạy `d2s schema export` để lấy schema của mọi loại file, rồi đổi từng file thành
// src/api/generated/<loại>.ts. Cần lệnh `d2s` trên PATH (kích hoạt môi trường Python trước).
import { execFileSync } from "node:child_process";
import { mkdirSync, readdirSync, readFileSync, writeFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { compile } from "json-schema-to-typescript";

const schemaDir = "schemas";
const outDir = join("src", "api", "generated");
rmSync(schemaDir, { recursive: true, force: true });
execFileSync("d2s", ["schema", "export", "--out", schemaDir, "--quiet"], { stdio: "inherit" });
mkdirSync(outDir, { recursive: true });
for (const name of readdirSync(schemaDir).sort()) {
  const schema = JSON.parse(readFileSync(join(schemaDir, name), "utf8"));
  const kind = name.replace(".schema.json", "");
  const code = await compile(schema, schema.title, {
    bannerComment: `// Sinh tự động từ drive2sim/core/schema.py bằng npm run types; không sửa tay.`,
    additionalProperties: false,
  });
  writeFileSync(join(outDir, `${kind}.ts`), code);
}
rmSync(schemaDir, { recursive: true, force: true });
console.log(`đã sinh kiểu cho ${readdirSync(outDir).length} loại file vào ${outDir}`);
