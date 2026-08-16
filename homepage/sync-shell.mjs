import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { injectSeoShell, renderRobots, renderSitemap } from "./seoShell.js";

const projectDir = import.meta.dirname;
const source = await readFile(resolve(projectDir, "index.html"), "utf8");
const assetsDir = resolve(projectDir, "../web/home-assets");
const [cssAsset, jsAsset] = await Promise.all([
  readFile(resolve(assetsDir, "homepage.css")),
  readFile(resolve(assetsDir, "homepage.js")),
]);
const cssVersion = createHash("sha256").update(cssAsset).digest("hex").slice(0, 12);
const jsVersion = createHash("sha256").update(jsAsset).digest("hex").slice(0, 12);
const shell = injectSeoShell(source)
  .replace(
    "  </head>",
    `    <link rel="stylesheet" href="/static/home-assets/homepage.css?v=${cssVersion}" />\n  </head>`,
  )
  .replace(
  '    <script type="module" src="/src/main.jsx"></script>',
  `    <script type="module" src="/static/home-assets/homepage.js?v=${jsVersion}"></script>`,
  );

await Promise.all([
  writeFile(resolve(projectDir, "../web/home.html"), shell, "utf8"),
  writeFile(resolve(projectDir, "../web/robots.txt"), renderRobots(), "utf8"),
  writeFile(resolve(projectDir, "../web/sitemap.xml"), renderSitemap(), "utf8"),
]);
