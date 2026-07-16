import fs from "node:fs";
import { fileURLToPath } from "node:url";

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const settingsPath = fileURLToPath(new URL("../core/settings.py", import.meta.url));
const settingsSource = fs.readFileSync(settingsPath, "utf8");
const projectVersionMatch = settingsSource.match(
  /^PROJECT_VERSION\s*=\s*["']([^"']+)["']/m,
);

if (!projectVersionMatch) {
  throw new Error(`Unable to read PROJECT_VERSION from ${settingsPath}`);
}

const projectVersion = projectVersionMatch[1];

function buildMetadata() {
  return {
    name: "paper-reader-build-metadata",
    apply: "build",
    generateBundle() {
      this.emitFile({
        type: "asset",
        fileName: "build-meta.json",
        source: `${JSON.stringify(
          {
            schema_version: 1,
            project_version: projectVersion,
          },
          null,
          2,
        )}\n`,
      });
    },
  };
}

export default defineConfig({
  optimizeDeps: {
    include: ["react", "react-dom/client"],
  },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
    warmup: {
      clientFiles: ["./src/main.jsx"],
    },
  },
  plugins: [react(), buildMetadata()],
});
