import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "e2e",
  timeout: 60000,
  use: { launchOptions: { args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] } },
});
