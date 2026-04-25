import { z } from "zod";
import { resolve } from "path";
import { existsSync, statSync } from "fs";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { getVideoMetadata } from "../extractors/frames.js";

function validateVideoPath(inputPath: string): string {
  const resolved = resolve(inputPath);
  if (!existsSync(resolved)) {
    throw new Error(`File not found: ${resolved}`);
  }
  const stat = statSync(resolved);
  if (!stat.isFile()) {
    throw new Error(`Path is not a regular file: ${resolved}`);
  }
  return resolved;
}

export function registerVideoInfo(server: McpServer): void {
  server.tool(
    "video_info",
    "Get metadata about a video file without processing it (duration, resolution, codec, etc.)",
    { path: z.string().describe("Absolute or relative path to the video file") },
    async ({ path }) => {
      const safePath = validateVideoPath(path);
      const metadata = await getVideoMetadata(safePath);
      return {
        content: [{ type: "text", text: JSON.stringify(metadata, null, 2) }],
      };
    },
  );
}
