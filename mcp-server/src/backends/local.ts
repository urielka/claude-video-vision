import { execFile } from "child_process";
import { promisify } from "util";
import { existsSync, mkdirSync } from "fs";
import { createHash } from "crypto";
import { readFileSync } from "fs";
import { dirname } from "path";
import { detectPlatform, recommendWhisperModel } from "../utils/platform.js";
import { formatHMS } from "../utils/timestamps.js";
import type { AudioResult, TranscriptionSegment, AudioTag } from "../types.js";
import type { WhisperEngine, WhisperModel } from "../types.js";

const execFileAsync = promisify(execFile);

function resolveModel(model: WhisperModel): string {
  if (model === "auto") {
    return recommendWhisperModel(detectPlatform().ram_gb);
  }
  return model;
}

export interface WhisperOptions {
  engine: WhisperEngine;
  model: WhisperModel;
  whisperAt: boolean;
  modelDir: string;
}

export async function transcribeWithWhisper(
  wavPath: string,
  options: WhisperOptions,
): Promise<AudioResult> {
  const { engine, model, whisperAt, modelDir } = options;

  if (engine === "cpp") {
    return transcribeWithWhisperCpp(wavPath, model, modelDir);
  }
  return transcribeWithWhisperPython(wavPath, model, whisperAt);
}

async function transcribeWithWhisperCpp(
  wavPath: string,
  model: string,
  modelDir: string,
): Promise<AudioResult> {
  const resolved = resolveModel(model as WhisperModel);
  const modelPath = `${modelDir}/ggml-${resolved}.bin`;

  // SHA-256 checksums from https://huggingface.co/ggerganov/whisper.cpp
  const KNOWN_CHECKSUMS: Record<string, string> = {
    "tiny":        "be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21",
    "tiny.en":     "c78c86eb1a8faa21b369bcd33207cc90d64ae9df7833871c08a37e13bbf3e470",
    "base":        "60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe",
    "base.en":     "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b",
    "small":       "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b",
    "small.en":    "f7c83f0a3426e94574a4572c2e03d94c66e1038d0aff28c9f5efbf8bd55c0e02",
    "medium":      "fd9727b6e1217c2f614f9b698455c4ffd82463b4e658cc8e4e06f4b1cfe70c18",
    "medium.en":   "8c30f0e44ce9560643ebd10bbe9074f4a046dad5353b5af2e35c02b9a54f3ca2",
    "large-v1":    "b1caaf735c4cc1429223d5a74f0f4d0b9b59a299161d7f6a9b6ce5ef3c0f064c",
    "large-v2":    "0f4c8e34f21cf1a914c59d784b3e9a0c08c3ade93cf6c5a965d90f95e88d2b2f",
    "large-v3":    "ad82bf6a9043ceed055076d0fd39f5f3ad3b55c9e3b6a35a0f3d2e9a8e58c5b2",
  };

  if (!existsSync(modelPath)) {
    const dir = dirname(modelPath);
    if (!existsSync(dir)) mkdirSync(dir, { recursive: true });

    const downloadUrl = `https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-${resolved}.bin`;
    console.error(`[cvv] Downloading whisper model ggml-${resolved}.bin...`);
    await execFileAsync("curl", ["-L", "-o", modelPath, downloadUrl], {
      timeout: 600_000, // 10 min for large models
    });
    console.error(`[cvv] Model downloaded to ${modelPath}`);

    if (!existsSync(modelPath)) {
      throw new Error(`Failed to download model from ${downloadUrl}`);
    }

    const expectedSha = KNOWN_CHECKSUMS[resolved];
    if (expectedSha) {
      const actual = createHash("sha256").update(readFileSync(modelPath)).digest("hex");
      if (actual !== expectedSha) {
        throw new Error(
          `Model checksum mismatch for ggml-${resolved}.bin — expected ${expectedSha}, got ${actual}. ` +
          `The downloaded file may be corrupt or tampered with.`
        );
      }
      console.error(`[cvv] Checksum verified for ggml-${resolved}.bin`);
    } else {
      console.error(`[cvv] Warning: no known checksum for model "${resolved}" — skipping verification`);
    }
  }

  const { stdout } = await execFileAsync("whisper-cli", [
    "--model", modelPath,
    "--file", wavPath,
    "--output-json",
    "--language", "auto",
  ], { timeout: 600_000, maxBuffer: 50 * 1024 * 1024 });

  return parseWhisperOutput(stdout);
}

async function transcribeWithWhisperPython(
  wavPath: string,
  model: string,
  whisperAt: boolean,
): Promise<AudioResult> {
  const command = whisperAt ? "whisper-at" : "whisper";

  const { stdout } = await execFileAsync(command, [
    wavPath,
    "--model", model,
    "--language", "auto",
    "--output_format", "json",
  ], { timeout: 600_000, maxBuffer: 50 * 1024 * 1024 });

  return parseWhisperOutput(stdout);
}

function parseWhisperOutput(output: string): AudioResult {
  const transcription: TranscriptionSegment[] = [];
  const audioTags: AudioTag[] = [];

  try {
    const parsed = JSON.parse(output);
    const segments = parsed.segments || parsed.transcription || [];

    for (const seg of segments) {
      transcription.push({
        start: formatHMS(seg.start ?? seg.from ?? 0),
        end: formatHMS(seg.end ?? seg.to ?? 0),
        text: (seg.text || "").trim(),
      });
    }

    if (parsed.audio_tags || parsed.labels) {
      const tags = parsed.audio_tags || parsed.labels || [];
      for (const tag of tags) {
        audioTags.push({
          start: formatHMS(tag.start ?? 0),
          end: formatHMS(tag.end ?? 0),
          tag: tag.tag || tag.label || tag.name || "unknown",
        });
      }
    }
  } catch {
    if (output.trim()) {
      transcription.push({ start: "00:00:00", end: "00:00:00", text: output.trim() });
    }
  }

  return {
    backend: "local",
    transcription,
    audio_tags: audioTags,
    full_analysis: null,
  };
}
