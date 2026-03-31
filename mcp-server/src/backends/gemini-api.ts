import type { AudioResult } from "../types.js";

export async function analyzeWithGeminiApi(videoPath: string): Promise<AudioResult> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("GEMINI_API_KEY environment variable is not set. Run video_setup to configure.");
  }

  const { GoogleGenAI, createPartFromUri, createUserContent } = await import("@google/genai");
  const ai = new GoogleGenAI({ apiKey });

  const file = await ai.files.upload({
    file: videoPath,
    config: { mimeType: getMimeType(videoPath) },
  });

  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: createUserContent([
      createPartFromUri(file.uri!, file.mimeType!),
      `Analyze this video in detail. Provide:
1. A complete transcription of all speech with timestamps
2. Description of non-speech audio events (music, sound effects, ambient sounds) with timestamps
3. A detailed visual description of what happens

Format with clear sections: TRANSCRIPTION, AUDIO_EVENTS, VISUAL_DESCRIPTION.`,
    ]),
    config: {
      thinkingConfig: { thinkingBudget: 0 },
    },
  });

  await ai.files.delete({ name: file.name! });

  return {
    backend: "gemini-api",
    transcription: [],
    audio_tags: [],
    full_analysis: response.text || "",
  };
}

function getMimeType(path: string): string {
  const ext = path.split(".").pop()?.toLowerCase();
  const mimeTypes: Record<string, string> = {
    mp4: "video/mp4",
    mov: "video/quicktime",
    avi: "video/x-msvideo",
    mkv: "video/x-matroska",
    webm: "video/webm",
    flv: "video/x-flv",
    wmv: "video/x-ms-wmv",
  };
  return mimeTypes[ext || ""] || "video/mp4";
}
