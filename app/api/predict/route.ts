import { NextResponse } from "next/server";
import { execFile } from "child_process";
import { promisify } from "util";
import path from "path";
import os from "os";
import { writeFile, unlink } from "fs/promises";

export const runtime = "nodejs";

const execFileAsync = promisify(execFile);
const PYTHON_SERVER_URL = "http://127.0.0.1:5000/predict";

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const file = formData.get("image");

    if (!(file instanceof File)) {
      return NextResponse.json(
        { error: "Please select a valid image file." },
        { status: 400 }
      );
    }

    const fileBuffer = Buffer.from(await file.arrayBuffer());

    try {
      const serverResponse = await fetch(PYTHON_SERVER_URL, {
        method: "POST",
        headers: { "Content-Type": file.type || "application/octet-stream" },
        body: fileBuffer,
      });

      if (serverResponse.ok) {
        const prediction = await serverResponse.json();
        return NextResponse.json(prediction);
      }
    } catch {
      console.warn("Persistent Python inference server is offline. Falling back to subprocess CLI execution (higher latency). Start server with: python scripts/persistent_server.py 5000");
    }

    const tempDir = os.tmpdir();
    const uniqueName = `upload_${Date.now()}_${Math.random().toString(36).slice(2)}.tmp`;
    const tempFilePath = path.join(tempDir, uniqueName);
    
    await writeFile(tempFilePath, fileBuffer);
    const scriptPath = path.join(process.cwd(), "scripts", "predict_single_image.py");
    
    try {
      const { stdout } = await execFileAsync("python", [scriptPath, tempFilePath], {
        cwd: process.cwd(),
        env: { ...process.env, PYTHONPATH: path.join(process.cwd(), "src") },
      });
      const prediction = JSON.parse(stdout.trim());
      return NextResponse.json(prediction);
    } finally {
      await unlink(tempFilePath).catch(() => {});
    }
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : "Classification failed.";
    return NextResponse.json(
      { error: errorMessage },
      { status: 500 }
    );
  }
}
