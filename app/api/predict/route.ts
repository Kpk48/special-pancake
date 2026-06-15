import { readFile } from "fs/promises";
import path from "path";
import { NextResponse } from "next/server";

export const runtime = "nodejs";

type WasteModel = {
  k: number;
  labels: string[];
  vectors: number[][];
};

type PpmImage = {
  width: number;
  height: number;
  pixels: Array<[number, number, number]>;
};

let cachedModel: WasteModel | null = null;

async function loadModel(): Promise<WasteModel> {
  if (cachedModel) {
    return cachedModel;
  }

  const modelPath = path.join(process.cwd(), "artifacts", "waste_model.json");
  cachedModel = JSON.parse(await readFile(modelPath, "utf-8")) as WasteModel;
  return cachedModel;
}

function readToken(data: Buffer, startIndex: number): [string, number] {
  let index = startIndex;

  while (index < data.length) {
    const char = data[index];
    if (char === 35) {
      while (index < data.length && data[index] !== 10 && data[index] !== 13) {
        index += 1;
      }
    } else if (char <= 32) {
      index += 1;
    } else {
      break;
    }
  }

  const start = index;
  while (index < data.length && data[index] > 32) {
    index += 1;
  }

  return [data.subarray(start, index).toString("ascii"), index];
}

function parsePpm(data: Buffer): PpmImage {
  let index = 0;
  const [magic, afterMagic] = readToken(data, index);
  index = afterMagic;

  if (magic !== "P3" && magic !== "P6") {
    throw new Error("Upload must be a P3 or P6 PPM image.");
  }

  const [widthText, afterWidth] = readToken(data, index);
  const [heightText, afterHeight] = readToken(data, afterWidth);
  const [maxValueText, afterMax] = readToken(data, afterHeight);
  index = afterMax;

  const width = Number.parseInt(widthText, 10);
  const height = Number.parseInt(heightText, 10);
  const maxValue = Number.parseInt(maxValueText, 10);
  if (!Number.isFinite(width) || !Number.isFinite(height) || maxValue <= 0 || maxValue > 255) {
    throw new Error("Only 8-bit PPM images are supported.");
  }

  const pixelCount = width * height;
  const pixels: Array<[number, number, number]> = [];

  if (magic === "P3") {
    const values: number[] = [];
    while (values.length < pixelCount * 3) {
      const [token, nextIndex] = readToken(data, index);
      index = nextIndex;
      if (!token) {
        break;
      }
      values.push(Number.parseInt(token, 10));
    }
    if (values.length !== pixelCount * 3) {
      throw new Error("PPM image ended before all pixels were read.");
    }
    for (let i = 0; i < values.length; i += 3) {
      pixels.push([values[i], values[i + 1], values[i + 2]]);
    }
  } else {
    while (index < data.length && data[index] <= 32) {
      index += 1;
    }
    const raw = data.subarray(index, index + pixelCount * 3);
    if (raw.length !== pixelCount * 3) {
      throw new Error("PPM image ended before all pixels were read.");
    }
    for (let i = 0; i < raw.length; i += 3) {
      pixels.push([raw[i], raw[i + 1], raw[i + 2]]);
    }
  }

  return { width, height, pixels };
}

function channelStats(values: number[]): [number, number] {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  return [mean / 255, Math.sqrt(variance) / 255];
}

function extractFeatures(image: PpmImage): number[] {
  const red = image.pixels.map((pixel) => pixel[0]);
  const green = image.pixels.map((pixel) => pixel[1]);
  const blue = image.pixels.map((pixel) => pixel[2]);
  const [rMean, rStd] = channelStats(red);
  const [gMean, gStd] = channelStats(green);
  const [bMean, bStd] = channelStats(blue);
  const brightness = image.pixels.map(([r, g, b]) => Math.trunc((r + g + b) / 3));
  const [brightMean, brightStd] = channelStats(brightness);

  let edgeTotal = 0;
  let comparisons = 0;
  for (let y = 0; y < image.height; y += 1) {
    const row = y * image.width;
    for (let x = 0; x < image.width - 1; x += 1) {
      edgeTotal += Math.abs(brightness[row + x] - brightness[row + x + 1]) / 255;
      comparisons += 1;
    }
  }

  const warmRatio = image.pixels.filter(([r, g, b]) => r > g && r > b).length / image.pixels.length;
  const greenRatio = image.pixels.filter(([r, g, b]) => g > r && g > b).length / image.pixels.length;
  const blueRatio = image.pixels.filter(([r, g, b]) => b > r && b > g).length / image.pixels.length;

  return [
    rMean,
    gMean,
    bMean,
    rStd,
    gStd,
    bStd,
    brightMean,
    brightStd,
    comparisons ? edgeTotal / comparisons : 0,
    warmRatio,
    greenRatio,
    blueRatio
  ];
}

function distance(left: number[], right: number[]): number {
  return Math.sqrt(left.reduce((sum, value, index) => sum + (value - right[index]) ** 2, 0));
}

function predict(model: WasteModel, features: number[]) {
  const neighbors = model.vectors
    .map((vector, index) => ({ label: model.labels[index], distance: distance(features, vector) }))
    .sort((left, right) => left.distance - right.distance)
    .slice(0, model.k);

  const counts = new Map<string, number>();
  for (const neighbor of neighbors) {
    counts.set(neighbor.label, (counts.get(neighbor.label) ?? 0) + 1);
  }

  const ranked = [...counts.entries()].sort((left, right) => right[1] - left[1]);
  const probabilities = Object.fromEntries(
    ranked.map(([label, count]) => [label, count / neighbors.length])
  );

  return { label: ranked[0][0], probabilities };
}

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

    if (!file.name.toLowerCase().endsWith(".ppm")) {
      return NextResponse.json(
        {
          error: "This system supports PPM format images. Please upload a .ppm file. You can convert JPEG or PNG files to PPM using image editing tools."
        },
        { status: 400 }
      );
    }

    const model = await loadModel();
    const image = parsePpm(Buffer.from(await file.arrayBuffer()));
    const features = extractFeatures(image);
    return NextResponse.json(predict(model, features));
  } catch (error) {
    let errorMessage = "Unable to classify the image. ";
    if (error instanceof Error) {
      if (error.message.includes("PPM")) {
        errorMessage += "The file must be a valid PPM image.";
      } else if (error.message.includes("8-bit")) {
        errorMessage += "Only 8-bit PPM images are supported.";
      } else {
        errorMessage += error.message;
      }
    } else {
      errorMessage += "Please try again or contact support.";
    }
    return NextResponse.json(
      { error: errorMessage },
      { status: 500 }
    );
  }
}
