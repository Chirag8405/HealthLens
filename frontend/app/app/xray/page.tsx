"use client";

import { DragEvent, FormEvent, useState } from "react";
import { useMutation } from "@tanstack/react-query";

import ErrorBanner from "@/components/ErrorBanner";
import LoadingSpinner from "@/components/LoadingSpinner";
import XrayViewer from "@/components/XrayViewer";
import { uploadFile } from "@/lib/api";
import type { CnnPredictResponse } from "@/lib/types";

function isSupportedImage(file: File): boolean {
  const type = file.type.toLowerCase();
  return type === "image/jpeg" || type === "image/png" || file.name.toLowerCase().endsWith(".jpg") || file.name.toLowerCase().endsWith(".jpeg") || file.name.toLowerCase().endsWith(".png");
}

function readAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.onerror = () => reject(new Error("Could not read image."));
    reader.readAsDataURL(file);
  });
}

export default function ClinicalXrayPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>("");
  const [fileError, setFileError] = useState<string>("");

  const predictMutation = useMutation({
    mutationFn: (file: File) => uploadFile<CnnPredictResponse>("/dl/cnn/predict", file),
  });

  const handleFile = async (file: File) => {
    if (!isSupportedImage(file)) {
      setFileError("Please upload a .jpg, .jpeg, or .png image.");
      return;
    }

    setFileError("");
    setSelectedFile(file);
    setPreviewUrl(await readAsDataUrl(file));
  };

  const onDrop = async (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    const dropped = event.dataTransfer.files?.[0];
    if (!dropped) {
      return;
    }
    await handleFile(dropped);
  };

  const onAnalyze = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!selectedFile) {
      setFileError("Please select an X-ray image first.");
      return;
    }
    predictMutation.mutate(selectedFile);
  };

  const label = predictMutation.data?.label;

  return (
    <section className="space-y-7">
      <header className="space-y-2">
        <h2 className="text-3xl font-bold text-emerald-950">X-Ray Analysis</h2>
        <p className="text-sm text-emerald-900/80">Chest X-ray pneumonia screening.</p>
      </header>

      <form onSubmit={onAnalyze} className="space-y-4 rounded-2xl border border-emerald-200 bg-white p-6 shadow-sm">
        <label
          onDragOver={(event) => event.preventDefault()}
          onDrop={onDrop}
          className="block cursor-pointer rounded-2xl border-2 border-dashed border-emerald-300 bg-emerald-50 px-6 py-14 text-center"
        >
          <input
            type="file"
            accept=".jpg,.jpeg,.png,image/jpeg,image/png"
            className="hidden"
            onChange={async (event) => {
              const file = event.target.files?.[0];
              if (file) {
                await handleFile(file);
              }
            }}
          />
          <p className="text-sm font-semibold text-emerald-900">Drag and drop an X-ray image here</p>
          <p className="mt-1 text-xs text-emerald-800/80">or click to upload (.jpg, .jpeg, .png)</p>
        </label>

        {previewUrl ? (
          <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
            <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">Preview</p>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={previewUrl} alt="Uploaded X-ray preview" className="mt-2 max-h-80 rounded-lg border border-slate-200" />
          </div>
        ) : null}

        <button
          type="submit"
          disabled={predictMutation.isPending}
          className="rounded-xl bg-emerald-700 px-5 py-3 text-sm font-semibold text-white transition hover:bg-emerald-800 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {predictMutation.isPending ? "Analyzing..." : "Analyze X-Ray"}
        </button>
      </form>

      {fileError ? <ErrorBanner message={fileError} /> : null}
      {predictMutation.error ? (
        <ErrorBanner message={predictMutation.error instanceof Error ? predictMutation.error.message : "Analysis failed."} />
      ) : null}
      {predictMutation.isPending ? <LoadingSpinner label="Running chest X-ray screening" /> : null}

      {label ? (
        <section className="space-y-4 rounded-2xl border border-emerald-200 bg-white p-6 shadow-sm">
          <XrayViewer
            original_b64={previewUrl}
            gradcam_b64={predictMutation.data?.gradcam_b64}
            label={label}
            confidence={predictMutation.data?.confidence}
            showConfidence={false}
          />

          <p className="rounded-xl bg-slate-50 px-4 py-3 text-sm text-slate-700">
            The highlighted regions show areas the AI focused on during analysis. Red and warm areas indicate regions of concern. This is a screening tool and should always be confirmed with clinical assessment.
          </p>

          <p className="rounded-xl border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900">
            HealthLens is a decision support tool. Results should be reviewed by a qualified medical professional before any clinical decisions are made.
          </p>
        </section>
      ) : null}
    </section>
  );
}
