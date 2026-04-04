type XrayViewerProps = {
  original_b64?: string | null;
  gradcam_b64?: string | null;
  label?: string;
  confidence?: number;
  showConfidence?: boolean;
};

function toImageSrc(value: string | null | undefined): string | null {
  if (!value) {
    return null;
  }
  if (value.startsWith("data:image")) {
    return value;
  }
  return `data:image/png;base64,${value}`;
}

function isPositiveLabel(label: string | undefined): boolean {
  if (!label) {
    return false;
  }
  return label.toLowerCase().includes("pneumonia");
}

export default function XrayViewer({
  original_b64,
  gradcam_b64,
  label,
  confidence,
  showConfidence = false,
}: XrayViewerProps) {
  const originalSrc = toImageSrc(original_b64);
  const gradcamSrc = toImageSrc(gradcam_b64);

  const positive = isPositiveLabel(label);
  const badgeClasses = positive
    ? "border-red-300 bg-red-50 text-red-900"
    : "border-emerald-300 bg-emerald-50 text-emerald-900";

  return (
    <div className="grid gap-5 lg:grid-cols-2">
      <div className="rounded-2xl border border-slate-200 bg-white p-4">
        <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Original X-ray</p>
        {originalSrc ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={originalSrc} alt="Original chest X-ray" className="mt-3 w-full rounded-xl border border-slate-100" />
        ) : (
          <div className="mt-3 rounded-xl border border-dashed border-slate-300 bg-slate-50 p-8 text-center text-sm text-slate-500">
            Upload an image to preview.
          </div>
        )}
      </div>

      <div className="rounded-2xl border border-slate-200 bg-white p-4">
        <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Grad-CAM Overlay</p>
        {gradcamSrc ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={gradcamSrc} alt="Grad-CAM overlay" className="mt-3 w-full rounded-xl border border-slate-100" />
        ) : (
          <div className="mt-3 rounded-xl border border-dashed border-slate-300 bg-slate-50 p-8 text-center text-sm text-slate-500">
            Heatmap preview is unavailable.
          </div>
        )}
      </div>

      {label ? (
        <div className="lg:col-span-2">
          <div className={`rounded-xl border px-4 py-3 ${badgeClasses}`}>
            <p className="text-lg font-bold">{positive ? "Pneumonia Indicators Found" : "No Pneumonia Detected"}</p>
            {showConfidence && typeof confidence === "number" ? (
              <p className="mt-1 text-sm">Confidence: {(confidence * 100).toFixed(1)}%</p>
            ) : null}
          </div>
        </div>
      ) : null}
    </div>
  );
}
