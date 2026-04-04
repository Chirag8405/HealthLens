type PlotCardProps = {
  title: string;
  description: string;
  image_b64?: string | null;
  downloadable?: boolean;
  downloadName?: string;
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

export default function PlotCard({
  title,
  description,
  image_b64,
  downloadable = false,
  downloadName = "plot.png",
}: PlotCardProps) {
  const src = toImageSrc(image_b64);

  return (
    <article className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-base font-semibold text-slate-900">{title}</h3>
          <p className="mt-1 text-sm text-slate-600">{description}</p>
        </div>

        {downloadable && src ? (
          <a
            href={src}
            download={downloadName}
            className="rounded-lg border border-slate-300 px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.1em] text-slate-700 transition hover:bg-slate-50"
          >
            Download
          </a>
        ) : null}
      </div>

      {src ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={src} alt={title} className="mt-4 w-full rounded-xl border border-slate-100" />
      ) : (
        <div className="mt-4 rounded-xl border border-dashed border-slate-300 bg-slate-50 p-8 text-center text-sm text-slate-500">
          Plot not available.
        </div>
      )}
    </article>
  );
}
