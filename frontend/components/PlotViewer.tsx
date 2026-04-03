type PlotViewerProps = {
  title: string;
  imageBase64?: string | null;
  downloadFileName?: string;
  description?: string;
  className?: string;
  delayMs?: number;
};

export default function PlotViewer({
  title,
  imageBase64,
  downloadFileName,
  description,
  className,
  delayMs = 0,
}: PlotViewerProps) {
  const imageSrc = imageBase64 ? `data:image/png;base64,${imageBase64}` : null;

  return (
    <article
      className={`animate-card-in rounded-2xl border border-slate-200 bg-white p-4 shadow-sm ${className ?? ""}`}
      style={{ animationDelay: `${delayMs}ms` }}
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-base font-semibold text-slate-900">{title}</h3>
          {description ? <p className="mt-1 text-sm text-slate-600">{description}</p> : null}
        </div>

        {imageSrc && downloadFileName ? (
          <a
            className="inline-flex items-center rounded-lg border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 transition hover:border-slate-400 hover:bg-slate-50"
            href={imageSrc}
            download={downloadFileName}
          >
            Download
          </a>
        ) : null}
      </div>

      {imageSrc ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={imageSrc}
          alt={title}
          className="mt-4 w-full rounded-xl border border-slate-100 object-cover"
        />
      ) : (
        <div className="mt-4 rounded-xl border border-dashed border-slate-300 bg-slate-50 p-8 text-center text-sm text-slate-500">
          Plot not available for this dataset.
        </div>
      )}
    </article>
  );
}