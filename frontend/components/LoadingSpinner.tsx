type LoadingSpinnerProps = {
  label?: string;
  className?: string;
};

export default function LoadingSpinner({ label = "Loading", className }: LoadingSpinnerProps) {
  return (
    <div className={`inline-flex items-center gap-2 text-sm text-slate-600 ${className ?? ""}`}>
      <span className="spinner" aria-hidden="true" />
      <span>{label}</span>
    </div>
  );
}