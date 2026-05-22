interface AnnotationToastProps {
  toast: { msg: string; ok: boolean };
}

export function AnnotationToast({ toast }: AnnotationToastProps) {
  return (
    <div
      className={`fixed bottom-6 right-6 z-50 rounded-xl px-5 py-3 text-sm font-medium shadow-lg transition-all ${toast.ok ? "bg-emerald-700 text-white" : "bg-red-700 text-white"}`}
    >
      {toast.msg}
    </div>
  );
}
