/**
 * SqlViewer component with monospace formatting and one-click clipboard copying.
 *
 * Trade-off: CSS-based whitespace preservation avoids heavy syntax highlighter bundles.
 * Clipboard API with timeout feedback gives immediate interaction assurance.
 */
import React, { useState } from "react";
import { Check, Code2, Copy } from "lucide-react";

interface SqlViewerProps {
  sql?: string;
  title?: string;
  className?: string;
}

export const SqlViewer: React.FC<SqlViewerProps> = ({
  sql,
  title = "Generated DuckDB SQL",
  className = "",
}) => {
  const [copied, setCopied] = useState<boolean>(false);

  if (!sql) {
    return (
      <div className="p-4 text-xs text-slate-500 text-center border border-dashed border-slate-800 rounded-lg">
        No SQL query generated
      </div>
    );
  }

  const handleCopy = () => {
    navigator.clipboard.writeText(sql);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className={`space-y-2 ${className}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-400 uppercase tracking-wider">
          <Code2 className="w-3.5 h-3.5 text-emerald-400" />
          <span>{title}</span>
        </div>
        <button
          onClick={handleCopy}
          className="inline-flex items-center gap-1 px-2 py-1 rounded bg-slate-800/80 hover:bg-slate-800 text-slate-300 hover:text-slate-100 text-xs font-medium transition border border-slate-700/80"
          title="Copy SQL to clipboard"
        >
          {copied ? (
            <>
              <Check className="w-3.5 h-3.5 text-emerald-400" />
              <span className="text-emerald-400">Copied</span>
            </>
          ) : (
            <>
              <Copy className="w-3.5 h-3.5" />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>

      <pre className="p-3 bg-slate-950 rounded-lg border border-slate-800 text-emerald-400 font-mono text-xs overflow-x-auto whitespace-pre-wrap leading-relaxed shadow-inner">
        {sql}
      </pre>
    </div>
  );
};
