/**
 * LineageBadge component rendering referenced database tables and columns.
 *
 * Trade-off: Discrete lineage pills avoid heavy graph visualizer dependencies on 8GB client RAM.
 * Explicit separation between tables and attributes provides clear provenance auditability.
 */
import React from "react";
import { Columns, Table } from "lucide-react";
import { Lineage } from "../types";

interface LineageBadgeProps {
  lineage?: Lineage;
  className?: string;
}

export const LineageBadge: React.FC<LineageBadgeProps> = ({ lineage, className = "" }) => {
  if (!lineage || (!lineage.tables.length && !lineage.columns.length)) {
    return null;
  }

  return (
    <div className={`flex flex-col gap-1.5 text-xs ${className}`}>
      {lineage.tables.length > 0 && (
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1">
            <Table className="w-3 h-3 text-emerald-400" />
            Tables:
          </span>
          {lineage.tables.map((tbl) => (
            <span
              key={tbl}
              className="px-2 py-0.5 rounded bg-emerald-950/70 border border-emerald-800/80 text-emerald-300 font-mono text-[11px] font-medium"
            >
              {tbl}
            </span>
          ))}
        </div>
      )}

      {lineage.columns.length > 0 && (
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1">
            <Columns className="w-3 h-3 text-sky-400" />
            Columns:
          </span>
          {lineage.columns.map((col) => (
            <span
              key={col}
              className="px-2 py-0.5 rounded bg-slate-800/80 border border-slate-700/80 text-slate-300 font-mono text-[11px]"
            >
              {col}
            </span>
          ))}
        </div>
      )}
    </div>
  );
};
