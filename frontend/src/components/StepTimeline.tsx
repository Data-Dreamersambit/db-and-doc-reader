/**
 * StepTimeline component rendering active and completed agent reasoning steps with latency.
 *
 * Trade-off: CSS-only pulse and spin animations ensure non-blocking 60fps UI on Apple Silicon.
 * Step-by-step progress keeps user informed during multi-second LLM SQL and RAG queries.
 */
import React from "react";
import { CheckCircle2, Clock, Loader2 } from "lucide-react";
import { AgentStep } from "../types";

interface StepTimelineProps {
  steps?: AgentStep[];
  latencyMs?: number;
  className?: string;
}

export const StepTimeline: React.FC<StepTimelineProps> = ({
  steps = [],
  latencyMs,
  className = "",
}) => {
  if (!steps.length) return null;

  return (
    <div className={`space-y-1.5 py-1 ${className}`}>
      {steps.map((st, i) => {
        const isRunning = st.status === "running";
        const isDone = st.status === "done";

        return (
          <div
            key={i}
            className="flex items-center justify-between text-xs py-1 px-2.5 rounded-lg bg-slate-900/60 border border-slate-800/80"
          >
            <div className="flex items-center gap-2">
              {isRunning ? (
                <Loader2 className="w-3.5 h-3.5 text-amber-400 animate-spin shrink-0" />
              ) : isDone ? (
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
              ) : (
                <div className="w-3.5 h-3.5 rounded-full bg-slate-700 shrink-0" />
              )}
              <span
                className={`font-medium ${
                  isRunning ? "text-amber-300" : isDone ? "text-slate-300" : "text-slate-500"
                }`}
              >
                {st.step}
              </span>
            </div>

            <div className="flex items-center gap-2 font-mono text-[10px]">
              {isRunning && (
                <span className="text-amber-400 animate-pulse">in progress</span>
              )}
              {isDone && (
                <span className="text-emerald-400/90 font-medium">completed</span>
              )}
            </div>
          </div>
        );
      })}

      {latencyMs !== undefined && latencyMs > 0 && (
        <div className="flex items-center justify-end gap-1 text-[11px] font-mono text-slate-500 pt-1 pr-1">
          <Clock className="w-3 h-3 text-slate-500" />
          <span>{latencyMs.toFixed(1)} ms</span>
        </div>
      )}
    </div>
  );
};
