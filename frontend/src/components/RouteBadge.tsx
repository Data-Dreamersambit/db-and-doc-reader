/**
 * RouteBadge component displaying styled visual indicator for query intent classification.
 *
 * Trade-off: Static route color mapping avoids runtime CSS generation on M1 browser threads.
 * Lucide icon integration provides immediate visual recognition of agent behavior.
 */
import React from "react";
import { Database, FileText, MessageSquare, Sparkles } from "lucide-react";
import { RouteType } from "../types";

interface RouteBadgeProps {
  route?: RouteType;
  className?: string;
}

export const RouteBadge: React.FC<RouteBadgeProps> = ({ route = "chitchat", className = "" }) => {
  const configs: Record<RouteType, { label: string; icon: React.ReactNode; style: string }> = {
    sql: {
      label: "SQL DATA",
      icon: <Database className="w-3 h-3 text-emerald-400" />,
      style: "bg-emerald-950/80 text-emerald-400 border-emerald-800/80",
    },
    docs: {
      label: "DOCUMENT RAG",
      icon: <FileText className="w-3 h-3 text-sky-400" />,
      style: "bg-sky-950/80 text-sky-400 border-sky-800/80",
    },
    both: {
      label: "SYNTHESIZED",
      icon: <Sparkles className="w-3 h-3 text-purple-400" />,
      style: "bg-purple-950/80 text-purple-400 border-purple-800/80",
    },
    chitchat: {
      label: "CHITCHAT",
      icon: <MessageSquare className="w-3 h-3 text-slate-400" />,
      style: "bg-slate-800 text-slate-300 border-slate-700",
    },
  };

  const current = configs[route] || configs.chitchat;

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[10px] font-mono font-medium border tracking-wider shadow-sm ${current.style} ${className}`}
    >
      {current.icon}
      <span>{current.label}</span>
    </span>
  );
};
