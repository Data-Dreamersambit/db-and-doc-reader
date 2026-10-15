/**
 * InspectorPanel coordinating SqlViewer, ChartRenderer, DataTable, CitationChips, and StepTimeline.
 */
import React, { useState } from "react";
import {
  Activity,
  BarChart2,
  Code2,
  FileCheck2,
} from "lucide-react";
import { ChatMessage, ChunkMatch } from "../types";
import { ChartRenderer } from "./ChartRenderer";
import { CitationChips } from "./CitationChips";
import { DataTable } from "./DataTable";
import { LineageBadge } from "./LineageBadge";
import { SqlViewer } from "./SqlViewer";
import { StepTimeline } from "./StepTimeline";

interface InspectorPanelProps {
  message?: ChatMessage | null;
  selectedCitation?: string | null;
  onSelectCitation?: (citation: string, chunk?: ChunkMatch) => void;
}

export const InspectorPanel: React.FC<InspectorPanelProps> = ({
  message,
  selectedCitation: externalCitation,
  onSelectCitation: externalSelectCitation,
}) => {
  const [activeTab, setActiveTab] = useState<"sql" | "data" | "citations" | "trace">("sql");
  const [internalCitation, setInternalCitation] = useState<string | null>(null);

  const activeCitation = externalCitation !== undefined ? externalCitation : internalCitation;

  const handleCitationClick = (cite: string, chunk?: ChunkMatch) => {
    setInternalCitation(cite);
    setActiveTab("citations");
    externalSelectCitation?.(cite, chunk);
  };

  const hasSql = Boolean(message?.sql);
  const hasData = Boolean(message?.rows && message.rows.length > 0);
  const hasCitations = Boolean(message?.citations && message.citations.length > 0) || Boolean(message?.chunks && message.chunks.length > 0);
  const hasTrace = Boolean(message?.steps && message.steps.length > 0);

  return (
    <aside className="w-96 bg-slate-900/60 border-l border-slate-800 flex flex-col h-full select-none">
      {/* Header Tabs */}
      <div className="p-3 border-b border-slate-800">
        <div className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">
          Inspector
        </div>
        <div className="flex bg-slate-950/80 p-0.5 rounded-lg border border-slate-800 text-xs">
          <button
            onClick={() => setActiveTab("sql")}
            className={`flex-1 py-1.5 rounded-md font-medium transition flex items-center justify-center gap-1 ${
              activeTab === "sql"
                ? "bg-slate-800 text-emerald-400 shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Code2 className="w-3.5 h-3.5" />
            SQL
          </button>
          <button
            onClick={() => setActiveTab("data")}
            className={`flex-1 py-1.5 rounded-md font-medium transition flex items-center justify-center gap-1 ${
              activeTab === "data"
                ? "bg-slate-800 text-sky-400 shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <BarChart2 className="w-3.5 h-3.5" />
            Data
          </button>
          <button
            onClick={() => setActiveTab("citations")}
            className={`flex-1 py-1.5 rounded-md font-medium transition flex items-center justify-center gap-1 ${
              activeTab === "citations"
                ? "bg-slate-800 text-amber-400 shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <FileCheck2 className="w-3.5 h-3.5" />
            Citations
          </button>
          <button
            onClick={() => setActiveTab("trace")}
            className={`flex-1 py-1.5 rounded-md font-medium transition flex items-center justify-center gap-1 ${
              activeTab === "trace"
                ? "bg-slate-800 text-violet-400 shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Activity className="w-3.5 h-3.5" />
            Trace
          </button>
        </div>
      </div>

      {/* Tab Panels */}
      <div className="flex-1 overflow-y-auto p-4">
        {!message ? (
          <div className="text-xs text-slate-500 text-center py-16">
            Click an assistant message to inspect generated SQL, charts, citations, and execution traces.
          </div>
        ) : (
          <>
            {/* SQL Tab */}
            {activeTab === "sql" && (
              <div className="space-y-4">
                {hasSql ? (
                  <>
                    <SqlViewer sql={message.sql} />
                    <LineageBadge lineage={message.lineage} className="pt-2 border-t border-slate-800" />
                  </>
                ) : (
                  <div className="text-xs text-slate-500 text-center py-10">
                    No SQL query executed for this turn ({message.route || "chitchat"})
                  </div>
                )}
              </div>
            )}

            {/* Data Tab */}
            {activeTab === "data" && (
              <div className="space-y-5">
                {hasData ? (
                  <>
                    <div className="space-y-1">
                      <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">
                        Visual Chart
                      </span>
                      <ChartRenderer
                        spec={message.chart_spec}
                        rows={message.rows}
                        columns={message.columns}
                      />
                    </div>

                    <div className="space-y-2 pt-3 border-t border-slate-800">
                      <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">
                        Table Records ({message.rows?.length})
                      </span>
                      <DataTable
                        columns={message.columns}
                        rows={message.rows || []}
                        defaultPageSize={10}
                      />
                    </div>
                  </>
                ) : (
                  <div className="text-xs text-slate-500 text-center py-10">
                    No structured records available for this turn
                  </div>
                )}
              </div>
            )}

            {/* Citations Tab */}
            {activeTab === "citations" && (
              <div className="space-y-4">
                {hasCitations ? (
                  <>
                    <div>
                      <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-2">
                        Referenced Sources ({message.citations?.length || 0})
                      </span>
                      <CitationChips
                        citations={message.citations}
                        chunks={message.chunks}
                        selectedCitation={activeCitation}
                        onSelectCitation={handleCitationClick}
                      />
                    </div>

                    {message.chunks && message.chunks.length > 0 && (
                      <div className="space-y-2.5 pt-3 border-t border-slate-800">
                        <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">
                          Retrieved Context Chunks ({message.chunks.length})
                        </span>
                        {message.chunks.map((chk) => {
                          const isHighlighted = activeCitation
                            ? activeCitation.includes(chk.doc_name) &&
                              activeCitation.includes(String(chk.page_number))
                            : false;

                          return (
                            <div
                              key={chk.chunk_id}
                              className={`p-3 bg-slate-950 rounded-lg border text-xs space-y-1.5 transition ${
                                isHighlighted
                                  ? "border-amber-500 ring-1 ring-amber-500/50 shadow-md"
                                  : "border-slate-800 hover:border-slate-700"
                              }`}
                            >
                              <div className="flex items-center justify-between text-slate-400 font-mono text-[11px]">
                                <span className="text-sky-400 font-semibold">
                                  [{chk.doc_name} p.{chk.page_number}]
                                </span>
                                <span className="text-[10px] text-emerald-400">
                                  Sim: {chk.cosine_score.toFixed(3)}
                                </span>
                              </div>
                              <p className="text-slate-300 line-clamp-5 leading-relaxed font-sans text-xs">
                                {chk.text}
                              </p>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </>
                ) : (
                  <div className="text-xs text-slate-500 text-center py-10">
                    No document citations for this response
                  </div>
                )}
              </div>
            )}

            {/* Trace Tab */}
            {activeTab === "trace" && (
              <div className="space-y-3">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">
                  Agent Steps Timeline
                </span>
                {hasTrace ? (
                  <StepTimeline steps={message.steps} />
                ) : (
                  <div className="text-xs text-slate-500 text-center py-10">
                    No step timeline recorded for this response
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </aside>
  );
};
