/**
 * CitationChips component rendering clickable document references that link to inspector sources.
 *
 * Trade-off: In-memory string matching links citations to chunks without relational ID lookups.
 * Interactive chips provide clear provenance trails connecting claims directly to PDF pages.
 */
import React from "react";
import { FileText } from "lucide-react";
import { ChunkMatch } from "../types";

interface CitationChipsProps {
  citations?: string[];
  chunks?: ChunkMatch[];
  selectedCitation?: string | null;
  onSelectCitation?: (citation: string, chunk?: ChunkMatch) => void;
  className?: string;
}

export const CitationChips: React.FC<CitationChipsProps> = ({
  citations = [],
  chunks = [],
  selectedCitation,
  onSelectCitation,
  className = "",
}) => {
  if (!citations.length) return null;

  return (
    <div className={`flex flex-wrap items-center gap-1.5 ${className}`}>
      {citations.map((cite) => {
        // Find matching chunk if available
        const matchedChunk = chunks.find((c) => {
          const expected = `[${c.doc_name} p.${c.page_number}]`;
          return cite.includes(expected) || expected.includes(cite);
        });

        const isSelected = selectedCitation === cite;

        return (
          <button
            key={cite}
            onClick={() => onSelectCitation?.(cite, matchedChunk)}
            className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-mono transition border shadow-sm ${
              isSelected
                ? "bg-amber-900/80 border-amber-500 text-amber-200 ring-1 ring-amber-500/50"
                : "bg-amber-950/40 border-amber-800/80 text-amber-300 hover:bg-amber-900/60 hover:text-amber-100"
            }`}
            title={
              matchedChunk
                ? `Score: ${matchedChunk.cosine_score.toFixed(3)} - Click to view source text in Inspector`
                : "Click to inspect citation"
            }
          >
            <FileText className="w-3 h-3 text-amber-400 shrink-0" />
            <span>{cite}</span>
            {matchedChunk && (
              <span className="text-[10px] text-amber-400/80 pl-0.5">
                • {matchedChunk.cosine_score.toFixed(2)}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
};
