/**
 * Sources panel with drag-and-drop CSV/PDF dropzones, upload progress, schema previews, and chunk counts.
 *
 * Trade-off: Client-side drag-and-drop handling avoids heavy dropzone libraries to keep bundle under 600kB.
 * Inline schema preview modal caches sample rows locally to eliminate repeat DuckDB read queries.
 */
import React, { useRef, useState } from "react";
import {
  AlertCircle,
  CheckCircle2,
  Database,
  Eye,
  FileSpreadsheet,
  FileText,
  Loader2,
  RefreshCw,
  Table as TableIcon,
  Trash2,
  UploadCloud,
} from "lucide-react";
import { api } from "../api/client";
import { DatasetSchema, DatasetSummary, DocumentInfo } from "../types";

interface SourcesPanelProps {
  datasets: DatasetSummary[];
  documents: DocumentInfo[];
  isLoading: boolean;
  onRefresh: () => void;
  onSelectDataset?: (schema: DatasetSchema) => void;
}

export const SourcesPanel: React.FC<SourcesPanelProps> = ({
  datasets,
  documents,
  isLoading,
  onRefresh,
}) => {
  const [activeTab, setActiveTab] = useState<"datasets" | "documents">("datasets");
  const [isDraggingCsv, setIsDraggingCsv] = useState<boolean>(false);
  const [isDraggingPdf, setIsDraggingPdf] = useState<boolean>(false);
  const [uploadingStatus, setUploadingStatus] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [selectedSchema, setSelectedSchema] = useState<DatasetSchema | null>(null);
  const [schemaModalTab, setSchemaModalTab] = useState<"columns" | "samples">("columns");

  const csvInputRef = useRef<HTMLInputElement>(null);
  const pdfInputRef = useRef<HTMLInputElement>(null);

  const simulateProgress = (targetStatus: string): ReturnType<typeof setInterval> => {
    setUploadingStatus(targetStatus);
    setUploadProgress(15);
    const interval = setInterval(() => {
      setUploadProgress((prev) => {
        if (prev >= 90) {
          clearInterval(interval);
          return 90;
        }
        return prev + 15;
      });
    }, 150);
    return interval;
  };

  const handleProcessCsv = async (file: File) => {
    if (!file.name.endsWith(".csv")) {
      setUploadError("Only CSV files (.csv) are accepted for datasets.");
      return;
    }
    setUploadError(null);
    const timer = simulateProgress(`Ingesting ${file.name} into DuckDB...`);
    try {
      await api.uploadDataset(file);
      clearInterval(timer);
      setUploadProgress(100);
      setUploadingStatus("Dataset ingested successfully!");
      setTimeout(() => {
        setUploadingStatus(null);
        setUploadProgress(0);
      }, 1500);
      onRefresh();
    } catch (err: any) {
      clearInterval(timer);
      setUploadingStatus(null);
      setUploadError(err.message || "Failed to upload CSV");
    }
  };

  const handleProcessPdf = async (file: File) => {
    if (!file.name.endsWith(".pdf") && !file.name.endsWith(".txt")) {
      setUploadError("Only PDF (.pdf) or text (.txt) files are accepted.");
      return;
    }
    setUploadError(null);
    const timer = simulateProgress(`Embedding and indexing ${file.name}...`);
    try {
      await api.uploadDocument(file);
      clearInterval(timer);
      setUploadProgress(100);
      setUploadingStatus("Document indexed successfully!");
      setTimeout(() => {
        setUploadingStatus(null);
        setUploadProgress(0);
      }, 1500);
      onRefresh();
    } catch (err: any) {
      clearInterval(timer);
      setUploadingStatus(null);
      setUploadError(err.message || "Failed to upload Document");
    }
  };

  const handleInspectSchema = async (tableName: string) => {
    try {
      const schema = await api.getDatasetSchema(tableName);
      setSelectedSchema(schema);
      setSchemaModalTab("columns");
    } catch (err: any) {
      setUploadError(err.message);
    }
  };

  const handleDeleteDocument = async (id: string) => {
    try {
      await api.deleteDocument(id);
      onRefresh();
    } catch (err: any) {
      setUploadError(err.message || "Failed to delete document");
    }
  };

  return (
    <aside className="w-80 bg-slate-900/60 dark:bg-slate-900/60 light:bg-slate-100 border-r border-slate-800 dark:border-slate-800 light:border-slate-200 flex flex-col h-full select-none transition-colors">
      {/* Top Header */}
      <div className="p-3 border-b border-slate-800 dark:border-slate-800 light:border-slate-200 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Database className="w-4 h-4 text-emerald-400" />
          <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-300 dark:text-slate-300 light:text-slate-700">
            Data & Sources
          </h2>
        </div>
        <button
          onClick={onRefresh}
          disabled={isLoading}
          className="p-1 hover:bg-slate-800 dark:hover:bg-slate-800 light:hover:bg-slate-200 rounded text-slate-400 hover:text-slate-200 transition"
          title="Refresh Sources"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? "animate-spin" : ""}`} />
        </button>
      </div>

      {/* Tabs */}
      <div className="p-2.5 border-b border-slate-800 dark:border-slate-800 light:border-slate-200">
        <div className="flex bg-slate-950/80 dark:bg-slate-950/80 light:bg-slate-200/80 p-0.5 rounded-lg text-xs">
          <button
            onClick={() => setActiveTab("datasets")}
            className={`flex-1 py-1.5 rounded-md font-medium transition flex items-center justify-center gap-1.5 ${
              activeTab === "datasets"
                ? "bg-slate-800 dark:bg-slate-800 light:bg-white text-emerald-400 shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <FileSpreadsheet className="w-3.5 h-3.5" />
            <span>Datasets ({datasets.length})</span>
          </button>
          <button
            onClick={() => setActiveTab("documents")}
            className={`flex-1 py-1.5 rounded-md font-medium transition flex items-center justify-center gap-1.5 ${
              activeTab === "documents"
                ? "bg-slate-800 dark:bg-slate-800 light:bg-white text-sky-400 shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <FileText className="w-3.5 h-3.5" />
            <span>Docs ({documents.length})</span>
          </button>
        </div>
      </div>

      {/* Upload Progress Bar */}
      {uploadingStatus && (
        <div className="mx-3 mt-3 p-2.5 bg-slate-950 border border-emerald-900/60 rounded-lg space-y-1.5">
          <div className="flex items-center justify-between text-[11px] font-mono">
            <span className="text-emerald-400 flex items-center gap-1.5">
              {uploadProgress < 100 ? (
                <Loader2 className="w-3 h-3 animate-spin text-emerald-400" />
              ) : (
                <CheckCircle2 className="w-3 h-3 text-emerald-400" />
              )}
              {uploadingStatus}
            </span>
            <span className="text-slate-400">{uploadProgress}%</span>
          </div>
          <div className="w-full bg-slate-800 rounded-full h-1 overflow-hidden">
            <div
              className="bg-emerald-500 h-1 rounded-full transition-all duration-200"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
        </div>
      )}

      {/* Error alert */}
      {uploadError && (
        <div className="mx-3 mt-2 p-2 bg-rose-950/80 border border-rose-800 text-rose-300 text-xs rounded-lg flex items-center gap-1.5">
          <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
          <span>{uploadError}</span>
        </div>
      )}

      {/* Scrollable Content */}
      <div className="flex-1 overflow-y-auto p-3 space-y-4">
        {activeTab === "datasets" && (
          <div className="space-y-3">
            {/* CSV Dropzone */}
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setIsDraggingCsv(true);
              }}
              onDragLeave={() => setIsDraggingCsv(false)}
              onDrop={(e) => {
                e.preventDefault();
                setIsDraggingCsv(false);
                const file = e.dataTransfer.files?.[0];
                if (file) handleProcessCsv(file);
              }}
              onClick={() => csvInputRef.current?.click()}
              className={`border-2 border-dashed rounded-xl p-4 text-center cursor-pointer transition ${
                isDraggingCsv
                  ? "border-emerald-500 bg-emerald-950/30 scale-[0.99]"
                  : "border-slate-800 dark:border-slate-800 light:border-slate-300 hover:border-emerald-500/60 hover:bg-slate-900/40"
              }`}
            >
              <UploadCloud className="w-6 h-6 text-emerald-400 mx-auto mb-1" />
              <div className="text-xs font-semibold text-slate-200">
                Drop CSV here or browse
              </div>
              <div className="text-[10px] text-slate-500 mt-0.5">
                Loads into DuckDB for fast local analytical SQL
              </div>
              <input
                type="file"
                ref={csvInputRef}
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) handleProcessCsv(f);
                }}
                accept=".csv"
                className="hidden"
              />
            </div>

            {/* Datasets list */}
            <div className="space-y-1.5">
              <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block px-1">
                Active Tables ({datasets.length})
              </span>
              {datasets.length === 0 ? (
                <div className="text-xs text-slate-500 py-6 text-center border border-dashed border-slate-800 rounded-lg">
                  No datasets loaded
                </div>
              ) : (
                datasets.map((d) => (
                  <div
                    key={d.table_name}
                    className="flex items-center justify-between p-2 rounded-lg bg-slate-950/60 dark:bg-slate-950/60 light:bg-white border border-slate-800/80 hover:border-slate-700 transition"
                  >
                    <div className="flex items-center gap-2 overflow-hidden">
                      <TableIcon className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                      <span className="text-xs font-medium text-slate-200 font-mono truncate">
                        {d.table_name}
                      </span>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      <span className="text-[10px] text-slate-400 font-mono">
                        {d.row_count} rows
                      </span>
                      <button
                        onClick={() => handleInspectSchema(d.table_name)}
                        className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-[10px] transition"
                        title="Preview Schema & Sample Rows"
                      >
                        <Eye className="w-3 h-3 text-emerald-400" />
                        Schema
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        )}

        {activeTab === "documents" && (
          <div className="space-y-3">
            {/* PDF Dropzone */}
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setIsDraggingPdf(true);
              }}
              onDragLeave={() => setIsDraggingPdf(false)}
              onDrop={(e) => {
                e.preventDefault();
                setIsDraggingPdf(false);
                const file = e.dataTransfer.files?.[0];
                if (file) handleProcessPdf(file);
              }}
              onClick={() => pdfInputRef.current?.click()}
              className={`border-2 border-dashed rounded-xl p-4 text-center cursor-pointer transition ${
                isDraggingPdf
                  ? "border-sky-500 bg-sky-950/30 scale-[0.99]"
                  : "border-slate-800 dark:border-slate-800 light:border-slate-300 hover:border-sky-500/60 hover:bg-slate-900/40"
              }`}
            >
              <UploadCloud className="w-6 h-6 text-sky-400 mx-auto mb-1" />
              <div className="text-xs font-semibold text-slate-200">
                Drop PDF / TXT here
              </div>
              <div className="text-[10px] text-slate-500 mt-0.5">
                Recursive ~500-token chunks with SQLite vector indexing
              </div>
              <input
                type="file"
                ref={pdfInputRef}
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) handleProcessPdf(f);
                }}
                accept=".pdf,.txt"
                className="hidden"
              />
            </div>

            {/* Documents list */}
            <div className="space-y-1.5">
              <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block px-1">
                Indexed Documents ({documents.length})
              </span>
              {documents.length === 0 ? (
                <div className="text-xs text-slate-500 py-6 text-center border border-dashed border-slate-800 rounded-lg">
                  No documents indexed
                </div>
              ) : (
                documents.map((doc) => (
                  <div
                    key={doc.id}
                    className="p-2.5 rounded-lg bg-slate-950/60 dark:bg-slate-950/60 light:bg-white border border-slate-800/80 hover:border-slate-700 transition space-y-1.5"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-1.5 overflow-hidden">
                        <FileText className="w-3.5 h-3.5 text-sky-400 shrink-0" />
                        <span className="text-xs font-medium text-slate-200 truncate">
                          {doc.filename}
                        </span>
                      </div>
                      <button
                        onClick={() => handleDeleteDocument(doc.id)}
                        className="text-slate-500 hover:text-rose-400 transition p-1"
                        title="Delete document"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>

                    <div className="flex items-center justify-between text-[10px] font-mono text-slate-400 pt-1 border-t border-slate-800/60">
                      <span className="px-1.5 py-0.5 rounded bg-sky-950/60 border border-sky-800/70 text-sky-300 font-semibold">
                        {doc.chunk_count} chunks
                      </span>
                      <span>{doc.page_count} page(s)</span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        )}
      </div>

      {/* Rich Schema Preview Modal */}
      {selectedSchema && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-xl w-full max-w-2xl p-5 shadow-2xl flex flex-col max-h-[85vh]">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-3">
              <div className="flex items-center gap-2">
                <TableIcon className="w-4 h-4 text-emerald-400" />
                <h3 className="text-sm font-semibold text-slate-100 font-mono">
                  {selectedSchema.table_name}
                </h3>
                <span className="text-xs font-mono text-slate-400">
                  ({selectedSchema.row_count} rows, {selectedSchema.columns.length} columns)
                </span>
              </div>
              <button
                onClick={() => setSelectedSchema(null)}
                className="text-xs text-slate-400 hover:text-slate-200 px-2 py-1 rounded hover:bg-slate-800"
              >
                ✕ Close
              </button>
            </div>

            {/* Modal Tabs */}
            <div className="flex gap-2 border-b border-slate-800 mb-3 text-xs">
              <button
                onClick={() => setSchemaModalTab("columns")}
                className={`pb-1.5 font-medium border-b-2 transition ${
                  schemaModalTab === "columns"
                    ? "border-emerald-400 text-emerald-400"
                    : "border-transparent text-slate-400 hover:text-slate-200"
                }`}
              >
                Columns & Types ({selectedSchema.columns.length})
              </button>
              <button
                onClick={() => setSchemaModalTab("samples")}
                className={`pb-1.5 font-medium border-b-2 transition ${
                  schemaModalTab === "samples"
                    ? "border-emerald-400 text-emerald-400"
                    : "border-transparent text-slate-400 hover:text-slate-200"
                }`}
              >
                Sample 5 Rows Preview
              </button>
            </div>

            <div className="flex-1 overflow-y-auto space-y-4">
              {schemaModalTab === "columns" ? (
                <>
                  <div className="divide-y divide-slate-800 border border-slate-800 rounded-lg overflow-hidden">
                    {selectedSchema.columns.map((c) => (
                      <div
                        key={c.name}
                        className="flex items-center justify-between px-3 py-2 text-xs bg-slate-950/40"
                      >
                        <span className="font-mono text-slate-200">{c.name}</span>
                        <span className="font-mono text-emerald-400 text-[11px]">
                          {c.type}
                        </span>
                      </div>
                    ))}
                  </div>

                  {Object.keys(selectedSchema.low_cardinality_values || {}).length > 0 && (
                    <div className="space-y-1.5">
                      <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">
                        Distinct Categorical Values (&le; 20 unique)
                      </span>
                      <div className="space-y-1.5">
                        {Object.entries(selectedSchema.low_cardinality_values).map(
                          ([k, v]) => (
                            <div
                              key={k}
                              className="p-2 bg-slate-950/80 rounded border border-slate-800 text-xs"
                            >
                              <span className="font-mono font-medium text-emerald-300">
                                {k}:{" "}
                              </span>
                              <span className="text-slate-400 text-[11px]">
                                {v.join(", ")}
                              </span>
                            </div>
                          )
                        )}
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="overflow-x-auto border border-slate-800 rounded-lg">
                  <table className="w-full text-left text-xs font-mono">
                    <thead className="bg-slate-950 text-slate-400 border-b border-slate-800">
                      <tr>
                        {selectedSchema.columns.map((c) => (
                          <th key={c.name} className="px-3 py-2 font-medium">
                            {c.name}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800">
                      {selectedSchema.sample_rows.map((row, idx) => (
                        <tr key={idx} className="hover:bg-slate-800/40">
                          {selectedSchema.columns.map((c) => (
                            <td key={c.name} className="px-3 py-1.5 text-slate-300">
                              {row[c.name] !== null && row[c.name] !== undefined
                                ? String(row[c.name])
                                : "-"}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </aside>
  );
};
