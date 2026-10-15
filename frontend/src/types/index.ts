export type RouteType = "sql" | "docs" | "both" | "chitchat";

export interface ColumnInfo {
  name: string;
  type: string;
}

export interface DatasetSchema {
  table_name: string;
  row_count: number;
  columns: ColumnInfo[];
  sample_rows: Record<string, any>[];
  low_cardinality_values: Record<string, any[]>;
}

export interface DatasetSummary {
  table_name: string;
  row_count: number;
  column_count: number;
}

export interface DocumentInfo {
  id: string;
  filename: string;
  page_count: number;
  chunk_count: number;
  created_at: string;
}

export interface ChartSpec {
  chart_type: "bar" | "line" | "pie" | "table";
  x_field?: string;
  y_fields?: string[];
  title?: string;
}

export interface Lineage {
  tables: string[];
  columns: string[];
}

export interface ChunkMatch {
  chunk_id: string;
  doc_id: string;
  doc_name: string;
  page_number: number;
  text: string;
  cosine_score: number;
  bm25_score: number;
  rrf_score: number;
}

export interface TraceRecord {
  id: number;
  trace_id: string;
  step: string;
  input?: string;
  output?: string;
  latency_ms: number;
  status: "success" | "error";
  error?: string;
  created_at: string;
}

export interface TraceDetailResponse {
  trace_id: string;
  step_count: number;
  total_latency_ms: number;
  steps: TraceRecord[];
}

export interface AgentStep {
  step: string;
  status: "running" | "done";
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  route?: RouteType;
  steps?: AgentStep[];
  sql?: string;
  columns?: string[];
  rows?: Record<string, any>[];
  chart_spec?: ChartSpec;
  lineage?: Lineage;
  citations?: string[];
  chunks?: ChunkMatch[];
  trace_id?: string;
  isStreaming?: boolean;
  error?: string;
}

export interface HealthStatus {
  status: "ok" | "degraded";
  ollama: {
    reachable: boolean;
    url: string;
    chat_model: string;
    embed_model: string;
  };
}
