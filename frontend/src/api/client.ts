/**
 * Typed REST API client for AgenticInsight backend.
 *
 * Trade-off: Native fetch API avoids heavy axios bundles for lightweight M1 client execution.
 * Centralized client enforces typed REST contracts across all API boundaries.
 */
import {
  DatasetSchema,
  DatasetSummary,
  DocumentInfo,
  HealthStatus,
  TraceDetailResponse,
} from "../types";

const BASE_URL = "";

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let errorDetail = `HTTP ${res.status}: ${res.statusText}`;
    try {
      const errJson = await res.json();
      if (errJson.detail) {
        errorDetail = typeof errJson.detail === "string" ? errJson.detail : JSON.stringify(errJson.detail);
      }
    } catch {
      // Ignore JSON parse error on non-JSON response
    }
    throw new Error(errorDetail);
  }
  return res.json();
}

export const api = {
  async getHealth(): Promise<HealthStatus> {
    const res = await fetch(`${BASE_URL}/api/health`);
    return handleResponse<HealthStatus>(res);
  },

  async listDatasets(): Promise<DatasetSummary[]> {
    const res = await fetch(`${BASE_URL}/api/datasets`);
    const data = await handleResponse<{ datasets: DatasetSummary[] }>(res);
    return data.datasets;
  },

  async uploadDataset(file: File): Promise<DatasetSchema> {
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch(`${BASE_URL}/api/datasets`, {
      method: "POST",
      body: formData,
    });
    return handleResponse<DatasetSchema>(res);
  },

  async getDatasetSchema(name: string): Promise<DatasetSchema> {
    const res = await fetch(`${BASE_URL}/api/datasets/${encodeURIComponent(name)}/schema`);
    return handleResponse<DatasetSchema>(res);
  },

  async listDocuments(): Promise<DocumentInfo[]> {
    const res = await fetch(`${BASE_URL}/api/documents`);
    const data = await handleResponse<{ documents: DocumentInfo[] }>(res);
    return data.documents;
  },

  async uploadDocument(file: File): Promise<DocumentInfo> {
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch(`${BASE_URL}/api/documents`, {
      method: "POST",
      body: formData,
    });
    return handleResponse<DocumentInfo>(res);
  },

  async deleteDocument(id: string): Promise<{ status: string; id: string }> {
    const res = await fetch(`${BASE_URL}/api/documents/${encodeURIComponent(id)}`, {
      method: "DELETE",
    });
    return handleResponse<{ status: string; id: string }>(res);
  },

  async getTrace(traceId: string): Promise<TraceDetailResponse> {
    const res = await fetch(`${BASE_URL}/api/traces/${encodeURIComponent(traceId)}`);
    return handleResponse<TraceDetailResponse>(res);
  },
};
