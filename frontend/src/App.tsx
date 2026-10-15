/**
 * AgenticInsight main application layout coordinating Sources, Chat, and Inspector panels.
 */
import { useEffect, useState } from "react";
import { Cpu, HardDrive, Moon, Sun } from "lucide-react";
import { api } from "./api/client";
import { ChatPanel } from "./components/ChatPanel";
import { InspectorPanel } from "./components/InspectorPanel";
import { SourcesPanel } from "./components/SourcesPanel";
import { useChatStream } from "./hooks/useChatStream";
import {
  ChatMessage,
  ChunkMatch,
  DatasetSummary,
  DocumentInfo,
  HealthStatus,
} from "./types";

export default function App() {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [sourcesLoading, setSourcesLoading] = useState<boolean>(false);
  const [selectedMessage, setSelectedMessage] = useState<ChatMessage | null>(null);
  const [selectedCitation, setSelectedCitation] = useState<string | null>(null);
  const [theme, setTheme] = useState<"dark" | "light">(() => {
    if (typeof window !== "undefined") {
      return (localStorage.getItem("theme") as "dark" | "light") || "dark";
    }
    return "dark";
  });

  useEffect(() => {
    if (theme === "dark") {
      document.documentElement.classList.add("dark");
      document.documentElement.classList.remove("light");
    } else {
      document.documentElement.classList.add("light");
      document.documentElement.classList.remove("dark");
    }
    localStorage.setItem("theme", theme);
  }, [theme]);

  const {
    messages,
    isLoading,
    error,
    sendMessage,
    cancel,
    clearMessages,
  } = useChatStream();

  const fetchSources = async () => {
    setSourcesLoading(true);
    try {
      const [dsets, docs] = await Promise.all([
        api.listDatasets(),
        api.listDocuments(),
      ]);
      setDatasets(dsets);
      setDocuments(docs);
    } catch (err) {
      console.error("Failed to load sources:", err);
    } finally {
      setSourcesLoading(false);
    }
  };

  const fetchHealth = async () => {
    try {
      const h = await api.getHealth();
      setHealth(h);
    } catch (err) {
      console.warn("Backend health check failed:", err);
    }
  };

  useEffect(() => {
    fetchHealth();
    fetchSources();
  }, []);

  // Update inspector when a new assistant message arrives or is updated
  useEffect(() => {
    const lastAssistant = [...messages].reverse().find((m) => m.role === "assistant");
    if (lastAssistant) {
      setSelectedMessage(lastAssistant);
    }
  }, [messages]);

  const handleSelectCitation = (citation: string, _chunk?: ChunkMatch) => {
    setSelectedCitation(citation);
  };

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-slate-950 font-sans">
      {/* Top Application Header */}
      <header className="h-12 border-b border-slate-800 bg-slate-900/90 px-4 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 rounded bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center text-emerald-400 font-bold text-xs">
            AI
          </div>
          <span className="font-bold text-sm text-slate-100 tracking-tight">
            AgenticInsight
          </span>
          <span className="text-[11px] text-slate-400 font-mono ml-2 hidden sm:inline">
            Local-First RAG & SQL
          </span>
        </div>

        {/* Local Ollama Daemon Status Badge */}
        <div className="flex items-center gap-3 text-xs">
          {health ? (
            <div className="flex items-center gap-2 px-2.5 py-1 rounded-full bg-slate-800/80 border border-slate-700/80 font-mono text-[11px]">
              <span
                className={`w-2 h-2 rounded-full ${
                  health.ollama.reachable ? "bg-emerald-400" : "bg-rose-400"
                }`}
              />
              <span className="text-slate-300">
                {health.ollama.chat_model}
              </span>
              <span className="text-slate-500">•</span>
              <span className="text-slate-400">
                {health.ollama.embed_model}
              </span>
            </div>
          ) : (
            <div className="flex items-center gap-1.5 text-slate-500 text-xs">
              <Cpu className="w-3.5 h-3.5 animate-pulse" />
              <span>Connecting to Ollama...</span>
            </div>
          )}

          <div className="hidden md:flex items-center gap-1.5 text-slate-400 text-xs font-mono">
            <HardDrive className="w-3.5 h-3.5 text-emerald-400" />
            <span>DuckDB + SQLite</span>
          </div>

          {/* Theme Toggle Button */}
          <button
            onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-slate-100 transition border border-slate-700"
            title={theme === "dark" ? "Switch to Light Mode" : "Switch to Dark Mode"}
          >
            {theme === "dark" ? (
              <Sun className="w-3.5 h-3.5 text-amber-400" />
            ) : (
              <Moon className="w-3.5 h-3.5 text-sky-400" />
            )}
          </button>
        </div>
      </header>

      {/* 3-Panel Main Layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel: Sources */}
        <SourcesPanel
          datasets={datasets}
          documents={documents}
          isLoading={sourcesLoading}
          onRefresh={fetchSources}
        />

        {/* Center Panel: Chat */}
        <ChatPanel
          messages={messages}
          isLoading={isLoading}
          error={error}
          onSendMessage={sendMessage}
          onCancel={cancel}
          onClear={() => {
            clearMessages();
            setSelectedMessage(null);
            setSelectedCitation(null);
          }}
          onSelectMessage={(msg) => setSelectedMessage(msg)}
          onSelectCitation={handleSelectCitation}
          selectedMessageId={selectedMessage?.id}
        />

        {/* Right Panel: Inspector */}
        <InspectorPanel
          message={selectedMessage}
          selectedCitation={selectedCitation}
          onSelectCitation={handleSelectCitation}
        />
      </div>
    </div>
  );
}
