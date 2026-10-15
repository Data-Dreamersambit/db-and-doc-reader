/**
 * Center chat panel rendering conversation messages, streaming tokens, steps, and input controls.
 */
import React, { useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  BarChart2,
  Bot,
  Code2,
  FileText,
  Loader2,
  Mic,
  MicOff,
  Send,
  Square,
  Trash2,
  User,
} from "lucide-react";
import { ChatMessage, ChunkMatch } from "../types";
import { CitationChips } from "./CitationChips";
import { RouteBadge } from "./RouteBadge";
import { StepTimeline } from "./StepTimeline";

interface ChatPanelProps {
  messages: ChatMessage[];
  isLoading: boolean;
  error: string | null;
  onSendMessage: (query: string) => void;
  onCancel: () => void;
  onClear: () => void;
  onSelectMessage: (msg: ChatMessage) => void;
  onSelectCitation?: (citation: string, chunk?: ChunkMatch) => void;
  selectedMessageId?: string;
}

export const ChatPanel: React.FC<ChatPanelProps> = ({
  messages,
  isLoading,
  error,
  onSendMessage,
  onCancel,
  onClear,
  onSelectMessage,
  onSelectCitation,
  selectedMessageId,
}) => {
  const [input, setInput] = useState<string>("");
  const [isListening, setIsListening] = useState<boolean>(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const recognitionRef = useRef<any>(null);

  const SpeechRecognitionConstructor =
    typeof window !== "undefined"
      ? (window as any).SpeechRecognition ||
        (window as any).webkitSpeechRecognition
      : null;
  const isSpeechSupported = Boolean(SpeechRecognitionConstructor);

  const toggleListening = () => {
    if (!SpeechRecognitionConstructor) return;

    if (isListening) {
      recognitionRef.current?.stop();
      setIsListening(false);
      return;
    }

    try {
      const recognition = new SpeechRecognitionConstructor();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = "en-US";

      recognition.onresult = (event: any) => {
        let transcript = "";
        for (let i = 0; i < event.results.length; i++) {
          transcript += event.results[i][0].transcript;
        }
        setInput(transcript);
      };

      recognition.onerror = () => setIsListening(false);
      recognition.onend = () => setIsListening(false);

      recognition.start();
      recognitionRef.current = recognition;
      setIsListening(true);
    } catch {
      setIsListening(false);
    }
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    onSendMessage(input);
    setInput("");
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <main className="flex-1 flex flex-col h-full bg-slate-950 relative">
      {/* Top action bar */}
      <div className="h-12 border-b border-slate-800 px-6 flex items-center justify-between text-xs text-slate-400">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-slate-200">Session Chat</span>
          <span>•</span>
          <span className="text-slate-500">{messages.length} messages</span>
        </div>
        <button
          onClick={onClear}
          disabled={messages.length === 0}
          className="inline-flex items-center gap-1.5 px-2.5 py-1 text-slate-400 hover:text-slate-200 hover:bg-slate-900 rounded transition disabled:opacity-40"
        >
          <Trash2 className="w-3.5 h-3.5" />
          Clear Chat
        </button>
      </div>

      {/* Messages Feed */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center max-w-md mx-auto text-slate-500 space-y-4">
            <div className="w-12 h-12 rounded-2xl bg-emerald-950/60 border border-emerald-800/80 flex items-center justify-center text-emerald-400 shadow-lg">
              <Bot className="w-6 h-6" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-slate-300">AgenticInsight Ready</h3>
              <p className="text-xs text-slate-500 mt-1">
                Ask questions over loaded DuckDB datasets or vector indexed PDF documents. The router agent will classify and synthesize answers local-first.
              </p>
            </div>
            <div className="flex flex-wrap gap-2 justify-center">
              {[
                "What is the total sales revenue by region?",
                "Which product category had the highest average discount?",
                "What are the employee meal reimbursement guidelines?",
              ].map((sample) => (
                <button
                  key={sample}
                  onClick={() => onSendMessage(sample)}
                  className="text-xs bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 px-3 py-1.5 rounded-lg transition text-left"
                >
                  {sample}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              onClick={() => msg.role === "assistant" && onSelectMessage(msg)}
              className={`flex gap-3 max-w-3xl ${
                msg.role === "user" ? "ml-auto justify-end" : "mr-auto"
              }`}
            >
              {msg.role === "assistant" && (
                <div className="w-8 h-8 rounded-lg bg-emerald-950/80 border border-emerald-800 flex items-center justify-center text-emerald-400 shrink-0 mt-0.5 shadow-sm">
                  <Bot className="w-4 h-4" />
                </div>
              )}

              <div
                className={`flex flex-col space-y-2 rounded-2xl p-4 text-xs max-w-xl transition ${
                  msg.role === "user"
                    ? "bg-emerald-600 text-white rounded-tr-none shadow-md"
                    : `bg-slate-900/90 text-slate-200 border rounded-tl-none cursor-pointer ${
                        selectedMessageId === msg.id
                          ? "border-emerald-500/80 shadow-md ring-1 ring-emerald-500/30"
                          : "border-slate-800 hover:border-slate-700"
                      }`
                }`}
              >
                {/* Assistant Metadata Badges */}
                {msg.role === "assistant" && (
                  <div className="flex items-center gap-2 mb-1 flex-wrap">
                    <RouteBadge route={msg.route} />

                    {msg.isStreaming && (
                      <span className="inline-flex items-center gap-1 text-[10px] text-amber-400 font-mono">
                        <Loader2 className="w-3 h-3 animate-spin" />
                        Generating
                      </span>
                    )}
                  </div>
                )}

                {/* Step timeline */}
                {msg.role === "assistant" && msg.steps && msg.steps.length > 0 && (
                  <StepTimeline steps={msg.steps} className="my-1" />
                )}

                {/* Message text content */}
                <div className="leading-relaxed whitespace-pre-wrap font-sans text-xs">
                  {msg.content || (msg.isStreaming ? "Thinking..." : "")}
                </div>

                {/* Citations chips inline */}
                {msg.role === "assistant" && msg.citations && msg.citations.length > 0 && (
                  <div className="pt-1">
                    <CitationChips
                      citations={msg.citations}
                      chunks={msg.chunks}
                      onSelectCitation={(cite, chk) => {
                        onSelectMessage(msg);
                        onSelectCitation?.(cite, chk);
                      }}
                    />
                  </div>
                )}

                {/* Quick inspect buttons */}
                {msg.role === "assistant" && (
                  <div className="flex items-center gap-3 pt-2 border-t border-slate-800/80 mt-2 text-[11px]">
                    {msg.sql && (
                      <span className="inline-flex items-center gap-1 text-emerald-400 hover:text-emerald-300 font-mono">
                        <Code2 className="w-3 h-3" />
                        SQL Query
                      </span>
                    )}
                    {msg.rows && msg.rows.length > 0 && (
                      <span className="inline-flex items-center gap-1 text-sky-400 hover:text-sky-300 font-mono">
                        <BarChart2 className="w-3 h-3" />
                        {msg.rows.length} rows
                      </span>
                    )}
                    {msg.citations && msg.citations.length > 0 && (
                      <span className="inline-flex items-center gap-1 text-amber-400 hover:text-amber-300 font-mono">
                        <FileText className="w-3 h-3" />
                        {msg.citations.length} sources
                      </span>
                    )}
                  </div>
                )}
              </div>

              {msg.role === "user" && (
                <div className="w-8 h-8 rounded-lg bg-emerald-700/80 flex items-center justify-center text-white shrink-0 mt-0.5 shadow-sm">
                  <User className="w-4 h-4" />
                </div>
              )}
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Error banner */}
      {error && (
        <div className="mx-6 mb-2 p-2.5 bg-rose-950/80 border border-rose-800 rounded-lg text-rose-300 text-xs flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Input controls bottom bar */}
      <div className="p-4 border-t border-slate-800 bg-slate-900/60">
        <form onSubmit={handleSubmit} className="relative max-w-4xl mx-auto">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your datasets or documents (e.g., 'What is total revenue by region?')..."
            rows={2}
            className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 pr-24 text-xs text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-1 focus:ring-emerald-500/80 resize-none font-sans"
          />
          <div className="absolute right-3 bottom-3 flex items-center gap-1.5">
            {isSpeechSupported && (
              <button
                type="button"
                onClick={toggleListening}
                className={`p-1.5 rounded-lg text-xs transition border ${
                  isListening
                    ? "bg-rose-950/80 border-rose-500 text-rose-300 animate-pulse ring-2 ring-rose-500/50"
                    : "bg-slate-800/80 border-slate-700/80 text-slate-400 hover:text-slate-200 hover:bg-slate-800"
                }`}
                title={isListening ? "Listening... click to stop" : "Speak question (Speech Recognition)"}
              >
                {isListening ? (
                  <MicOff className="w-3.5 h-3.5 text-rose-400" />
                ) : (
                  <Mic className="w-3.5 h-3.5 text-slate-300" />
                )}
              </button>
            )}

            {isLoading ? (
              <button
                type="button"
                onClick={onCancel}
                className="inline-flex items-center gap-1 px-3 py-1.5 bg-rose-600 hover:bg-rose-500 text-white rounded-lg text-xs font-medium transition shadow"
              >
                <Square className="w-3 h-3 fill-current" />
                Stop
              </button>
            ) : (
              <button
                type="submit"
                disabled={!input.trim()}
                className="inline-flex items-center gap-1 px-3.5 py-1.5 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-white rounded-lg text-xs font-medium transition shadow"
              >
                <Send className="w-3 h-3" />
                Send
              </button>
            )}
          </div>
        </form>
      </div>
    </main>
  );
};
