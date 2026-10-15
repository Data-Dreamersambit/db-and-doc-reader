/**
 * Hook for consuming Server-Sent Events chat streams with incremental state updates.
 *
 * Trade-off: Stream decoding via native TextDecoder parses NDJSON events without third-party SSE libs.
 * AbortController signals immediate backend cancellation to free M1 CPU/GPU inference.
 */
import { useCallback, useRef, useState } from "react";
import { AgentStep, ChatMessage } from "../types";

export function useChatStream() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string>(() => crypto.randomUUID());
  const abortControllerRef = useRef<AbortController | null>(null);

  const cancel = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setIsLoading(false);
      setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (last && last.role === "assistant" && last.isStreaming) {
          return [
            ...prev.slice(0, -1),
            { ...last, isStreaming: false, content: last.content + " [cancelled]" },
          ];
        }
        return prev;
      });
    }
  }, []);

  const clearMessages = useCallback(() => {
    cancel();
    setMessages([]);
    setError(null);
    setSessionId(crypto.randomUUID());
  }, [cancel]);

  const sendMessage = useCallback(
    async (prompt: string) => {
      const trimmed = prompt.trim();
      if (!trimmed || isLoading) return;

      setError(null);
      setIsLoading(true);

      const userMsgId = crypto.randomUUID();
      const assistantMsgId = crypto.randomUUID();

      const userMsg: ChatMessage = {
        id: userMsgId,
        role: "user",
        content: trimmed,
      };

      const assistantMsg: ChatMessage = {
        id: assistantMsgId,
        role: "assistant",
        content: "",
        isStreaming: true,
        steps: [],
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);

      const controller = new AbortController();
      abortControllerRef.current = controller;

      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: trimmed,
            session_id: sessionId,
          }),
          signal: controller.signal,
        });

        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`);
        }

        const reader = res.body?.getReader();
        if (!reader) {
          throw new Error("Response body is not readable.");
        }

        const decoder = new TextDecoder("utf-8");
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const parts = buffer.split("\n\n");
          buffer = parts.pop() || "";

          for (const block of parts) {
            if (!block.trim()) continue;

            let eventType = "message";
            let dataStr = "";

            const lines = block.split("\n");
            for (const line of lines) {
              if (line.startsWith("event: ")) {
                eventType = line.slice(7).trim();
              } else if (line.startsWith("data: ")) {
                dataStr = line.slice(6).trim();
              }
            }

            if (!dataStr) continue;

            try {
              const payload = JSON.parse(dataStr);

              setMessages((prev) => {
                return prev.map((msg) => {
                  if (msg.id !== assistantMsgId) return msg;

                  switch (eventType) {
                    case "route":
                      return {
                        ...msg,
                        route: payload.route,
                      };

                    case "step": {
                      const newStep: AgentStep = payload;
                      const existingSteps = msg.steps || [];
                      const stepIdx = existingSteps.findIndex(
                        (s) => s.step === newStep.step
                      );
                      let updatedSteps: AgentStep[];
                      if (stepIdx >= 0) {
                        updatedSteps = [...existingSteps];
                        updatedSteps[stepIdx] = newStep;
                      } else {
                        updatedSteps = [...existingSteps, newStep];
                      }
                      return { ...msg, steps: updatedSteps };
                    }

                    case "sql":
                      return {
                        ...msg,
                        sql: payload.sql,
                        lineage: payload.lineage,
                      };

                    case "rows":
                      return {
                        ...msg,
                        columns: payload.columns,
                        rows: payload.rows,
                        chart_spec: payload.chart_spec,
                      };

                    case "hits":
                      return {
                        ...msg,
                        chunks: payload.chunks,
                      };

                    case "token":
                      return {
                        ...msg,
                        content: msg.content + (payload.delta || ""),
                      };

                    case "final":
                      return {
                        ...msg,
                        content: payload.answer || msg.content,
                        route: payload.route || msg.route,
                        sql: payload.sql || msg.sql,
                        columns: payload.columns || msg.columns,
                        rows: payload.rows || msg.rows,
                        chart_spec: payload.chart_spec || msg.chart_spec,
                        lineage: payload.lineage || msg.lineage,
                        citations: payload.citations || msg.citations,
                        isStreaming: false,
                      };

                    case "error":
                      setError(payload.message || "Unknown error occurred.");
                      return {
                        ...msg,
                        error: payload.message,
                        isStreaming: false,
                      };

                    default:
                      return msg;
                  }
                });
              });
            } catch (err) {
              console.warn("Failed to parse SSE JSON block:", dataStr, err);
            }
          }
        }
      } catch (err: any) {
        if (err.name === "AbortError") {
          console.log("Chat stream aborted by user.");
        } else {
          const errMsg = err.message || "Failed to stream chat response.";
          setError(errMsg);
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId
                ? { ...m, isStreaming: false, error: errMsg }
                : m
            )
          );
        }
      } finally {
        setIsLoading(false);
        abortControllerRef.current = null;
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsgId ? { ...m, isStreaming: false } : m
          )
        );
      }
    },
    [isLoading, sessionId]
  );

  return {
    messages,
    isLoading,
    error,
    sessionId,
    sendMessage,
    cancel,
    clearMessages,
  };
}
