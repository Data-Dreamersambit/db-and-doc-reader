/**
 * Sortable and paginated data table component for DuckDB query record inspection.
 *
 * Trade-off: Client-side sorting and slicing avoids redundant DuckDB roundtrips for <= 1000 rows.
 * Compact pagination controls keep screen real estate focused on analytical answers.
 */
import React, { useMemo, useState } from "react";
import {
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  ChevronsUpDown,
} from "lucide-react";

interface DataTableProps {
  columns?: string[];
  rows: Record<string, any>[];
  defaultPageSize?: number;
  className?: string;
}

type SortDirection = "asc" | "desc" | null;

export const DataTable: React.FC<DataTableProps> = ({
  columns,
  rows,
  defaultPageSize = 10,
  className = "",
}) => {
  const [sortColumn, setSortColumn] = useState<string | null>(null);
  const [sortDirection, setSortDirection] = useState<SortDirection>(null);
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [pageSize, setPageSize] = useState<number>(defaultPageSize);

  const displayColumns = useMemo(() => {
    if (columns && columns.length > 0) return columns;
    if (rows.length > 0) return Object.keys(rows[0]);
    return [];
  }, [columns, rows]);

  const handleSort = (col: string) => {
    if (sortColumn !== col) {
      setSortColumn(col);
      setSortDirection("asc");
    } else if (sortDirection === "asc") {
      setSortDirection("desc");
    } else {
      setSortColumn(null);
      setSortDirection(null);
    }
    setCurrentPage(1);
  };

  const sortedRows = useMemo(() => {
    if (!sortColumn || !sortDirection) return rows;

    return [...rows].sort((a, b) => {
      const valA = a[sortColumn];
      const valB = b[sortColumn];

      if (valA === valB) return 0;
      if (valA === null || valA === undefined) return 1;
      if (valB === null || valB === undefined) return -1;

      if (typeof valA === "number" && typeof valB === "number") {
        return sortDirection === "asc" ? valA - valB : valB - valA;
      }

      const strA = String(valA).toLowerCase();
      const strB = String(valB).toLowerCase();
      return sortDirection === "asc"
        ? strA.localeCompare(strB)
        : strB.localeCompare(strA);
    });
  }, [rows, sortColumn, sortDirection]);

  const totalPages = Math.max(1, Math.ceil(sortedRows.length / pageSize));
  const startIndex = (currentPage - 1) * pageSize;
  const paginatedRows = sortedRows.slice(startIndex, startIndex + pageSize);

  if (!rows || rows.length === 0) {
    return (
      <div className="p-4 text-xs text-slate-500 text-center border border-dashed border-slate-800 rounded-lg">
        No records to display
      </div>
    );
  }

  return (
    <div className={`space-y-2.5 ${className}`}>
      {/* Table Container */}
      <div className="overflow-x-auto border border-slate-800 rounded-lg max-h-80 shadow-inner">
        <table className="w-full text-left text-xs border-collapse">
          <thead className="bg-slate-900/90 text-slate-400 sticky top-0 border-b border-slate-800 z-10 select-none">
            <tr>
              {displayColumns.map((col) => {
                const isSorted = sortColumn === col;
                return (
                  <th
                    key={col}
                    onClick={() => handleSort(col)}
                    className="px-3 py-2 font-medium cursor-pointer hover:bg-slate-800/80 transition"
                  >
                    <div className="flex items-center gap-1.5 font-mono">
                      <span>{col}</span>
                      {isSorted && sortDirection === "asc" ? (
                        <ChevronUp className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                      ) : isSorted && sortDirection === "desc" ? (
                        <ChevronDown className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                      ) : (
                        <ChevronsUpDown className="w-3 h-3 text-slate-600 shrink-0 hover:text-slate-400" />
                      )}
                    </div>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 font-mono">
            {paginatedRows.map((row, rowIdx) => (
              <tr key={rowIdx} className="hover:bg-slate-900/50 transition">
                {displayColumns.map((col) => {
                  const val = row[col];
                  const isNum = typeof val === "number";
                  return (
                    <td
                      key={col}
                      className={`px-3 py-2 text-slate-300 text-xs ${
                        isNum ? "text-emerald-400/90" : ""
                      }`}
                    >
                      {val !== null && val !== undefined ? String(val) : "-"}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination Footer */}
      <div className="flex items-center justify-between text-xs text-slate-400 px-1 pt-1">
        <div className="text-[11px] font-mono">
          Showing {sortedRows.length === 0 ? 0 : startIndex + 1}–
          {Math.min(startIndex + pageSize, sortedRows.length)} of {sortedRows.length} records
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1 text-[11px]">
            <span>Page size:</span>
            <select
              value={pageSize}
              onChange={(e) => {
                setPageSize(Number(e.target.value));
                setCurrentPage(1);
              }}
              className="bg-slate-900 border border-slate-800 rounded px-1.5 py-0.5 text-slate-300 font-mono text-[11px]"
            >
              {[5, 10, 25, 50].map((sz) => (
                <option key={sz} value={sz}>
                  {sz}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-1 font-mono text-[11px]">
            <button
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              disabled={currentPage <= 1}
              className="p-1 rounded hover:bg-slate-800 text-slate-300 disabled:opacity-30 disabled:hover:bg-transparent"
              title="Previous Page"
            >
              <ChevronLeft className="w-3.5 h-3.5" />
            </button>
            <span>
              {currentPage} / {totalPages}
            </span>
            <button
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              disabled={currentPage >= totalPages}
              className="p-1 rounded hover:bg-slate-800 text-slate-300 disabled:opacity-30 disabled:hover:bg-transparent"
              title="Next Page"
            >
              <ChevronRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
