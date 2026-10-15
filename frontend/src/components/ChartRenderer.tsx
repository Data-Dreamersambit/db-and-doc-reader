/**
 * Recharts visualizer driven by chart_spec and falling back to DataTable for tabular views.
 *
 * Trade-off: ResponsiveContainer adapts charts dynamically to inspector panel resize events.
 * Automatic fallback to DataTable ensures raw records are always readable when unsuited for 2D charts.
 */
import React from "react";
import {
  Bar,
  BarChart,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ChartSpec } from "../types";
import { DataTable } from "./DataTable";

interface ChartRendererProps {
  spec?: ChartSpec;
  rows?: Record<string, any>[];
  columns?: string[];
  className?: string;
}

const PALETTE = ["#10b981", "#3b82f6", "#f59e0b", "#ec4899", "#8b5cf6", "#06b6d4"];

export const ChartRenderer: React.FC<ChartRendererProps> = ({
  spec,
  rows,
  columns,
  className = "",
}) => {
  if (!rows || rows.length === 0) {
    return (
      <div className="text-xs text-slate-500 py-6 text-center border border-dashed border-slate-800 rounded-lg">
        No records available for visualization
      </div>
    );
  }

  const chartType = spec?.chart_type || "table";

  // If table requested, use sortable and paginated DataTable
  if (chartType === "table") {
    return <DataTable columns={columns} rows={rows} className={className} />;
  }

  const xField = spec?.x_field || (columns ? columns[0] : Object.keys(rows[0])[0]);
  const yFields = spec?.y_fields?.length
    ? spec.y_fields
    : (columns ? columns.slice(1, 3) : Object.keys(rows[0]).slice(1, 3));

  return (
    <div className={`w-full space-y-2 ${className}`}>
      {spec?.title && (
        <div className="text-xs font-semibold text-slate-300 text-center tracking-wide">
          {spec.title}
        </div>
      )}
      <div className="w-full h-64 p-2 bg-slate-950/80 rounded-xl border border-slate-800">
        <ResponsiveContainer width="100%" height="100%">
          {chartType === "bar" ? (
            <BarChart data={rows} margin={{ top: 10, right: 10, left: -10, bottom: 25 }}>
              <XAxis
                dataKey={xField}
                stroke="#64748b"
                fontSize={11}
                tickLine={false}
                angle={-15}
                textAnchor="end"
              />
              <YAxis stroke="#64748b" fontSize={11} tickLine={false} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#0f172a",
                  borderColor: "#334155",
                  fontSize: "12px",
                  borderRadius: "8px",
                }}
              />
              <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "8px" }} />
              {yFields.map((y, idx) => (
                <Bar
                  key={y}
                  dataKey={y}
                  fill={PALETTE[idx % PALETTE.length]}
                  radius={[4, 4, 0, 0]}
                />
              ))}
            </BarChart>
          ) : chartType === "line" ? (
            <LineChart data={rows} margin={{ top: 10, right: 10, left: -10, bottom: 25 }}>
              <XAxis
                dataKey={xField}
                stroke="#64748b"
                fontSize={11}
                tickLine={false}
                angle={-15}
                textAnchor="end"
              />
              <YAxis stroke="#64748b" fontSize={11} tickLine={false} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#0f172a",
                  borderColor: "#334155",
                  fontSize: "12px",
                  borderRadius: "8px",
                }}
              />
              <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "8px" }} />
              {yFields.map((y, idx) => (
                <Line
                  key={y}
                  type="monotone"
                  dataKey={y}
                  stroke={PALETTE[idx % PALETTE.length]}
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  activeDot={{ r: 5 }}
                />
              ))}
            </LineChart>
          ) : (
            <PieChart>
              <Pie
                data={rows}
                dataKey={yFields[0] || Object.keys(rows[0])[1]}
                nameKey={xField}
                cx="50%"
                cy="50%"
                outerRadius={75}
                innerRadius={30}
                paddingAngle={2}
              >
                {rows.map((_, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={PALETTE[index % PALETTE.length]}
                  />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  backgroundColor: "#0f172a",
                  borderColor: "#334155",
                  fontSize: "12px",
                  borderRadius: "8px",
                }}
              />
              <Legend wrapperStyle={{ fontSize: "11px" }} />
            </PieChart>
          )}
        </ResponsiveContainer>
      </div>
    </div>
  );
};
