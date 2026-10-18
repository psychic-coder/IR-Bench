import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { LatencyBreakdown } from "@/components/LatencyBreakdown";
import { ResultGrid } from "@/components/ResultGrid";
import { SortableTable, type Column } from "@/components/SortableTable";
import { WarmingUp } from "@/components/StateViews";

describe("LatencyBreakdown", () => {
  it("shows every stage, the total and the cache badge", () => {
    render(
      <LatencyBreakdown latency={{ extract: 12.5, search: 1.25, rerank: 40, total: 53.75 }} cache="hit" />,
    );
    expect(screen.getByText("extract")).toBeInTheDocument();
    expect(screen.getByText("12.5 ms")).toBeInTheDocument();
    expect(screen.getByText("40.0 ms")).toBeInTheDocument();
    expect(screen.getByText("53.8 ms")).toBeInTheDocument();
    expect(screen.getByTestId("cache-badge")).toHaveTextContent("embedding cache hit");
  });
});

describe("SortableTable", () => {
  interface Row {
    name: string;
    score: number;
  }
  const rows: Row[] = [
    { name: "b", score: 2 },
    { name: "a", score: 3 },
    { name: "c", score: 1 },
  ];
  const columns: Column<Row>[] = [
    { key: "name", header: "Name", value: (r) => r.name },
    { key: "score", header: "Score", value: (r) => r.score, numeric: true },
  ];

  const firstColumn = () =>
    screen
      .getAllByRole("row")
      .slice(1)
      .map((tr) => within(tr).getAllByRole("cell")[0]?.textContent);

  it("sorts by the initial column and toggles direction on click", async () => {
    render(
      <SortableTable
        rows={rows}
        columns={columns}
        rowKey={(r) => r.name}
        initialSort={{ key: "score", dir: "desc" }}
      />,
    );
    expect(firstColumn()).toEqual(["a", "b", "c"]);
    await userEvent.click(screen.getByRole("button", { name: /Score/ }));
    expect(firstColumn()).toEqual(["c", "b", "a"]);
    expect(screen.getByRole("columnheader", { name: /Score/ })).toHaveAttribute("aria-sort", "ascending");
  });
});

describe("ResultGrid", () => {
  it("highlights results unique to a configuration", () => {
    render(
      <ResultGrid
        results={[
          { rank: 1, id: "oxford5k/x", score: 0.9, image_url: "", thumbnail_url: "", inliers: 42 },
          { rank: 2, id: "oxford5k/y", score: 0.8, image_url: "", thumbnail_url: "", inliers: null },
        ]}
        highlight={new Set(["oxford5k/y"])}
      />,
    );
    expect(screen.getByText("42 RANSAC inliers")).toBeInTheDocument();
    expect(screen.getAllByText("only in this config")).toHaveLength(1);
  });
});

describe("WarmingUp", () => {
  it("renders progress while warming up", () => {
    render(
      <WarmingUp
        status={{
          ready: false,
          stage: "extracting",
          message: "extracting dinov2",
          progress: { done: 50, total: 200 },
          built: [],
          dataset: "oxford5k",
        }}
      />,
    );
    expect(screen.getByText("Backend warming up")).toBeInTheDocument();
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "25");
  });

  it("renders manual download instructions on error", () => {
    render(
      <WarmingUp
        status={{
          ready: false,
          stage: "error",
          message: "dataset unavailable",
          error: "Open https://example and place files in ./data/oxford5k",
          built: [],
          dataset: "oxford5k",
        }}
      />,
    );
    expect(screen.getByText(/place files in .\/data\/oxford5k/)).toBeInTheDocument();
  });
});
