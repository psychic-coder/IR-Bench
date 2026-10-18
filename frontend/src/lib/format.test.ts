import { describe, expect, it } from "vitest";

import { fmtBytes, fmtCompact, fmtMs, fmtNumber, logTicks, paretoFrontier, sortRows } from "@/lib/format";

describe("formatters", () => {
  it("formats missing values as an en dash", () => {
    expect(fmtNumber(null)).toBe("–");
    expect(fmtMs(undefined)).toBe("–");
    expect(fmtBytes(Number.NaN)).toBe("–");
  });

  it("formats milliseconds with adaptive precision", () => {
    expect(fmtMs(3.14159)).toBe("3.1 ms");
    expect(fmtMs(250.4)).toBe("250 ms");
    expect(fmtMs(1500)).toBe("1.50 s");
  });

  it("formats bytes with binary units", () => {
    expect(fmtBytes(512)).toBe("512 B");
    expect(fmtBytes(1536)).toBe("1.5 KB");
    expect(fmtBytes(5 * 1024 * 1024)).toBe("5.0 MB");
  });
});

describe("paretoFrontier", () => {
  it("keeps only non-dominated (fast and accurate) points", () => {
    const pts = [
      { x: 10, y: 50 },
      { x: 5, y: 40 },
      { x: 20, y: 45 }, // dominated by (10, 50)
      { x: 30, y: 70 },
      { x: 5, y: 30 }, // dominated by (5, 40)
    ];
    expect(paretoFrontier(pts)).toEqual([
      { x: 5, y: 40 },
      { x: 10, y: 50 },
      { x: 30, y: 70 },
    ]);
  });
});

describe("sortRows", () => {
  const rows = [{ v: 2 }, { v: null }, { v: 5 }, { v: 1 }];
  it("sorts descending with nulls last", () => {
    expect(sortRows(rows, (r) => r.v, "desc").map((r) => r.v)).toEqual([5, 2, 1, null]);
  });
  it("sorts ascending with nulls last", () => {
    expect(sortRows(rows, (r) => r.v, "asc").map((r) => r.v)).toEqual([1, 2, 5, null]);
  });
});

describe("logTicks", () => {
  it("produces 1-2-5 ticks inside a padded domain", () => {
    const { ticks, domain } = logTicks(3, 400);
    expect(ticks).toEqual([2, 5, 10, 20, 50, 100, 200, 500]);
    expect(domain[0]).toBeLessThan(2);
    expect(domain[1]).toBeGreaterThan(500);
  });
  it("handles non-positive input", () => {
    expect(logTicks(0, 10).ticks).toEqual([]);
  });
  it("formats compact labels", () => {
    expect(fmtCompact(1536)).toBe("1.54k");
    expect(fmtCompact(2_000_000)).toBe("2M");
    expect(fmtCompact(0.5)).toBe("0.5");
  });
});
