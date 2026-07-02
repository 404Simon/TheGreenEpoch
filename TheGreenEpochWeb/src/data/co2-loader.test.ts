import { describe, it, expect } from "vitest";
import type { YearCO2 } from "../domain/types";
import { averageYears } from "./co2-loader";

function makeYear(zone: string, year: number, len: number, offset: number): YearCO2 {
  const ts: string[] = [];
  const ci: number[] = [];
  for (let i = 0; i < len; i++) {
    const d = new Date(Date.UTC(year, 0, 1, 0, i * 5));
    ts.push(d.toISOString());
    ci.push(100 + offset + Math.sin(i * 0.1) * 50);
  }
  return { zone, year, timestamps: ts, carbonIntensity: ci, isEstimated: [] };
}

describe("averageYears", () => {
  it("single year returns same data", () => {
    const data = makeYear("DE", 2022, 10, 0);
    const result = averageYears([data]);
    expect(result.zone).toBe("DE");
    expect(result.years).toEqual([2022]);
    expect(result.timestamps).toEqual(data.timestamps);
    expect(result.carbonIntensity).toEqual(data.carbonIntensity);
  });

  it("averages two years with matching timestamps", () => {
    const y1 = makeYear("DE", 2022, 10, 0);
    const y2 = makeYear("DE", 2023, 10, 10);
    const result = averageYears([y1, y2]);

    expect(result.years).toEqual([2022, 2023]);
    expect(result.timestamps.length).toBe(10);

    for (let i = 0; i < 10; i++) {
      // y1: 100 + sin, y2: 110 + sin
      const expected = (y1.carbonIntensity[i] + y2.carbonIntensity[i]) / 2;
      expect(result.carbonIntensity[i]).toBeCloseTo(expected, 6);
    }
  });

  it("handles Feb 29 → Feb 28 collision (leap year)", () => {
    const leapYear: YearCO2 = {
      zone: "DE",
      year: 2024,
      timestamps: [
        "2024-02-28T00:00:00.000Z",
        "2024-02-29T00:00:00.000Z",
        "2024-03-01T00:00:00.000Z",
      ],
      carbonIntensity: [100, 200, 300],
      isEstimated: [],
    };

    const nonLeapYear: YearCO2 = {
      zone: "DE",
      year: 2023,
      timestamps: [
        "2023-02-28T00:00:00.000Z",
        "2023-03-01T00:00:00.000Z",
      ],
      carbonIntensity: [150, 250],
      isEstimated: [],
    };

    const result = averageYears([leapYear, nonLeapYear]);
    // canonical = 2023
    // 2024 data shifted by -1:
    //   2024 Feb 28 → 2023 Feb 28 (no collision)
    //   2024 Feb 29 → 2023 Feb 28 (collision! clamped to 28)
    //   2024 Mar 01 → 2023 Mar 01 (no collision)
    // Within-year grouping for 2024: (Feb 28 + Feb 29 data) → (100+200)/2 = 150 at "2023-02-28"
    // Inner join: only timestamps present in ALL years
    //   2023 Feb 28: [150 (2024 avg), 150 (2023)] → avg 150
    //   2023 Mar 01: [300 (2024), 250 (2023)] → avg 275

    expect(result.years).toEqual([2024, 2023]); // order matches allData input order
    expect(result.timestamps).toHaveLength(2);
    expect(result.timestamps[0]).toBe("2023-02-28T00:00:00.000Z");
    expect(result.timestamps[1]).toBe("2023-03-01T00:00:00.000Z");
    expect(result.carbonIntensity[0]).toBeCloseTo(150, 6);
    expect(result.carbonIntensity[1]).toBeCloseTo(275, 6);
  });

  it("inner join drops timestamps not present in all years", () => {
    const y1: YearCO2 = {
      zone: "DE", year: 2022,
      timestamps: ["2022-01-01T00:00:00.000Z", "2022-01-01T00:05:00.000Z", "2022-01-01T00:10:00.000Z"],
      carbonIntensity: [100, 200, 300],
      isEstimated: [],
    };
    const y2: YearCO2 = {
      zone: "DE", year: 2023,
      timestamps: ["2023-01-01T00:00:00.000Z", "2023-01-01T00:10:00.000Z"],
      carbonIntensity: [110, 310],
      isEstimated: [],
    };

    const result = averageYears([y1, y2]);
    // canonical = 2022
    // y1 stays same, y2 shifts by -1
    // y2 -> [2022-01-01T00:00:00.000Z, 2022-01-01T00:10:00.000Z]
    // Inner join: only timestamps in both years → 00:00 and 00:10
    // 00:05 is dropped because y2 doesn't have it

    expect(result.timestamps).toHaveLength(2);
    expect(result.timestamps[0]).toBe("2022-01-01T00:00:00.000Z");
    expect(result.timestamps[1]).toBe("2022-01-01T00:10:00.000Z");
    expect(result.carbonIntensity[0]).toBeCloseTo(105, 6);  // (100+110)/2
    expect(result.carbonIntensity[1]).toBeCloseTo(305, 6);  // (300+310)/2
  });

  it("three years with matching data", () => {
    const y1 = makeYear("DE", 2022, 6, 0);
    const y2 = makeYear("DE", 2023, 6, 20);
    const y3 = makeYear("DE", 2024, 6, 40);
    const result = averageYears([y1, y2, y3]);

    expect(result.years).toEqual([2022, 2023, 2024]);
    expect(result.timestamps.length).toBe(6);

    for (let i = 0; i < 6; i++) {
      const expected = (y1.carbonIntensity[i] + y2.carbonIntensity[i] + y3.carbonIntensity[i]) / 3;
      expect(result.carbonIntensity[i]).toBeCloseTo(expected, 6);
    }
  });

  it("throws on empty input", () => {
    expect(() => averageYears([])).toThrow("No CO2 data");
  });
});
