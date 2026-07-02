import type { YearCO2, CO2Timeline } from "../domain/types";

/**
 * Shift an ISO timestamp string by a year offset.
 * Handles Feb 29 -> Feb 28 when shifting to a non-leap year.
 */
function shiftTimestamp(iso: string, shift: number): string {
  const dt = new Date(iso);
  const year = dt.getUTCFullYear() + shift;
  const month = dt.getUTCMonth();
  const day = dt.getUTCDate();
  const hours = dt.getUTCHours();
  const minutes = dt.getUTCMinutes();
  const seconds = dt.getUTCSeconds();
  const ms = dt.getUTCMilliseconds();

  const d = new Date(Date.UTC(year, month, day, hours, minutes, seconds, ms));
  if (d.getUTCMonth() !== month || d.getUTCDate() !== day) {
    const clamped = new Date(Date.UTC(year, month, 28, hours, minutes, seconds, ms));
    return clamped.toISOString();
  }
  return d.toISOString();
}

/**
 * Average multiple years of CO2 data into a single timeline.
 *
 * Port of Python's `build_timeline` in preprocess_data.py:
 *   1. Shift all timestamps to canonical (min) year
 *   2. Handle Feb 29 → Feb 28 collisions
 *   3. Average within-year duplicate timestamps (leap-year resolution)
 *   4. Inner join — keep only timestamps with data in ALL years
 *   5. Average across years
 */
export function averageYears(allData: YearCO2[]): CO2Timeline {
  if (allData.length === 0) {
    throw new Error("No CO2 data provided");
  }

  if (allData.length === 1) {
    const d = allData[0];
    return {
      zone: d.zone,
      years: [d.year],
      timestamps: [...d.timestamps],
      carbonIntensity: [...d.carbonIntensity],
    };
  }

  const zone = allData[0].zone;
  const years = allData.map((d) => d.year);
  const canonicalYear = Math.min(...years);

  // Step 1 & 2: shift all timestamps to canonical year, group by (year, shifted_ts)
  const byYearTs = new Map<string, number[]>();

  for (const data of allData) {
    const shift = canonicalYear - data.year;
    for (let i = 0; i < data.timestamps.length; i++) {
      const shifted = shiftTimestamp(data.timestamps[i], shift);
      const key = `${data.year}|${shifted}`;
      const arr = byYearTs.get(key);
      if (arr) {
        arr.push(data.carbonIntensity[i]);
      } else {
        byYearTs.set(key, [data.carbonIntensity[i]]);
      }
    }
  }

  // Step 3: average within-year duplicates, group by shifted timestamp
  const groups = new Map<string, number[]>();

  for (const [key, vals] of byYearTs) {
    const pipeIdx = key.indexOf("|");
    const shiftedTs = key.slice(pipeIdx + 1);
    const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
    const arr = groups.get(shiftedTs);
    if (arr) {
      arr.push(avg);
    } else {
      groups.set(shiftedTs, [avg]);
    }
  }

  // Step 4 & 5: inner join — keep only timestamps with data from ALL years
  const nYears = allData.length;
  const sortedTs = [...groups.keys()].sort();
  const timestamps: string[] = [];
  const carbonIntensity: number[] = [];

  for (const ts of sortedTs) {
    const vals = groups.get(ts)!;
    if (vals.length === nYears) {
      timestamps.push(ts);
      carbonIntensity.push(vals.reduce((a, b) => a + b, 0) / nYears);
    }
  }

  return { zone, years, timestamps, carbonIntensity };
}
