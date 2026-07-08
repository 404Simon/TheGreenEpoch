export function round2(n: number): number {
  return Math.round(n * 100) / 100;
}

export function stripProxies<T>(data: T): T {
  try {
    return structuredClone(data);
  } catch {
    return JSON.parse(JSON.stringify(data));
  }
}
