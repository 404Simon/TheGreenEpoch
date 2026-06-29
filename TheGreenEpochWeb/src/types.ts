export interface Constants {
  gpu_power_train: number;
  gpu_power_pause: number;
  pue: number;
  checkpoint_pause_time: number;
  checkpoint_resume_time: number;
}

export interface TrainingProfile {
  name: string;
  modelParams: number;
  datasetTokens: number;
  gpuCount: number;
}

export interface FullProfile {
  name: string;
  modelParams: number;
  datasetTokens: number;
  gpuCount: number;
  gpuPowerTrain: number;
  gpuPowerPause: number;
  pue: number;
  checkpointPauseTime: number;
  checkpointResumeTime: number;
}

export interface Scenario {
  id: string;
  description: string;
  model: string;
  thresholds: number[];
  hysteresis: number[];
  region: string;
  startTimes: string[];
  historicalYears: number[];
  overheadBudgetPct: number;
}

export interface SimConfig {
  scenarioDescription: string;
  region: string;
  historicalYears: number[];
  startTime: string;
  thetaPause: number;
  thetaResume: number;
  overheadBudgetPct: number;
}

export interface CO2Timeline {
  zone: string;
  years: number[];
  timestamps: string[];
  carbonIntensity: number[];
}

export enum SimState {
  RUNNING = "running",
  PAUSED = "paused",
}

export interface SimProgress {
  timestamp: string;
  carbonIntensity: number;
  state: SimState;
  tokensRemaining: number;
  tokensTotal: number;
  totalWallS: number;
  trainingS: number;
  pausedS: number;
  checkpointS: number;
  totalEnergyWh: number;
  trainingEnergyWh: number;
  pausedEnergyWh: number;
  checkpointEnergyWh: number;
  totalEmissionsG: number;
  numPauses: number;
  done: boolean;
  stopReason: string;
  issues: string[];
  nanFallbacks: number;
}

export interface SimResult {
  id: string;
  scenarioDescription: string;
  model: string;
  region: string;
  historicalYears: number[];
  startTime: string;
  threshold: number;
  hysteresisMargin: number;
  totalWallTimeH: number;
  trainingTimeH: number;
  pausedTimeH: number;
  checkpointOverheadH: number;
  totalEnergyKwh: number;
  trainingEnergyKwh: number;
  pausedEnergyKwh: number;
  checkpointEnergyKwh: number;
  totalEmissionsKgco2: number;
  tokensProcessed: number;
  tokensTotal: number;
  completed: boolean;
  numPauses: number;
  overheadBudgetPct: number;
  actualOverheadPct: number;
  withinOverheadBudget: boolean;
  timestamps: string[];
  carbonIntensitySeries: number[];
  stateSeries: string[];
  emissionsSeries: number[];
  tokensRemainingSeries: number[];
  issues: string[];
  stopReason: string;
  baselineEmissionsKgco2: number;
  baselineTimeH: number;
  co2SavingsPct: number;
  score: number;
  idleTimeH: number;
  completionPct: number;
  ok: boolean;
}
