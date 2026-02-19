
export interface AnalysisResult {
  verdict: 'REAL' | 'FAKE' | 'MISLEADING' | 'UNVERIFIED';
  confidence: number;
  explanation: string;
  keyPoints: string[];
  sources: { title: string; uri: string; verified: boolean }[];
  categories: {
    bias: number;
    sensationalism: number;
    logicalConsistency: number;
  };
}
