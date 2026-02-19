
import { GoogleGenAI } from "@google/genai";
import { AnalysisResult } from "../types";

export class AnalysisError extends Error {
  constructor(public message: string, public type: 'AUTH' | 'SAFETY' | 'UNKNOWN') {
    super(message);
    this.name = 'AnalysisError';
  }
}

const extractJson = (text: string) => {
  try {
    const start = text.indexOf('{');
    const end = text.lastIndexOf('}');
    if (start !== -1 && end !== -1) {
      return JSON.parse(text.substring(start, end + 1));
    }
    return JSON.parse(text);
  } catch (e) {
    throw new Error("Neural output parsing failed.");
  }
};

export const analyzeNews = async (newsText: string): Promise<AnalysisResult> => {
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  
  try {
    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: `Audit this news claim for veracity.
      
      INPUT: "${newsText}"
      
      CRITICAL:
      1. Use Search Grounding to cross-reference facts.
      2. Analyze for bias, sensationalism, and logic.
      3. Return ONLY valid JSON.
      
      JSON SCHEMA:
      {
        "verdict": "REAL" | "FAKE" | "MISLEADING" | "UNVERIFIED",
        "confidence": number (0-100),
        "explanation": "Summary (15 words max)",
        "keyPoints": ["3 short bullet points"],
        "bias": number,
        "sensationalism": number,
        "logicalConsistency": number,
        "sourceVerification": [{ "uri": "string", "verified": boolean }]
      }`,
      config: {
        tools: [{ googleSearch: {} }],
        temperature: 0.1,
      },
    });

    if (!response.text) throw new AnalysisError("Safety block triggered.", 'SAFETY');

    const data = extractJson(response.text);
    const grounded = response.candidates?.[0]?.groundingMetadata?.groundingChunks || [];
    
    const sources = grounded.map((chunk: any) => ({
      title: chunk.web?.title || "Verification Node",
      uri: chunk.web?.uri || "#",
      verified: data.sourceVerification?.find((v: any) => v.uri === chunk.web?.uri)?.verified ?? true
    })).filter((s: any) => s.uri !== "#");

    return {
      verdict: data.verdict || 'UNVERIFIED',
      confidence: data.confidence || 0,
      explanation: data.explanation || "No summary provided.",
      keyPoints: data.keyPoints || [],
      sources: sources,
      categories: {
        bias: data.bias || 0,
        sensationalism: data.sensationalism || 0,
        logicalConsistency: data.logicalConsistency || 0,
      },
    };
  } catch (error: any) {
    if (error.message?.includes('API_KEY')) throw new AnalysisError("Invalid API Key", 'AUTH');
    throw new AnalysisError(error.message || "System error", 'UNKNOWN');
  }
};
