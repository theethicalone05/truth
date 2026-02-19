
import React, { useEffect, useState } from 'react';
import { motion, animate } from 'framer-motion';
import { 
  RefreshCcw, ExternalLink, Zap, Search, Globe, 
  CheckCircle2, HelpCircle, ShieldCheck 
} from 'lucide-react';
import { AnalysisResult } from '../types';

interface ResultsCardProps {
  result: AnalysisResult;
  onReset: () => void;
}

const NumberCounter = ({ value }: { value: number }) => {
  const [display, setDisplay] = useState(0);
  useEffect(() => {
    const controls = animate(0, value, {
      duration: 2,
      ease: [0.16, 1, 0.3, 1],
      onUpdate: (latest) => setDisplay(Math.round(latest)),
    });
    return () => controls.stop();
  }, [value]);
  return <span>{display}</span>;
};

const ResultsCard: React.FC<ResultsCardProps> = ({ result, onReset }) => {
  const meta = {
    REAL: { color: '#10b981', glow: 'rgba(16, 185, 129, 0.15)' },
    FAKE: { color: '#f43f5e', glow: 'rgba(244, 63, 94, 0.15)' },
    MISLEADING: { color: '#f59e0b', glow: 'rgba(245, 158, 11, 0.15)' },
    UNVERIFIED: { color: '#71717a', glow: 'rgba(113, 113, 122, 0.15)' },
  }[result.verdict];

  const radius = 42;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (result.confidence / 100) * circumference;

  const getFavicon = (uri: string) => {
    try { return `https://www.google.com/s2/favicons?domain=${new URL(uri).hostname}&sz=64`; }
    catch { return null; }
  };

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-8 duration-1000 max-w-5xl mx-auto">
      {/* Header Verdict Card */}
      <div className="glass rounded-[2.5rem] p-8 md:p-12 relative overflow-hidden border-white/5 shadow-2xl">
        <div 
          className="absolute top-0 right-0 w-[500px] h-[500px] rounded-full -translate-y-1/2 translate-x-1/2 blur-[120px] pointer-events-none"
          style={{ backgroundColor: meta.glow }}
        />
        
        <div className="flex flex-col md:flex-row items-center gap-12 relative z-10">
          <div className="relative w-36 h-36 md:w-44 md:h-44 flex items-center justify-center shrink-0">
            <svg viewBox="0 0 100 100" className="w-full h-full transform -rotate-90">
              <circle cx="50" cy="50" r={radius} fill="none" stroke="rgba(255,255,255,0.03)" strokeWidth="4" />
              <motion.circle
                cx="50" cy="50" r={radius} fill="none" stroke={meta.color} strokeWidth="4"
                strokeDasharray={circumference}
                initial={{ strokeDashoffset: circumference }}
                animate={{ strokeDashoffset: offset }}
                transition={{ duration: 2.2, ease: [0.16, 1, 0.3, 1] }}
                strokeLinecap="round"
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <div className="text-3xl md:text-5xl font-display font-black text-white tracking-tighter">
                <NumberCounter value={result.confidence} />%
              </div>
              <span className="text-[7px] font-black text-zinc-500 uppercase tracking-[0.4em] mt-2">Accuracy</span>
            </div>
          </div>

          <div className="flex-1 text-center md:text-left">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/[0.03] border border-white/5 mb-4">
              <div className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ backgroundColor: meta.color }} />
              <span className="text-[8px] font-black text-zinc-500 uppercase tracking-widest">Analysis Result</span>
            </div>
            <h2 className="text-5xl md:text-8xl font-display font-black tracking-tighter leading-none mb-4" style={{ color: meta.color }}>
              {result.verdict}
            </h2>
            <p className="text-zinc-400 text-sm font-medium opacity-80 max-w-md">
              Evaluated with {result.confidence}% precision across global verification nodes.
            </p>
          </div>

          <button onClick={onReset} className="group p-4 glass rounded-2xl hover:bg-white hover:text-black transition-all">
            <RefreshCcw className="w-5 h-5 group-hover:rotate-180 transition-transform duration-700" />
          </button>
        </div>
      </div>

      {/* Grid Content */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        <div className="lg:col-span-7 glass rounded-[2.5rem] p-8 space-y-6">
          <div className="space-y-2">
            <h4 className="text-[8px] font-black text-zinc-600 uppercase tracking-[0.5em] flex items-center gap-2">
              <Zap className="w-3 h-3 text-indigo-400" /> Executive Brief
            </h4>
            <p className="text-xl md:text-2xl font-bold leading-tight text-white">{result.explanation}</p>
          </div>
          <div className="space-y-3">
            {result.keyPoints.map((p, i) => (
              <div key={i} className="px-5 py-4 rounded-2xl bg-white/[0.01] border border-white/[0.03] flex gap-4 items-start">
                <span className="text-[10px] font-black text-zinc-700 pt-1">0{i+1}</span>
                <p className="text-sm font-medium text-zinc-400 leading-relaxed">{p}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="lg:col-span-5 glass rounded-[2.5rem] p-8 space-y-6">
          <h4 className="text-[8px] font-black text-zinc-600 uppercase tracking-[0.5em] flex items-center gap-2">
            <Search className="w-3 h-3 text-indigo-400" /> Grounding Assets
          </h4>
          <div className="space-y-3 max-h-[300px] overflow-y-auto pr-2 custom-scrollbar">
            {result.sources.length > 0 ? result.sources.map((s, i) => (
              <a key={i} href={s.uri} target="_blank" rel="noreferrer" className="flex items-center gap-4 p-3 glass rounded-2xl hover:bg-white/[0.05] transition-all group">
                <div className="w-10 h-10 rounded-xl bg-black flex items-center justify-center shrink-0 border border-white/5">
                  <img src={getFavicon(s.uri) || ''} className="w-5 h-5 object-contain" alt="" onError={(e) => (e.currentTarget.style.display='none')} />
                  <Globe className="w-5 h-5 text-zinc-700 absolute" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-1.5">
                    <span className="text-[10px] font-bold text-zinc-300 truncate uppercase tracking-tight group-hover:text-indigo-400">{s.title}</span>
                    {s.verified ? <CheckCircle2 className="w-3 h-3 text-emerald-500" /> : <HelpCircle className="w-3 h-3 text-zinc-600" />}
                  </div>
                  <span className="text-[8px] font-mono text-zinc-700 truncate block">{new URL(s.uri).hostname}</span>
                </div>
                <ExternalLink className="w-3 h-3 text-zinc-800" />
              </a>
            )) : (
              <div className="py-12 text-center opacity-20 flex flex-col items-center gap-4 border border-dashed border-white/10 rounded-[2rem]">
                <ShieldCheck className="w-8 h-8" />
                <p className="text-[9px] font-black uppercase tracking-widest">Neural Verification Only</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResultsCard;
