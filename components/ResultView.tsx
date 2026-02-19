
import React, { useEffect, useState } from 'react';
import { motion, animate } from 'framer-motion';
import { RefreshCcw, ExternalLink, Zap, Search, Globe, ShieldCheck, CheckCircle2, AlertCircle, HelpCircle } from 'lucide-react';
import { AnalysisResult } from '../types';

interface ResultViewProps {
  result: AnalysisResult;
  onReset: () => void;
}

const NumberCounter = ({ value }: { value: number }) => {
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    const controls = animate(0, value, {
      duration: 2,
      ease: [0.16, 1, 0.3, 1],
      onUpdate: (latest) => setDisplayValue(Math.round(latest)),
    });
    return () => controls.stop();
  }, [value]);

  return <span>{displayValue}</span>;
};

const ResultView: React.FC<ResultViewProps> = ({ result, onReset }) => {
  const meta = {
    REAL: { color: '#10b981', label: 'Verified Real', glow: 'rgba(16, 185, 129, 0.2)' },
    FAKE: { color: '#f43f5e', label: 'Verified Fake', glow: 'rgba(244, 63, 94, 0.2)' },
    MISLEADING: { color: '#f59e0b', label: 'Misleading', glow: 'rgba(245, 158, 11, 0.2)' },
    UNVERIFIED: { color: '#71717a', label: 'Unverified', glow: 'rgba(113, 113, 122, 0.2)' },
  }[result.verdict] || { color: '#71717a', label: 'Unverified', glow: 'rgba(113, 113, 122, 0.2)' };

  const radius = 42;
  const strokeWidth = 3.5;
  const circumference = 2 * Math.PI * radius;
  
  const offset = circumference - (result.confidence / 100) * circumference;

  const getFavicon = (uri: string) => {
    try {
      const url = new URL(uri);
      return `https://www.google.com/s2/favicons?domain=${url.hostname}&sz=64`;
    } catch (e) {
      return null;
    }
  };

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-8 duration-1000 max-w-5xl mx-auto">
      {/* Primary Verdict Card */}
      <div className="glass rounded-[2.5rem] p-8 md:p-12 relative overflow-hidden border-white/5 shadow-2xl">
        {/* Dynamic Background Glow */}
        <motion.div 
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 1.5, ease: "easeOut" }}
          className="absolute top-0 right-0 w-[500px] h-[500px] rounded-full -translate-y-1/2 translate-x-1/2 blur-[120px] pointer-events-none"
          style={{ backgroundColor: meta.glow }}
        />
        
        <div className="flex flex-col md:flex-row items-center justify-between w-full gap-12 relative z-10">
          {/* Refined Circular Gauge */}
          <div className="relative w-36 h-36 md:w-44 md:h-44 flex items-center justify-center shrink-0">
            <svg viewBox="0 0 100 100" className="w-full h-full transform -rotate-90 drop-shadow-[0_0_12px_rgba(0,0,0,0.5)]">
              <circle 
                cx="50" cy="50" r={radius} 
                fill="none" 
                stroke="currentColor" 
                strokeWidth={strokeWidth} 
                className="text-white/[0.03]" 
              />
              <motion.circle
                cx="50" cy="50" r={radius}
                fill="none" 
                stroke={meta.color} 
                strokeWidth={strokeWidth}
                strokeDasharray={circumference}
                initial={{ strokeDashoffset: circumference }}
                animate={{ strokeDashoffset: offset }}
                transition={{ 
                  duration: 2.2, 
                  ease: [0.16, 1, 0.3, 1],
                  delay: 0.2 
                }}
                strokeLinecap="round"
                style={{ filter: `drop-shadow(0 0 6px ${meta.color}44)` }}
              />
            </svg>
            
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <motion.div 
                initial={{ opacity: 0, scale: 0.5 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.8, delay: 0.5 }}
                className="text-3xl md:text-5xl font-display font-black text-white tracking-tighter leading-none"
              >
                <NumberCounter value={result.confidence} />
                <span className="text-sm text-zinc-600 font-bold ml-0.5">%</span>
              </motion.div>
              <span className="text-[7px] font-black text-zinc-500 uppercase tracking-[0.4em] mt-2 opacity-60">Confidence</span>
            </div>
          </div>

          <div className="flex-1 text-center md:text-left space-y-3">
            <div className="inline-flex items-center gap-2.5 px-3 py-1 rounded-full bg-white/[0.03] border border-white/5 backdrop-blur-md">
              <div className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ backgroundColor: meta.color }} />
              <span className="text-[8px] font-black text-zinc-400 uppercase tracking-widest">Neural Scan Complete</span>
            </div>
            
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8, delay: 0.3 }}
            >
              <h2 
                className="text-5xl md:text-8xl font-display font-black tracking-tighter leading-none" 
                style={{ color: meta.color }}
              >
                {result.verdict}
              </h2>
            </motion.div>

            <p className="text-zinc-400 text-xs md:text-sm font-medium tracking-tight max-w-md leading-relaxed opacity-80">
              Cross-referenced with global verification nodes. Intelligence synchronization finalized with {result.confidence}% accuracy.
            </p>
          </div>

          <div className="flex flex-col gap-3">
             <button 
              onClick={onReset}
              className="group px-8 py-3 glass rounded-2xl hover:bg-white hover:text-black transition-all duration-500 active:scale-95 flex items-center justify-center gap-3 border-white/10 shadow-xl"
            >
              <RefreshCcw className="w-4 h-4 group-hover:rotate-180 transition-transform duration-700 ease-in-out" />
              <span className="text-[10px] font-black uppercase tracking-widest">New Analysis</span>
            </button>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        <div className="lg:col-span-7 glass rounded-[2.5rem] p-8 space-y-6 border-white/5 relative overflow-hidden group">
          <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-indigo-500/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
          
          <div className="space-y-2">
            <h4 className="text-[8px] font-black text-zinc-600 uppercase tracking-[0.5em] flex items-center gap-2">
              <Zap className="w-3 h-3 text-indigo-400" />
              Intelligence Briefing
            </h4>
            <p className="text-xl md:text-2xl font-bold leading-tight text-white tracking-tight">{result.explanation}</p>
          </div>

          <div className="grid gap-3">
            {result.keyPoints.map((point, i) => (
              <motion.div 
                key={i} 
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.5 + i * 0.1 }}
                className="px-5 py-4 rounded-2xl bg-white/[0.01] border border-white/[0.03] flex gap-4 items-start hover:bg-white/[0.03] transition-colors"
              >
                <span className="text-[10px] font-black text-zinc-700 pt-1">0{i+1}</span>
                <p className="text-xs md:text-sm font-medium text-zinc-400 leading-relaxed">{point}</p>
              </motion.div>
            ))}
          </div>
        </div>

        <div className="lg:col-span-5 glass rounded-[2.5rem] p-8 space-y-6 flex flex-col border-white/5">
          <h4 className="text-[8px] font-black text-zinc-600 uppercase tracking-[0.5em] flex items-center gap-2">
            <Search className="w-3 h-3 text-indigo-400" />
            Grounding Assets
          </h4>
          
          <div className="space-y-3 flex-1 overflow-y-auto max-h-[300px] pr-2 custom-scrollbar">
            {result.sources.length > 0 ? result.sources.map((s, i) => (
              <motion.a 
                key={i} 
                href={s.uri} 
                target="_blank" 
                rel="noopener noreferrer"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.7 + i * 0.05 }}
                className={`flex items-center gap-4 p-3 bg-white/[0.01] hover:bg-white/[0.06] border border-white/5 rounded-2xl transition-all group relative ${!s.verified ? 'opacity-50 grayscale' : ''}`}
              >
                <div className="w-10 h-10 rounded-xl bg-black flex items-center justify-center overflow-hidden shrink-0 border border-white/5 group-hover:border-indigo-500/30 transition-colors">
                  {getFavicon(s.uri) ? (
                    <img src={getFavicon(s.uri)!} alt="" className="w-5 h-5 object-contain" />
                  ) : (
                    <Globe className="w-5 h-5 text-zinc-700" />
                  )}
                </div>
                <div className="flex flex-col gap-0.5 min-w-0 flex-1">
                  <div className="flex items-center gap-1.5">
                    <span className="text-[10px] font-bold text-zinc-300 truncate group-hover:text-indigo-400 transition-colors uppercase tracking-tight">
                      {s.title}
                    </span>
                    {s.verified ? (
                      <CheckCircle2 className="w-3 h-3 text-emerald-500 shrink-0" />
                    ) : (
                      <HelpCircle className="w-3 h-3 text-zinc-600 shrink-0" />
                    )}
                  </div>
                  <span className="text-[8px] font-mono text-zinc-700 truncate uppercase tracking-widest">
                    {new URL(s.uri).hostname}
                  </span>
                </div>
                <ExternalLink className="w-3 h-3 text-zinc-800 group-hover:text-white transition-colors" />
                
                {!s.verified && (
                  <div className="absolute inset-0 bg-black/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity rounded-2xl pointer-events-none">
                    <span className="text-[7px] font-black text-white uppercase tracking-[0.2em]">Verification Pending</span>
                  </div>
                )}
              </motion.a>
            )) : (
              <div className="py-12 text-center opacity-20 border border-dashed border-white/10 rounded-[2rem] flex flex-col items-center gap-4">
                 <ShieldCheck className="w-8 h-8" />
                 <p className="text-[9px] font-black uppercase tracking-[0.4em]">Proprietary Neural Verification</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResultView;
