
import React from 'react';
import { 
  CheckCircle2, 
  AlertTriangle, 
  ShieldAlert, 
  HelpCircle, 
  ExternalLink, 
  ChevronLeft,
  Activity,
  Zap,
  Fingerprint,
  Link2,
  Trophy,
  Target
} from 'lucide-react';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, ResponsiveContainer } from 'recharts';
import { AnalysisResult } from '../types';

interface ResultsViewProps {
  result: AnalysisResult;
  onReset: () => void;
}

const ResultsView: React.FC<ResultsViewProps> = ({ result, onReset }) => {
  const styles = (() => {
    switch (result.verdict) {
      case 'REAL': return { color: 'text-emerald-400', bg: 'bg-emerald-500/5', border: 'border-emerald-500/10', icon: CheckCircle2, fill: '#10b981' };
      case 'FAKE': return { color: 'text-rose-500', bg: 'bg-rose-500/5', border: 'border-rose-500/10', icon: ShieldAlert, fill: '#f43f5e' };
      case 'MISLEADING': return { color: 'text-amber-500', bg: 'bg-amber-500/5', border: 'border-amber-500/10', icon: AlertTriangle, fill: '#f59e0b' };
      default: return { color: 'text-neutral-400', bg: 'bg-neutral-500/5', border: 'border-neutral-500/10', icon: HelpCircle, fill: '#64748b' };
    }
  })();

  const Icon = styles.icon;

  const chartData = [
    { subject: 'BIAS', A: result.categories.bias },
    { subject: 'SENSATION', A: result.categories.sensationalism },
    { subject: 'LOGIC', A: result.categories.logicalConsistency },
  ];

  return (
    <div className="w-full max-w-7xl mx-auto space-y-12 animate-reveal">
      {/* Top Header Controls */}
      <div className="flex items-center justify-between border-b border-white/[0.05] pb-8">
        <button 
          onClick={onReset}
          className="group flex items-center gap-3 text-neutral-500 hover:text-white transition-all text-[10px] font-black uppercase tracking-[0.4em]"
        >
          <ChevronLeft className="h-4 w-4 group-hover:-translate-x-1 transition-transform" />
          Close Report
        </button>
        <div className="flex items-center gap-6">
          <div className="hidden lg:flex flex-col items-end">
             <span className="text-[9px] font-black text-neutral-600 uppercase tracking-widest">Global Timestamp</span>
             <span className="text-[10px] font-mono text-neutral-400">{new Date().toISOString()}</span>
          </div>
          <div className="h-8 w-px bg-white/10 hidden lg:block"></div>
          <div className="flex items-center gap-3 bg-emerald-500/10 px-4 py-1.5 rounded-full border border-emerald-500/20">
             <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
             <span className="text-[9px] font-black text-emerald-500 uppercase tracking-widest">Identity Match 1.0</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-10">
        {/* Main Result Column */}
        <div className="xl:col-span-8 space-y-8">
          <div className="premium-card rounded-[2.5rem] p-10 md:p-14 relative overflow-hidden shadow-2xl">
            {/* Visual Decoration */}
            <div className={`absolute -top-32 -right-32 w-96 h-96 blur-[120px] opacity-10 rounded-full ${styles.bg}`}></div>
            
            <div className="flex flex-col lg:flex-row lg:items-start justify-between gap-12 mb-20">
              <div className="space-y-8 flex-1">
                <div className={`inline-flex items-center gap-3 px-4 py-2 rounded-full ${styles.bg} ${styles.border} text-[10px] font-black uppercase tracking-widest ${styles.color}`}>
                  <Target className="h-3.5 w-3.5" />
                  Primary Verdict Analysis
                </div>
                <div className="flex items-center gap-10">
                  <div className={`p-6 rounded-[2rem] ${styles.bg} ${styles.border} shadow-lg`}>
                    <Icon className={`h-16 w-16 md:h-20 md:w-20 ${styles.color}`} />
                  </div>
                  <div>
                    <h2 className={`text-6xl md:text-9xl font-black tracking-tighter leading-none ${styles.color} mb-4`}>{result.verdict}</h2>
                    <div className="flex flex-wrap gap-4">
                      <p className="flex items-center gap-2 text-neutral-500 font-mono text-[10px] uppercase tracking-widest">
                        <Fingerprint className="h-4 w-4" />
                        Confidence Score: {result.confidence}%
                      </p>
                      <span className="text-neutral-800">•</span>
                      <p className="flex items-center gap-2 text-neutral-500 font-mono text-[10px] uppercase tracking-widest">
                        ID: TRUTH-{Math.floor(Math.random()*10000)}
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="flex flex-col items-center justify-center p-10 premium-card rounded-[2.5rem] min-w-[200px] shadow-inner border-white/5 bg-white/[0.01]">
                <Trophy className="h-6 w-6 text-neutral-700 mb-4" />
                <span className="text-[10px] font-black text-neutral-600 uppercase tracking-widest mb-1">Precision</span>
                <span className="text-6xl font-mono font-bold text-white tracking-tighter">{result.confidence}<span className="text-xl text-neutral-600 ml-1">%</span></span>
                <div className="w-full h-1.5 bg-white/5 rounded-full mt-8 overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-blue-600 to-indigo-500 transition-all duration-1000 ease-out" 
                    style={{ width: `${result.confidence}%` }}
                  ></div>
                </div>
              </div>
            </div>

            <div className="space-y-16">
              <div className="space-y-6">
                <div className="flex items-center gap-3 text-[11px] font-black text-blue-500 uppercase tracking-[0.4em]">
                  <Zap className="h-4 w-4" />
                  Executive Intelligence Summary
                </div>
                <p className="text-3xl md:text-4xl text-white leading-[1.15] font-bold tracking-tight">
                  {result.explanation}
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {result.keyPoints.map((point, i) => (
                  <div key={i} className="group p-8 bg-white/[0.01] rounded-[2rem] border border-white/[0.05] flex gap-6 hover:bg-white/[0.03] hover:border-white/[0.1] transition-all duration-300">
                    <span className="text-neutral-700 font-mono text-sm font-bold pt-1">0{i+1}</span>
                    <p className="text-sm text-neutral-400 leading-relaxed font-semibold uppercase tracking-wide group-hover:text-neutral-200 transition-colors">{point}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Sidebar Metrics & Grounding Column */}
        <div className="xl:col-span-4 space-y-8">
          {/* Linguistic Profile: Fixed visibility for PC mode */}
          <div className="premium-card rounded-[2.5rem] p-10 flex flex-col shadow-xl">
            <div className="flex items-center justify-between mb-12">
              <span className="text-[10px] font-black text-neutral-500 uppercase tracking-[0.4em]">Linguistic Profile</span>
              <Activity className="h-5 w-5 text-blue-500" />
            </div>

            <div className="h-72 sm:h-80 w-full mb-12">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart cx="50%" cy="50%" outerRadius="75%" data={chartData}>
                  <PolarGrid stroke="rgba(255,255,255,0.06)" strokeDasharray="4 4" />
                  <PolarAngleAxis 
                    dataKey="subject" 
                    tick={{ fill: '#525252', fontSize: 10, fontWeight: 800, letterSpacing: '0.15em' }} 
                  />
                  <Radar
                    dataKey="A"
                    stroke={styles.fill}
                    fill={styles.fill}
                    fillOpacity={0.08}
                    animationDuration={1500}
                    animationBegin={300}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>

            <div className="space-y-10">
              {[
                { label: 'Bias Intensity', val: result.categories.bias, color: 'bg-blue-600' },
                { label: 'Sensationalism', val: result.categories.sensationalism, color: 'bg-indigo-600' },
                { label: 'Logic Consistency', val: result.categories.logicalConsistency, color: 'bg-emerald-600' }
              ].map((m, i) => (
                <div key={i} className="space-y-4">
                  <div className="flex justify-between text-[10px] font-bold uppercase tracking-[0.2em]">
                    <span className="text-neutral-500">{m.label}</span>
                    <span className="text-neutral-300 font-mono">{m.val}%</span>
                  </div>
                  <div className="h-1.5 w-full bg-white/[0.04] rounded-full overflow-hidden">
                    <div 
                      className={`h-full ${m.color} transition-all duration-1000 ease-out shadow-lg`} 
                      style={{ width: `${m.val}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Verification Network Grounding */}
          <div className="premium-card rounded-[2.5rem] p-10 flex flex-col flex-grow shadow-xl">
            <div className="flex items-center justify-between mb-8">
              <span className="text-[10px] font-black text-neutral-500 uppercase tracking-[0.4em]">Grounding Assets</span>
              <Link2 className="h-5 w-5 text-neutral-700" />
            </div>
            
            <div className="space-y-4">
              {result.sources.length > 0 ? result.sources.map((s, i) => (
                <a 
                  key={i} 
                  href={s.uri} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="group flex items-center justify-between p-5 bg-white/[0.01] hover:bg-white/[0.04] border border-white/[0.04] rounded-2xl transition-all duration-300"
                >
                  <div className="flex flex-col gap-1 min-w-0">
                    <span className="text-xs text-neutral-300 font-bold truncate pr-4 group-hover:text-blue-400 transition-colors uppercase tracking-tight">
                      {s.title}
                    </span>
                    <span className="text-[9px] text-neutral-600 font-mono truncate">{s.uri.slice(0, 40)}...</span>
                  </div>
                  <ExternalLink className="h-3.5 w-3.5 text-neutral-800 group-hover:text-blue-500 transition-colors shrink-0" />
                </a>
              )) : (
                <div className="flex flex-col items-center justify-center py-16 opacity-30 space-y-4 border border-dashed border-white/5 rounded-3xl">
                  <Zap className="h-8 w-8 text-neutral-600" />
                  <p className="text-[10px] font-black uppercase tracking-[0.3em]">Neural Verification only</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResultsView;
