
import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ShieldCheck, Key, Terminal } from 'lucide-react';
import InputForm from './components/InputForm';
import ResultsCard from './components/ResultsCard';
import { analyzeNews } from './services/geminiService';
import { AnalysisResult } from './types';

const App: React.FC = () => {
  const [hasKey, setHasKey] = useState<boolean | null>(null);
  const [status, setStatus] = useState<'idle' | 'analyzing' | 'result'>('idle');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const checkKey = async () => {
      const aistudio = (window as any).aistudio;
      if (aistudio) {
        setHasKey(await aistudio.hasSelectedApiKey());
      } else {
        setHasKey(true);
      }
    };
    checkKey();
  }, []);

  const handleAnalyze = async (text: string) => {
    setStatus('analyzing');
    setError(null);
    try {
      const data = await analyzeNews(text);
      setResult(data);
      setStatus('result');
    } catch (err: any) {
      setError(err.message || 'Verification failed');
      setStatus('idle');
    }
  };

  const reset = () => {
    setResult(null);
    setStatus('idle');
    setError(null);
  };

  if (hasKey === false) {
    return (
      <div className="min-h-screen bg-charcoal flex items-center justify-center p-6 text-center">
        <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} className="glass rounded-[2.5rem] p-12 max-w-md space-y-8">
          <Terminal className="w-12 h-12 text-indigo-400 mx-auto" />
          <div className="space-y-4">
            <h2 className="text-2xl font-display font-black uppercase">System Locked</h2>
            <p className="text-zinc-500 text-sm">Real-time grounding requires an active Neural Link. Please select a paid API key.</p>
          </div>
          <button onClick={() => (window as any).aistudio.openSelectKey().then(() => setHasKey(true))} className="w-full py-4 bg-white text-black rounded-2xl font-black text-xs uppercase tracking-widest shadow-xl">
            Connect Neural Key
          </button>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-charcoal text-white selection:bg-indigo-500/30">
      <header className="fixed top-0 w-full p-6 flex justify-between items-center z-50 bg-charcoal/40 backdrop-blur-xl border-b border-white/5">
        <div className="flex items-center gap-2 cursor-pointer" onClick={reset}>
          <ShieldCheck className="w-5 h-5 text-indigo-400" />
          <span className="font-display font-bold text-xs tracking-[0.4em] uppercase">TruthGuard</span>
        </div>
        <div className="flex items-center gap-6">
           <div className="text-[9px] font-black uppercase tracking-widest text-zinc-500 flex items-center gap-2">
            <Key className="w-3 h-3 text-indigo-400" /> Active Protocol 3.0
          </div>
        </div>
      </header>

      <main className="relative z-10 pt-32 pb-20 px-6">
        <AnimatePresence mode="wait">
          {status === 'idle' && (
            <motion.div key="idle" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} className="max-w-4xl mx-auto space-y-16 text-center">
              <div className="space-y-6">
                <h1 className="text-5xl md:text-8xl font-display font-extrabold tracking-tighter gradient-text leading-[1.1]">
                  Verified <br /> Intelligence.
                </h1>
                <p className="text-zinc-500 text-lg md:text-xl font-medium max-w-2xl mx-auto">
                  Cross-reference claims against global news nodes using <br /> deep-grounded neural analysis.
                </p>
              </div>
              <InputForm onAnalyze={handleAnalyze} error={error} isLoading={false} />
            </motion.div>
          )}

          {status === 'analyzing' && (
            <motion.div key="analyzing" className="flex flex-col items-center justify-center py-40 space-y-8">
              <div className="w-20 h-20 rounded-full border border-indigo-500/10 border-t-indigo-500 animate-spin" />
              <div className="text-center space-y-2">
                <h3 className="text-[10px] font-black uppercase tracking-[0.5em]">Neural Grounding</h3>
                <p className="text-[8px] text-zinc-600 font-bold uppercase tracking-widest">Scanning verification nodes...</p>
              </div>
            </motion.div>
          )}

          {status === 'result' && result && (
            <motion.div key="result" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <ResultsCard result={result} onReset={reset} />
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      <footer className="fixed bottom-0 w-full p-8 flex justify-between items-center opacity-20 pointer-events-none">
        <span className="text-[8px] font-black uppercase tracking-widest">TruthGuard AI Engine</span>
        <span className="text-[8px] font-black uppercase tracking-widest">© 2025 Neural Systems</span>
      </footer>
    </div>
  );
};

export default App;
