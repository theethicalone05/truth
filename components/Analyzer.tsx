
import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { ArrowRight, AlertCircle } from 'lucide-react';

interface AnalyzerProps {
  onAnalyze: (text: string) => void;
  error: string | null;
}

const Analyzer: React.FC<AnalyzerProps> = ({ onAnalyze, error }) => {
  const [text, setText] = useState('');

  const handleSubmit = () => {
    if (text.trim().length < 10) return;
    onAnalyze(text);
  };

  return (
    <div className="w-full max-w-2xl mx-auto space-y-6">
      <motion.div 
        className="glass glow rounded-[2rem] p-4 group transition-all duration-500 focus-within:ring-1 focus-within:ring-white/20"
        whileHover={{ scale: 1.01 }}
      >
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste a news article, post, or claim..."
          className="w-full h-40 bg-transparent border-none text-white placeholder:text-zinc-700 focus:ring-0 p-6 text-lg font-medium resize-none leading-relaxed"
        />
        <div className="flex items-center justify-between p-2 pl-6">
          <span className={`text-[10px] font-bold tracking-widest uppercase transition-colors ${text.length > 0 ? 'text-zinc-500' : 'text-zinc-800'}`}>
            {text.length} characters
          </span>
          <button
            onClick={handleSubmit}
            disabled={text.trim().length < 10}
            className={`px-8 py-3 rounded-2xl font-bold flex items-center gap-3 transition-all active:scale-[0.98] ${
              text.trim().length < 10
                ? 'bg-zinc-900 text-zinc-700 cursor-not-allowed border border-white/5'
                : 'bg-white text-black hover:shadow-[0_0_20px_rgba(255,255,255,0.2)]'
            }`}
          >
            <span className="text-xs uppercase tracking-widest">Analyze Truth</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </motion.div>

      {error && (
        <motion.div 
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center gap-3 justify-center text-rose-500 text-xs font-bold uppercase tracking-widest"
        >
          <AlertCircle className="w-4 h-4" />
          {error}
        </motion.div>
      )}
    </div>
  );
};

export default Analyzer;
