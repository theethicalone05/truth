
import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { ArrowRight, AlertCircle } from 'lucide-react';

interface InputFormProps {
  onAnalyze: (text: string) => void;
  error: string | null;
  isLoading: boolean;
}

const InputForm: React.FC<InputFormProps> = ({ onAnalyze, error, isLoading }) => {
  const [text, setText] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (text.trim().length < 10 || isLoading) return;
    onAnalyze(text);
  };

  return (
    <div className="w-full max-w-2xl mx-auto space-y-6">
      <motion.form 
        onSubmit={handleSubmit}
        className="glass rounded-[2rem] p-4 transition-all duration-500 focus-within:ring-2 focus-within:ring-indigo-500/20"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste an article, social post, or specific claim to verify..."
          className="w-full h-44 bg-transparent border-none text-white placeholder:text-zinc-700 focus:ring-0 p-6 text-lg font-medium resize-none leading-relaxed"
          disabled={isLoading}
        />
        <div className="flex items-center justify-between p-2 pl-6">
          <span className="text-[10px] font-black tracking-widest uppercase text-zinc-600">
            {text.length} characters
          </span>
          <button
            type="submit"
            disabled={text.trim().length < 10 || isLoading}
            className={`px-8 py-3 rounded-2xl font-bold flex items-center gap-3 transition-all active:scale-95 ${
              text.trim().length < 10 || isLoading
                ? 'bg-zinc-900 text-zinc-700 cursor-not-allowed border border-white/5'
                : 'bg-white text-black hover:bg-zinc-200'
            }`}
          >
            <span className="text-[10px] uppercase tracking-widest font-black">Analyze</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </motion.form>

      {error && (
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex items-center gap-2 justify-center text-rose-500 text-[10px] font-black uppercase tracking-widest"
        >
          <AlertCircle className="w-4 h-4" />
          {error}
        </motion.div>
      )}
    </div>
  );
};

export default InputForm;
