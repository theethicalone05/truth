
import React from 'react';
import { Layers, Database, Cpu, Zap, Code2, Rocket } from 'lucide-react';

const DevHub: React.FC = () => {
  const guides = [
    {
      icon: <Cpu className="text-blue-400" />,
      title: "Model Optimization",
      content: "Use RAG (Retrieval Augmented Generation) to keep models updated with real-time news data without constant retraining."
    },
    {
      icon: <Layers className="text-purple-400" />,
      title: "Architecture",
      content: "Implement a message queue (Kafka/RabbitMQ) for batch processing high volumes of news analysis requests asynchronously."
    },
    {
      icon: <Database className="text-pink-400" />,
      title: "Scalability",
      content: "Store analysis fingerprints in a vector database (Pinecone/Milvus) to avoid re-analyzing the same viral fake news."
    }
  ];

  return (
    <section className="py-24 relative overflow-hidden">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl font-black text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400 mb-4">
            HACKATHON DEV HUB
          </h2>
          <p className="text-slate-400 max-w-2xl mx-auto">
            Ready to scale? Here is the blueprint for turning this prototype into a production-grade misinformation platform.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {guides.map((item, idx) => (
            <div key={idx} className="glass p-8 rounded-3xl border border-slate-800 hover:border-slate-700 transition-all group">
              <div className="p-3 bg-slate-800/50 rounded-2xl w-fit mb-6 group-hover:scale-110 transition-transform">
                {item.icon}
              </div>
              <h3 className="text-xl font-bold text-slate-100 mb-4">{item.title}</h3>
              <p className="text-slate-400 text-sm leading-relaxed">{item.content}</p>
            </div>
          ))}
        </div>

        <div className="mt-16 glass rounded-[2.5rem] p-12 border border-blue-500/20 bg-blue-500/5">
          <div className="flex flex-col lg:flex-row items-center gap-12">
            <div className="flex-1 space-y-6">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/20 text-blue-400 text-xs font-bold uppercase">
                <Rocket className="h-3 w-3" />
                Roadmap to production
              </div>
              <h3 className="text-4xl font-black text-slate-100">Strengthening System Architecture</h3>
              <ul className="space-y-4">
                {[
                  "Integrate OAuth for verified journalist accounts",
                  "Deploy microservices for separate bias vs fact-check logic",
                  "Enable browser extension for real-time social media overlays",
                  "Implement Trust-Rank scores for news domains"
                ].map((point, i) => (
                  <li key={i} className="flex items-center gap-3 text-slate-400">
                    <div className="h-1.5 w-1.5 bg-blue-500 rounded-full"></div>
                    {point}
                  </li>
                ))}
              </ul>
            </div>
            <div className="flex-1 w-full lg:w-auto relative">
               <div className="p-8 bg-slate-900 rounded-3xl border border-slate-800 shadow-2xl font-mono text-sm overflow-hidden">
                  <div className="flex gap-2 mb-4 border-b border-slate-800 pb-2">
                    <div className="h-3 w-3 rounded-full bg-rose-500"></div>
                    <div className="h-3 w-3 rounded-full bg-amber-500"></div>
                    <div className="h-3 w-3 rounded-full bg-emerald-500"></div>
                  </div>
                  <pre className="text-blue-300">
                    {`const analyzeRequest = async (data) => {
  const response = await ai.generate({
    model: "gemini-3-pro-preview",
    tools: ["google_search"],
    // Scalability logic
    cache_policy: "TTL_24H",
    region: "global-edge"
  });
  return response;
};`}
                  </pre>
               </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default DevHub;
