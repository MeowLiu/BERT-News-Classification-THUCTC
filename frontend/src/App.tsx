import React, { useState, useEffect } from 'react';
import { Bot, Eraser, Play, Loader2, Sparkles, BrainCircuit, Quote } from 'lucide-react';
import { clsx } from 'clsx';
import { InputSection } from './components/InputSection';
import { ResultSection } from './components/ResultSection';
import { Button } from './components/ui/Button';
import { ModelType, PredictResponse } from './types';
import { mockAnalyze } from './services/mockApi';
import { SAMPLE_TEXT } from './constants';

const ROTATING_WORDS = ["财经", "彩票", "房产", "股票", "家居", "时尚", "时政", "体育", "科技", "娱乐", "教育", "星座", "游戏", "社会"];

const App: React.FC = () => {
  const [text, setText] = useState<string>('');
  const [model, setModel] = useState<ModelType>(ModelType.BERT);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [result, setResult] = useState<PredictResponse | null>(null);

  // Hero Animation State
  const [wordIndex, setWordIndex] = useState(0);
  const [fadeWord, setFadeWord] = useState(true);

  // Typewriter effect interval
  useEffect(() => {
    const interval = setInterval(() => {
      setFadeWord(false);
      setTimeout(() => {
        setWordIndex((prev) => (prev + 1) % ROTATING_WORDS.length);
        setFadeWord(true);
      }, 500); // Wait for fade out
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleAnalyze = async () => {
    if (!text.trim()) return;

    setIsLoading(true);
    setResult(null);

    try {
      const response = await mockAnalyze(text, model);
      setResult(response);
      // Smooth scroll to results
      setTimeout(() => {
        document.getElementById('results-anchor')?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    } catch (error) {
      console.error("Analysis failed", error);
      alert("分析过程中发生错误，请重试。");
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    setText('');
    setResult(null);
  };

  return (
    <div className="min-h-screen relative overflow-hidden bg-slate-50 selection:bg-indigo-200">

      {/* --- Animated Background Elements --- */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none z-0">
        <div className="absolute top-[-10%] left-[-10%] w-96 h-96 bg-purple-300 rounded-full mix-blend-multiply filter blur-3xl opacity-30 animate-blob"></div>
        <div className="absolute top-[-10%] right-[-10%] w-96 h-96 bg-yellow-300 rounded-full mix-blend-multiply filter blur-3xl opacity-30 animate-blob animation-delay-2000"></div>
        <div className="absolute bottom-[-20%] left-[20%] w-96 h-96 bg-indigo-300 rounded-full mix-blend-multiply filter blur-3xl opacity-30 animate-blob animation-delay-4000"></div>
        {/* Grid Pattern overlay */}
        <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20"></div>
      </div>

      {/* --- Header / Nav --- */}
      <nav className="relative z-10 w-full px-6 py-6 flex justify-between items-center max-w-7xl mx-auto">
        <div className="flex items-center gap-2.5">
          <div className="bg-slate-900 text-white p-2 rounded-xl shadow-lg shadow-indigo-500/20">
            <BrainCircuit className="w-6 h-6" />
          </div>
          <span className="text-xl font-bold tracking-tight text-slate-900">THUCTCNews<span className="text-indigo-600"> Classification</span></span>
        </div>
        <div className="hidden sm:flex items-center gap-4">
          <span className="text-sm font-medium text-slate-500">智能模型: v1.0.0</span>
          {/* <a href="#" className="text-sm font-semibold text-slate-900 hover:text-indigo-600 transition-colors">Documentation</a> */}
          <a href="https://github.com/MeowLiu/BERT-News-Classification-THUCTC.git" className="text-sm font-semibold text-slate-900 hover:text-indigo-600 transition-colors">Github</a>
        </div>
      </nav>

      <main className="relative z-10 max-w-5xl mx-auto px-4 sm:px-6 pt-10 pb-24">

        {/* --- Hero Section --- */}
        <div className="text-center mb-12 space-y-6">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/50 border border-indigo-100 backdrop-blur-sm text-indigo-700 text-xs font-semibold uppercase tracking-wider shadow-sm mb-4">
            <Sparkles className="w-3 h-3" />
            <span>AI赋能新闻分类</span>
          </div>

          <h1 className="text-5xl sm:text-7xl font-extrabold text-slate-900 tracking-tight leading-[1.1]">
            智能识别 <br className="sm:hidden" />
            <span className="text-gradient">
              {ROTATING_WORDS[wordIndex]}
            </span>
            <span className={clsx("transition-opacity duration-500", fadeWord ? "opacity-100" : "opacity-0")}>
              新闻
            </span>
            <span className="inline-block w-1 h-12 sm:h-16 ml-2 align-middle bg-indigo-500 cursor-blink rounded-full"></span>
          </h1>

          <p className="max-w-2xl mx-auto text-lg text-slate-600 leading-relaxed">
            基于 <span className="font-semibold text-slate-800">BERT</span> 与 <span className="font-semibold text-slate-800">RoBERTa</span> 预训练模型，为您提供毫秒级的专业新闻文本分类服务。支持文件拖拽，一键分析。
          </p>
        </div>

        {/* --- Main Interaction Card --- */}
        <div className="fade-up-enter stagger-1">
          <div className="bg-white/40 backdrop-blur-lg border border-white/60 p-1.5 rounded-[20px] shadow-2xl shadow-indigo-100/50">
            <div className="bg-white/50 rounded-2xl p-6 sm:p-8 space-y-6">

              {/* Toolbar */}
              <div className="flex flex-col sm:flex-row justify-between items-center gap-4 mb-2">
                {/* Model Switcher */}
                <div className="bg-slate-100/80 p-1 rounded-xl flex items-center w-full sm:w-auto">
                  <button
                    onClick={() => setModel(ModelType.BERT)}
                    className={clsx(
                      "flex-1 sm:flex-none px-6 py-2 rounded-lg text-sm font-bold transition-all duration-200",
                      model === ModelType.BERT
                        ? "bg-white text-indigo-600 shadow-md transform scale-105"
                        : "text-slate-500 hover:text-slate-700"
                    )}
                  >
                    BERT
                  </button>
                  <button
                    onClick={() => setModel(ModelType.ROBERTA)}
                    className={clsx(
                      "flex-1 sm:flex-none px-6 py-2 rounded-lg text-sm font-bold transition-all duration-200",
                      model === ModelType.ROBERTA
                        ? "bg-white text-purple-600 shadow-md transform scale-105"
                        : "text-slate-500 hover:text-slate-700"
                    )}
                  >
                    RoBERTa
                  </button>
                </div>

                {/* Actions */}
                <div className="flex items-center gap-2 w-full sm:w-auto justify-end">
                  <Button variant="ghost" size="sm" onClick={() => setText(SAMPLE_TEXT)} className="text-slate-600 hover:bg-amber-50 hover:text-amber-600">
                    <Quote className="w-4 h-4 mr-1.5" />
                    填充示例
                  </Button>
                  <Button variant="ghost" size="sm" onClick={handleClear} disabled={!text} className="text-slate-600 hover:bg-red-50 hover:text-red-600">
                    <Eraser className="w-4 h-4 mr-1.5" />
                    清空
                  </Button>
                </div>
              </div>

              {/* The Big Input */}
              <InputSection
                text={text}
                setText={setText}
                disabled={isLoading}
              />

              {/* Action Button */}
              <div className="flex justify-center pt-4">
                <button
                  onClick={handleAnalyze}
                  disabled={isLoading || !text.trim()}
                  className={clsx(
                    "group relative overflow-hidden w-full sm:w-auto min-w-[240px] px-8 py-4",
                    "rounded-lg shadow-lg shadow-indigo-500/25 hover:shadow-purple-500/40",
                    "bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500",
                    "text-white text-lg font-bold tracking-wide",
                    "transition-all duration-300 ease-out transform hover:-translate-y-1 active:translate-y-0 active:scale-95",
                    "border border-white/20",
                    (isLoading || !text.trim()) && "opacity-80 grayscale cursor-not-allowed hover:transform-none hover:shadow-none"
                  )}
                >
                  {/* Shimmer Effect Layer */}
                  <div className="absolute inset-0 -translate-x-full group-hover:animate-shimmer bg-gradient-to-r from-transparent via-white/25 to-transparent z-10" />

                  {/* Button Content */}
                  <div className="relative z-20 flex items-center justify-center gap-2">
                    {isLoading ? (
                      <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        <span>AI 思考中...</span>
                      </>
                    ) : (
                      <>
                        <Play className="w-5 h-5 fill-current" />
                        <span>开始智能识别</span>
                      </>
                    )}
                  </div>
                </button>
              </div>

            </div>
          </div>
        </div>

        {/* --- Results Anchor & Section --- */}
        <div id="results-anchor" className="scroll-mt-24">
          {result && (
            <ResultSection result={result} />
          )}
        </div>

      </main>

      {/* Footer decorative */}
      <footer className="py-8 text-center text-slate-400 text-sm relative z-10">
        <p>© 2025 THUCTCNews 新闻文本分类系统</p>
      </footer>
    </div>
  );
};

export default App;