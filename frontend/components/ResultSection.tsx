import React from 'react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  Cell 
} from 'recharts';
import { PredictResponse } from '../types';
import { Activity, Clock, Award, Zap } from 'lucide-react';

interface ResultSectionProps {
  result: PredictResponse | null;
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-slate-900/95 backdrop-blur text-white border border-slate-700 p-3 rounded-xl shadow-2xl text-sm">
        <p className="font-semibold mb-1 opacity-90">{label}</p>
        <p className="text-indigo-300 font-mono">
          {payload[0].value}%
        </p>
      </div>
    );
  }
  return null;
};

export const ResultSection: React.FC<ResultSectionProps> = ({ result }) => {
  if (!result) return null;

  return (
    <div className="space-y-8 mt-8">
      
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {/* Category Card */}
        <div className="fade-up-enter stagger-1 bg-white/60 backdrop-blur-md rounded-2xl border border-white/50 p-6 shadow-lg hover:shadow-xl transition-shadow duration-300 flex flex-col justify-between h-32 relative overflow-hidden group">
          <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
            <Award className="w-16 h-16 text-indigo-600 transform rotate-12 translate-x-4 -translate-y-4" />
          </div>
          <div className="flex items-center gap-3">
             <div className="p-2.5 bg-indigo-100 text-indigo-600 rounded-lg">
                <Award className="w-5 h-5" />
             </div>
             <p className="text-sm font-semibold text-slate-500 uppercase tracking-wider">预测分类</p>
          </div>
          <h3 className="text-3xl font-black text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 to-purple-600">
            {result.category}
          </h3>
        </div>

        {/* Confidence Card */}
        <div className="fade-up-enter stagger-2 bg-white/60 backdrop-blur-md rounded-2xl border border-white/50 p-6 shadow-lg hover:shadow-xl transition-shadow duration-300 flex flex-col justify-between h-32 group">
          <div className="flex items-center gap-3">
             <div className="p-2.5 bg-emerald-100 text-emerald-600 rounded-lg">
                <Activity className="w-5 h-5" />
             </div>
             <p className="text-sm font-semibold text-slate-500 uppercase tracking-wider">置信度</p>
          </div>
          <div className="flex items-baseline gap-1">
            <h3 className="text-3xl font-black text-slate-900">
              {(result.confidence * 100).toFixed(1)}
            </h3>
            <span className="text-lg font-medium text-slate-400">%</span>
          </div>
          {/* Simple progress bar */}
          <div className="w-full bg-slate-100 h-1.5 rounded-full mt-2 overflow-hidden">
             <div 
               className="h-full bg-emerald-500 rounded-full transition-all duration-1000 ease-out" 
               style={{ width: `${result.confidence * 100}%` }}
             />
          </div>
        </div>

        {/* Time Card */}
        <div className="fade-up-enter stagger-3 bg-white/60 backdrop-blur-md rounded-2xl border border-white/50 p-6 shadow-lg hover:shadow-xl transition-shadow duration-300 flex flex-col justify-between h-32">
          <div className="flex items-center gap-3">
             <div className="p-2.5 bg-amber-100 text-amber-600 rounded-lg">
                <Zap className="w-5 h-5" />
             </div>
             <p className="text-sm font-semibold text-slate-500 uppercase tracking-wider">推理耗时</p>
          </div>
          <div className="flex items-baseline gap-1">
            <h3 className="text-3xl font-black text-slate-900">
              {result.inference_time_ms}
            </h3>
            <span className="text-lg font-medium text-slate-400">ms</span>
          </div>
        </div>
      </div>

      {/* Chart Section */}
      <div className="fade-up-enter stagger-4 bg-white/80 backdrop-blur-xl rounded-2xl border border-white/60 p-8 shadow-xl relative overflow-hidden">
        {/* Background glow for chart */}
        <div className="absolute top-0 right-0 w-64 h-64 bg-indigo-100/50 rounded-full blur-3xl -z-10 translate-x-1/3 -translate-y-1/3" />
        
        <div className="flex items-center justify-between mb-8">
            <h4 className="text-lg font-bold text-slate-900 flex items-center gap-2">
              <span className="w-1.5 h-6 bg-indigo-500 rounded-full"></span>
              Top 5 概率分布
            </h4>
        </div>

        <div className="h-[300px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={result.probabilities}
              layout="vertical"
              margin={{ top: 0, right: 40, left: 40, bottom: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="#e2e8f0" />
              <XAxis type="number" hide domain={[0, 100]} />
              <YAxis 
                dataKey="name" 
                type="category" 
                tick={{ fill: '#475569', fontSize: 14, fontWeight: 500 }} 
                width={60}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip cursor={{ fill: 'transparent' }} content={<CustomTooltip />} />
              <Bar 
                dataKey="value" 
                radius={[0, 6, 6, 0]}
                barSize={32}
                animationDuration={1500}
                animationEasing="ease-out"
              >
                {result.probabilities.map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={index === 0 ? 'url(#colorGradient)' : '#cbd5e1'} 
                  />
                ))}
              </Bar>
              <defs>
                <linearGradient id="colorGradient" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#4f46e5" />
                  <stop offset="100%" stopColor="#9333ea" />
                </linearGradient>
              </defs>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};