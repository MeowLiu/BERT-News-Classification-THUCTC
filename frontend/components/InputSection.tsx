import React, { useRef, useState, useCallback } from 'react';
import { UploadCloud, FileText, X, Sparkles } from 'lucide-react';
import { clsx } from 'clsx';

interface InputSectionProps {
  text: string;
  setText: (text: string) => void;
  disabled?: boolean;
}

export const InputSection: React.FC<InputSectionProps> = ({ text, setText, disabled }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isFocused, setIsFocused] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) setIsDragging(true);
  }, [disabled]);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.currentTarget.contains(e.relatedTarget as Node)) return;
    setIsDragging(false);
  }, []);

  const readFile = (file: File) => {
    if (file.type !== 'text/plain') {
      alert('仅支持 .txt 文本文件');
      return;
    }
    const reader = new FileReader();
    reader.onload = (event) => {
      const content = event.target?.result as string;
      setText(content);
    };
    reader.readAsText(file);
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (disabled) return;
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      readFile(files[0]);
    }
  }, [disabled, setText]);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      readFile(e.target.files[0]);
      e.target.value = '';
    }
  };

  const triggerFileSelect = () => {
    fileInputRef.current?.click();
  };

  return (
    <div 
      className={clsx(
        "relative group rounded-2xl transition-all duration-300 ease-out bg-white/80 backdrop-blur-xl",
        isDragging 
          ? "border-indigo-500 ring-4 ring-indigo-100 scale-[1.02] shadow-2xl" 
          : isFocused
            ? "border-indigo-300 ring-4 ring-indigo-50 shadow-xl"
            : "border-slate-200/60 shadow-lg hover:shadow-xl hover:border-indigo-200"
      )}
      style={{
        borderWidth: '1px'
      }}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* Visual Overlay for Dragging */}
      <div className={clsx(
        "absolute inset-0 z-20 flex flex-col items-center justify-center bg-indigo-50/95 backdrop-blur-sm rounded-2xl transition-opacity duration-200 pointer-events-none",
        isDragging ? "opacity-100" : "opacity-0"
      )}>
        <div className="bg-white p-4 rounded-full shadow-lg mb-4 animate-bounce">
          <UploadCloud className="w-10 h-10 text-indigo-600" />
        </div>
        <p className="text-xl font-bold text-indigo-900">释放以读取文件</p>
        <p className="text-indigo-500 mt-1">支持 .txt 格式</p>
      </div>

      {/* Header/Toolbar */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-slate-100/80">
        <div className="flex items-center gap-2">
          <div className="p-1.5 bg-indigo-50 rounded-md text-indigo-600">
            <FileText className="w-4 h-4" />
          </div>
          <span className="text-sm font-semibold text-slate-700 tracking-wide">输入文本</span>
        </div>
        {!disabled && (
           <button 
             onClick={triggerFileSelect}
             className="group flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-slate-50 text-xs font-medium text-slate-600 hover:bg-indigo-50 hover:text-indigo-600 transition-colors"
           >
             <UploadCloud className="w-3.5 h-3.5 transition-transform group-hover:-translate-y-0.5" />
             上传文件
           </button>
        )}
      </div>

      <textarea
        ref={textareaRef}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onFocus={() => setIsFocused(true)}
        onBlur={() => setIsFocused(false)}
        disabled={disabled}
        placeholder="在此输入需要分类的新闻内容，或者直接将 .txt 文件拖拽至此..."
        className={clsx(
          "w-full h-72 p-6 resize-none outline-none text-slate-700 placeholder:text-slate-400 bg-transparent text-lg leading-relaxed font-light",
          disabled && "opacity-50 cursor-not-allowed"
        )}
        spellCheck={false}
      />

      {/* Decorative corner accent */}
      {!disabled && !text && (
        <div className="absolute bottom-4 right-4 pointer-events-none opacity-40">
           <Sparkles className="w-12 h-12 text-indigo-200" />
        </div>
      )}

      {/* Hidden File Input */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileSelect}
        accept=".txt"
        className="hidden"
      />
    </div>
  );
};