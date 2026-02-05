

import React, { useEffect, useRef } from 'react';

interface SystemLoggerProps {
  logs: any[];
  isVisible: boolean;
  onExportJSON?: () => void;
  onClearHistory?: () => void;
  historyCount?: number;
  onLogMouseEnter: (log: any) => void;
  onLogMouseLeave: () => void;
  onLogClick: (log: any, index: number) => void;
  selectedLogIndex: number | null;
}

export const SystemLogger: React.FC<SystemLoggerProps> = ({ 
    logs, 
    isVisible, 
    onExportJSON, 
    onClearHistory, 
    historyCount = 0, 
    onLogMouseEnter, 
    onLogMouseLeave,
    onLogClick,
    selectedLogIndex,
}) => {
  const logContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs]);

  if (!isVisible) {
    return null;
  }

  return (
    <div
      className="relative w-full max-h-64 bg-paper/50 rounded-lg p-3 font-mono text-mono-mid text-[10px] flex flex-col gap-1 shadow-inner border border-ridge mt-4"
    >
      <div className="flex justify-between items-center border-b border-ridge pb-1 mb-1">
        <div className="text-ink font-bold tracking-widest uppercase flex items-center gap-2">
          <span>SYSTEM LOG</span>
          {historyCount > 0 && <span className="text-[7px] bg-selection text-paper px-1 rounded-sm">{historyCount}</span>}
        </div>
        <div className="flex gap-2 items-center">
            <button 
              onClick={onExportJSON}
              disabled={historyCount === 0}
              className="text-[8px] font-bold text-mono-mid hover:text-ink disabled:opacity-30 transition-colors uppercase"
            >
              JSON
            </button>
            <button 
              onClick={onClearHistory}
              disabled={historyCount === 0}
              title="Clear recording history"
              className="text-[8px] text-accent-red hover:opacity-70 disabled:opacity-20 transition-opacity"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
            </button>
        </div>
      </div>
      <div ref={logContainerRef} className="overflow-y-auto custom-scrollbar pr-2 h-32">
        {logs.length === 0 ? (
          <div className="text-mono-light italic text-center py-4">Awaiting system input...</div>
        ) : (
          logs.map((log, index) => {
            const isSelected = selectedLogIndex === index;
            const hasPoseData = !!log.pivotOffsets;
            return (
                <div 
                  key={index} 
                  className={`flex leading-tight py-0.5 border-b border-ridge/20 last:border-0 group rounded-sm -mx-1 px-1 transition-colors ${isSelected ? 'bg-selection text-paper' : 'hover:bg-selection-super-light'} ${hasPoseData ? 'cursor-pointer' : 'cursor-default'}`}
                  onMouseEnter={() => hasPoseData && onLogMouseEnter(log)}
                  onMouseLeave={onLogMouseLeave}
                  onClick={() => onLogClick(log, index)}
                >
                  <span className={`mr-2 shrink-0 group-hover:text-ink transition-colors ${isSelected ? 'text-paper/80' : 'text-mono-light'}`}>{new Date(log.timestamp).toLocaleTimeString()}</span>
                  <span className="flex-1 whitespace-pre-wrap break-words">{log.label || `Pose Snapshot`}</span>
                </div>
            )
          })
        )}
      </div>
    </div>
  );
};