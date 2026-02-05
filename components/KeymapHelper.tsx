import React from 'react';

interface KeymapHelperProps {
  onClose: () => void;
}

export const KeymapHelper: React.FC<KeymapHelperProps> = ({ onClose }) => {
  return (
    <div 
      className="fixed inset-0 bg-black/60 z-[100] flex items-center justify-center p-4 animate-in fade-in duration-200"
      onClick={onClose}
      aria-modal="true"
      role="dialog"
    >
      <div 
        className="bg-mono-darker rounded-lg border border-ridge p-6 shadow-2xl w-full max-w-2xl text-ink font-mono"
        onClick={(e) => e.stopPropagation()} // Prevent closing when clicking inside
      >
        <div className="flex justify-between items-center border-b border-ridge pb-3 mb-6">
          <h2 className="text-xl font-archaic tracking-widest uppercase">8BitDo Controller Map</h2>
          <button onClick={onClose} className="p-1 hover:bg-selection-super-light rounded" aria-label="Close">
            <svg className="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 6 6 18M6 6l12 12"/></svg>
          </button>
        </div>

        <div className="flex flex-col items-center gap-4 text-xs">
          {/* Controller graphic */}
          <div className="relative w-[500px] h-[230px] bg-[#D1D5DB] rounded-[100px] border-b-4 border-[#9CA3AF] p-8 flex justify-between items-center">
            {/* Shoulder Buttons */}
            <div className="absolute -top-4 left-12 w-20 h-8 bg-[#B0B4BA] rounded-t-lg border-t-2 border-l-2 border-r-2 border-[#9CA3AF] flex items-center justify-center">
                <span className="text-sm text-[#6B7280] font-bold">L</span>
            </div>
            <div className="absolute -top-4 right-12 w-20 h-8 bg-[#B0B4BA] rounded-t-lg border-t-2 border-l-2 border-r-2 border-[#9CA3AF] flex items-center justify-center">
                <span className="text-sm text-[#6B7280] font-bold">R</span>
            </div>

            {/* D-Pad */}
            <div className="relative w-24 h-24">
                <div className="absolute top-1/2 left-0 -translate-y-1/2 w-full h-8 bg-[#4B5563] rounded-md shadow-inner"></div>
                <div className="absolute left-1/2 top-0 -translate-x-1/2 h-full w-8 bg-[#4B5563] rounded-md shadow-inner"></div>
            </div>

            {/* Center Buttons */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 flex gap-4 mt-8">
                <div className="w-12 h-6 bg-[#4B5563] rounded-full transform -rotate-12 flex items-center justify-center text-paper/50 text-[10px] font-bold shadow-inner">SELECT</div>
                <div className="w-12 h-6 bg-[#4B5563] rounded-full transform -rotate-12 flex items-center justify-center text-paper/50 text-[10px] font-bold shadow-inner">START</div>
            </div>

            {/* Action Buttons */}
            <div className="relative w-28 h-28 grid grid-cols-2 grid-rows-2 gap-3 transform -translate-y-2">
                <div className="flex items-center justify-center"><div className="w-10 h-10 rounded-full bg-accent-purple border-b-2 border-black/20 flex items-center justify-center text-white font-bold shadow-md">X</div></div>
                <div className="flex items-center justify-center"><div className="w-10 h-10 rounded-full bg-accent-green border-b-2 border-black/20 flex items-center justify-center text-white font-bold shadow-md">Y</div></div>
                <div className="flex items-center justify-center"><div className="w-10 h-10 rounded-full bg-accent-red border-b-2 border-black/20 flex items-center justify-center text-white font-bold shadow-md">B</div></div>
                <div className="flex items-center justify-center"><div className="w-10 h-10 rounded-full bg-[#2563eb] border-b-2 border-black/20 flex items-center justify-center text-white font-bold shadow-md">A</div></div>
            </div>
          </div>
          
          {/* Legend */}
          <div className="w-full grid grid-cols-2 gap-x-8 gap-y-2 mt-4 p-4 border border-ridge/50 bg-paper/30 rounded text-mono-mid">
            <div className="flex items-center gap-2"><span className="font-bold w-24 text-ink">D-PAD:</span><span>Nudge Selected Joint</span></div>
            <div className="flex items-center gap-2"><span className="font-bold w-24 text-ink">START:</span><span>Toggle Console Panel</span></div>
            <div className="flex items-center gap-2"><span className="font-bold w-24 text-ink">SELECT:</span><span>Reset to T-Pose</span></div>
            <div className="flex items-center gap-2"><span className="font-bold w-24 text-ink">L SHOULDER:</span><span>Undo Last Action</span></div>
            <div className="flex items-center gap-2"><span className="font-bold w-24 text-ink">R SHOULDER:</span><span>Redo Last Action</span></div>
            <div className="flex items-center gap-2"><div className="w-4 h-4 rounded-full bg-[#2563eb] border border-black/20"></div><span className="font-bold w-20 text-ink">A Button:</span><span>Cycle Selection Scope</span></div>
            <div className="flex items-center gap-2"><div className="w-4 h-4 rounded-full bg-accent-red border border-black/20"></div><span className="font-bold w-20 text-ink">B Button:</span><span>Toggle BEND Mode</span></div>
            <div className="flex items-center gap-2"><div className="w-4 h-4 rounded-full bg-accent-purple border border-black/20"></div><span className="font-bold w-20 text-ink">X Button:</span><span>Toggle STRETCH Mode</span></div>
            <div className="flex items-center gap-2"><div className="w-4 h-4 rounded-full bg-accent-green border border-black/20"></div><span className="font-bold w-20 text-ink">Y Button:</span><span>Toggle LEAD Mode</span></div>
            <div className="col-span-2 text-center text-mono-light mt-2 italic">Analog sticks are reserved for future physics-based locomotion.</div>
          </div>
        </div>
      </div>
    </div>
  );
};
