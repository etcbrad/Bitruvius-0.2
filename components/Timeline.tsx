import React, { useRef, useCallback } from 'react';
import { Keyframe } from '../types';
import './Timeline.css';

interface TimelineProps {
  keyframes: Keyframe[];
  onPlay: () => void;
  onPause: () => void;
  onReset: () => void;
  isAnimating: boolean;
  animationTime: number;
  totalDuration: number;
  onAddKeyframe: () => void;
  onSelectKeyframe: (index: number) => void;
  selectedKeyframeIndex: number | null;
  onScrub: (time: number) => void;
  onUpdateKeyframeTime: (index: number, time: number) => void;
  onAddTweenFrame: () => void;
}

const MIN_SEGMENT_DURATION = 100; // 100ms

export const Timeline: React.FC<TimelineProps> = ({ 
    keyframes,
    onPlay,
    onPause,
    onReset,
    isAnimating,
    animationTime,
    totalDuration,
    onAddKeyframe,
    onSelectKeyframe,
    selectedKeyframeIndex,
    onScrub,
    onUpdateKeyframeTime,
    onAddTweenFrame,
}) => {
  const trackRef = useRef<HTMLDivElement>(null);
  const playbackHeadPosition = totalDuration > 0 ? (animationTime / totalDuration) * 100 : 0;

  const handleKeyframeMouseDown = useCallback((e: React.MouseEvent<HTMLDivElement>, index: number) => {
    e.stopPropagation();
    if (index === 0) return; // Cannot drag the first keyframe

    const startX = e.clientX;
    const startTime = keyframes[index].time;
    const trackWidth = trackRef.current?.offsetWidth;
    if (!trackWidth) return;

    const handleMouseMove = (moveEvent: MouseEvent) => {
      const deltaX = moveEvent.clientX - startX;
      const deltaTime = (deltaX / trackWidth) * totalDuration;
      let newTime = startTime + deltaTime;
      
      const prevTime = keyframes[index - 1].time;
      const nextTime = (index + 1 < keyframes.length) ? keyframes[index + 1].time : Infinity;

      // Clamp the new time
      newTime = Math.max(prevTime + MIN_SEGMENT_DURATION, newTime);
      newTime = Math.min(nextTime - MIN_SEGMENT_DURATION, newTime);

      onUpdateKeyframeTime(index, newTime);
    };

    const handleMouseUp = () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
  }, [keyframes, totalDuration, onUpdateKeyframeTime]);

  const handleScrubStart = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!trackRef.current || keyframes.length < 2) return;
    
    const rect = trackRef.current.getBoundingClientRect();

    const updateScrubTime = (clientX: number) => {
        const x = clientX - rect.left;
        const percentage = Math.max(0, Math.min(1, x / rect.width));
        const targetTime = totalDuration * percentage;
        onScrub(targetTime);
    };

    updateScrubTime(e.clientX);

    const handleMouseMove = (moveEvent: MouseEvent) => {
      updateScrubTime(moveEvent.clientX);
    };

    const handleMouseUp = () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
  }, [totalDuration, onScrub, keyframes.length]);

  return (
    <div className="timeline-container">
      <div className="timeline-controls">
        {isAnimating ? (
            <button onClick={onPause} className="timeline-button" disabled={keyframes.length < 2}>❚❚</button>
        ) : (
            <button onClick={onPlay} className="timeline-button" disabled={keyframes.length < 2}>▶</button>
        )}
        <button onClick={onReset} className="timeline-button" disabled={keyframes.length === 0}>⟲</button>
        <button onClick={onAddKeyframe} className="timeline-button add-keyframe-button" title="Add keyframe from current pose">
            +<span className="button-label">Pose</span>
        </button>
        <button onClick={onAddTweenFrame} className="timeline-button add-keyframe-button" title="Add current interpolated pose as a new keyframe" disabled={isAnimating || keyframes.length < 2}>
            +<span className="button-label">Tween</span>
        </button>
        <span className="timeline-duration">{(totalDuration / 1000).toFixed(2)}s</span>
      </div>
      <div 
        ref={trackRef}
        className="timeline-track-wrapper" 
        onMouseDown={handleScrubStart}
      >
        <div className="timeline-track" style={{ width: `100%` }}>
          {keyframes.map((keyframe, index) => {
            const leftPercentage = totalDuration > 0 ? (keyframe.time / totalDuration) * 100 : 0;
            const isSelected = selectedKeyframeIndex === index;
            return (
              <div
                key={keyframe.id}
                onMouseDown={(e) => handleKeyframeMouseDown(e, index)}
                onClick={(e) => {
                  e.stopPropagation();
                  onSelectKeyframe(index);
                }}
                className={`timeline-keyframe ${isSelected ? 'selected' : ''} ${index === 0 ? 'locked' : ''}`}
                style={{ left: `${leftPercentage}%` }}
                title={`Keyframe ${index + 1} @ ${(keyframe.time / 1000).toFixed(2)}s`}
              />
            );
          })}
          <div 
            className="timeline-playback-head" 
            style={{ left: `${playbackHeadPosition}%` }}
          />
        </div>
      </div>
    </div>
  );
};
