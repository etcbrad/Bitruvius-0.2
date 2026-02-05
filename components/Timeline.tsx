

import React from 'react';
import { AnimationClip } from '../types';
import './Timeline.css';

interface TimelineProps {
  animationSequence: AnimationClip[];
  onPlay: () => void;
  onPause: () => void;
  onReset: () => void;
  isAnimating: boolean;
  animationTime: number;
  totalDuration: number;
}

export const Timeline: React.FC<TimelineProps> = ({ 
    animationSequence,
    onPlay,
    onPause,
    onReset,
    isAnimating,
    animationTime,
    totalDuration
}) => {
  const trackWidth = 500;
  const pixelsPerMillisecond = totalDuration > 0 ? trackWidth / totalDuration : 0;
  const playbackHeadPosition = animationTime * pixelsPerMillisecond;

  return (
    <div className="timeline-container">
      <div className="timeline-controls">
        {isAnimating ? (
            <button onClick={onPause} className="timeline-button">❚❚</button>
        ) : (
            <button onClick={onPlay} className="timeline-button">▶</button>
        )}
        <button onClick={onReset} className="timeline-button">⟲</button>
        <button className="timeline-button">+</button>
        <span className="timeline-duration">{(totalDuration / 1000).toFixed(2)}s</span>
      </div>
      <div className="timeline-track-wrapper">
        <div className="timeline-track" style={{ width: `${trackWidth}px` }}>
          {animationSequence.map((clip, index) => {
            const left = animationSequence.slice(0, index).reduce((sum, c) => sum + (c.duration * pixelsPerMillisecond), 0);
            const width = clip.duration * pixelsPerMillisecond;
            
            return (
              <div
                key={clip.id}
                className="timeline-clip"
                style={{ left: `${left}px`, width: `${width}px` }}
                title={`${clip.name} (${(clip.duration / 1000).toFixed(2)}s)`}
              >
                <span className="clip-name">{clip.name}</span>
              </div>
            );
          })}
          <div 
            className="timeline-playback-head" 
            style={{ left: `${playbackHeadPosition}px` }}
          />
        </div>
      </div>
    </div>
  );
};