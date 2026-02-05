

import React from 'react';
import { Bone } from './Bone';
import { WalkingEnginePivotOffsets, WalkingEngineProportions } from '../types';
import { partDefinitions } from './Mannequin';

interface JointVisualizerProps {
  jointKey: keyof WalkingEnginePivotOffsets;
  pivotOffsets: WalkingEnginePivotOffsets;
  proportions: WalkingEngineProportions;
  baseUnitH: number;
}

const PIVOT_TO_PART_MAP: Record<keyof WalkingEnginePivotOffsets, keyof WalkingEngineProportions> = {
  waist: 'waist', torso: 'torso', collar: 'collar', neck: 'head',
  l_shoulder: 'l_upper_arm', l_elbow: 'l_lower_arm', l_hand: 'l_hand',
  r_shoulder: 'r_upper_arm', r_elbow: 'r_lower_arm', r_hand: 'r_hand',
  l_hip: 'l_upper_leg', l_knee: 'l_lower_leg', l_foot: 'l_foot', l_toe: 'l_toe',
  r_hip: 'r_upper_leg', r_knee: 'r_lower_leg', r_foot: 'r_foot', r_toe: 'r_toe',
};

export const JointVisualizer: React.FC<JointVisualizerProps> = ({ jointKey, pivotOffsets, proportions, baseUnitH }) => {
  const partKey = PIVOT_TO_PART_MAP[jointKey];
  const definition = partDefinitions[partKey];

  if (!definition) return null;

  const getScaledDimension = (raw: number, key: keyof WalkingEngineProportions, axis: 'w' | 'h') => {
    return raw * baseUnitH * (proportions[key]?.[axis] || 1);
  };

  const length = getScaledDimension(definition.rawH, partKey, 'h');
  const width = getScaledDimension(definition.rawW, partKey, 'w');
  const rotation = pivotOffsets[jointKey] || 0;

  return (
    <div className="w-full h-32 bg-paper/50 rounded flex items-center justify-center p-2 border border-ridge/50 shadow-inner">
      <svg viewBox="-100 -150 200 200" className="w-full h-full">
        <g transform="translate(0, 50)">
          <Bone
            rotation={rotation}
            length={length * 0.8}
            width={width * 0.8}
            variant={definition.variant}
            drawsUpwards={definition.drawsUpwards}
            showPivots={true}
            visible={true}
            colorClass={partKey === 'collar' ? 'fill-olive' : 'fill-mono-dark'}
          />
        </g>
      </svg>
    </div>
  );
};