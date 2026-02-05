

import React from 'react';
import { Vector2D } from '../types'; // Assuming Vector2D is defined in types.ts
import { ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT, GROUND_STRIP_HEIGHT_RAW_H_UNIT } from '../constants'; // Import ANATOMY and new ground constants

const GROUND_STRIP_COLOR = '#F3F4F6'; // A very subtle light grey for the ground strip on a white background.

interface AdvancedGridProps {
  origin: Vector2D;
  gridSize: number;
  viewBox: { x: number; y: number; width: number; height: number };
}

interface SystemGuidesProps {
  floorY: number; // This will be the absolute SVG coordinate for the floor
  baseUnitH: number; // The H from the walking engine
}

export const Scanlines: React.FC = () => (
  <svg width="100%" height="100%" className="absolute inset-0 z-10 pointer-events-none opacity-20">
    <defs>
      <pattern id="scanlines" patternUnits="userSpaceOnUse" width="1" height="4">
        <line x1="0" y1="1" x2="1" y2="1" stroke="#E5E7EB" strokeWidth="1" />
      </pattern>
    </defs>
    <rect width="100%" height="100%" fill="url(#scanlines)" />
  </svg>
);

export const AdvancedGrid: React.FC<AdvancedGridProps> = ({ origin, gridSize, viewBox }) => {
  if (gridSize <= 0) return null;

  const lines: React.ReactNode[] = [];
  const minorGridSize = gridSize / 4;

  // Vertical lines
  const startX = Math.floor((viewBox.x - origin.x) / minorGridSize) * minorGridSize + origin.x;
  const endX = viewBox.x + viewBox.width;
  for (let x = startX; x <= endX; x += minorGridSize) {
    const isMajor = Math.abs(Math.round((x - origin.x) / minorGridSize)) % 4 === 0;
    lines.push(
      <line
        key={`v-${x}`}
        x1={x}
        y1={viewBox.y}
        x2={x}
        y2={viewBox.y + viewBox.height}
        stroke={isMajor ? 'rgba(0, 0, 0, 0.1)' : 'rgba(0, 0, 0, 0.05)'} // Changed to monochrome rgba
        strokeWidth={isMajor ? 1 : 0.5}
      />
    );
  }

  // Horizontal lines
  const startY = Math.floor((viewBox.y - origin.y) / minorGridSize) * minorGridSize + origin.y;
  const endY = viewBox.y + viewBox.height;
  for (let y = startY; y <= endY; y += minorGridSize) {
    const isMajor = Math.abs(Math.round((y - origin.y) / minorGridSize)) % 4 === 0;
    lines.push(
      <line
        key={`h-${y}`}
        x1={viewBox.x}
        y1={y}
        x2={viewBox.x + viewBox.width}
        y2={y}
        stroke={isMajor ? 'rgba(0, 0, 0, 0.1)' : 'rgba(0, 0, 0, 0.05)'} // Changed to monochrome rgba
        strokeWidth={isMajor ? 1 : 0.5}
      />
    );
  }

  return <g className="pointer-events-none">{lines}</g>;
};

export const SystemGuides: React.FC<SystemGuidesProps> = ({ floorY, baseUnitH }) => {
  const guideColor = 'rgba(0, 0, 0, 0.08)'; // Changed to monochrome rgba
  const span = 20000; // Extend guide lines far beyond typical viewport

  // Calculate ground strip width relative to H and position it centered on X=0
  const groundStripWidth = ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.FOOT * baseUnitH * 4; // Adjusted for visual balance
  const groundStripX = -groundStripWidth / 2;

  return (
    <g className="pointer-events-none">
      {/* Center X-axis guide */}
      <line x1={-span} y1="0" x2={span} y2="0" stroke={guideColor} strokeWidth="1" opacity="0.3" strokeDasharray="10 5" />
      {/* Center Y-axis guide */}
      <line x1="0" y1={-span} x2="0" y2={span} stroke={guideColor} strokeWidth="1" opacity="0.3" strokeDasharray="10 5" />

      {/* Floor guide line and ground strip */}
        <g style={{ transition: 'all 0.2s ease-in-out' }}>
          {/* Main floor guide line */}
          <line x1={-span} y1={floorY} x2={span} y2={floorY} stroke={guideColor} strokeWidth={1} opacity="0.9" />
          {/* Darker ground strip at the floor level */}
          <rect
            x={groundStripX} 
            y={floorY}
            width={groundStripWidth} 
            height={GROUND_STRIP_HEIGHT_RAW_H_UNIT * baseUnitH}
            fill={GROUND_STRIP_COLOR}
            opacity="0.9"
          />
        </g>
    </g>
  );
};