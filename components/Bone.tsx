import React, { useMemo, useState, useCallback, useRef } from 'react';
import { Vector2D, PartName, WalkingEnginePivotOffsets, WalkingEngineProportions } from '../types';
import { adjustBrightness } from '../utils/color-utils';
// ANATOMY is no longer directly imported; dimensions come from props.

// Exported for use in Mannequin.tsx cloneElement
export interface BoneProps { 
  rotation: number;
  length: number; // Final scaled kinematic length
  width?: number; // Final scaled kinematic width
  variant?: 'diamond' | 'waist-teardrop-pointy-up' | 'torso-teardrop-pointy-down' | 'collar-horizontal-oval-shape' | 'deltoid-shape' | 'limb-tapered' | 'head-wedge' | 'hand-foot-arrowhead-shape' | 'foot-block-shape' | 'toe-rounded-cap';
  showPivots: boolean;
  visible?: boolean;
  offset?: Vector2D;
  children?: React.ReactNode;
  drawsUpwards?: boolean;
  colorClass?: string;
  showLabel?: boolean;
  label?: string;
  boneKey?: keyof WalkingEnginePivotOffsets; // Key to identify this bone for pivotOffsets
  proportionKey?: keyof WalkingEngineProportions; // Key to identify this bone for images/props
  onAnchorMouseDown?: (boneKey: keyof WalkingEnginePivotOffsets, clientX: number, e: React.MouseEvent | React.TouchEvent) => void;
  onBodyMouseDown?: (boneKey: keyof WalkingEnginePivotOffsets, clientX: number, e: React.MouseEvent | React.TouchEvent) => void;
  isBeingDragged?: boolean;
  isPausedAndPivotsVisible?: boolean;
  patternFillId?: string;
  isPinned?: boolean;
  isGhost?: boolean;
  // FIX: Added 'static' to the ghostType to allow for static ghost rendering.
  ghostType?: 'ik' | 'fk' | 'static' | null;
  isSelected?: boolean;
  isUnderTension?: boolean;
  tension?: number;
  isAtLimit?: boolean;
}

export const COLORS = {
  ANCHOR_RED: "#F87171", // Anchor dots explicitly red
  SELECTION: "#D1D5DB", // Light monochrome shade
  SELECTION_HIGHLIGHT: "#f59e0b", // amber-500
  RIDGE: "#333333", // For wireframe stroke - kept dark
  PIN_HIGHLIGHT: "#F43F5E", // rose-500 for active pin (pinkish-red)
  DEFAULT_FILL: "#000000", // Fallback / solid black for silhouette
  FOCUS_RING: "#E5E7EB", // Added focus ring color
  IK_HIGHLIGHT: "#0891B2", // Changed to a vibrant cyan for high visibility
  GHOST_IK: "#a855f7",      // Purple-500 for IK ghost prediction
  GHOST_FK: "#a3e635",      // Lime-400 for FK ghost prediction
  GHOST_STATIC: "#9CA3AF",  // Gray-400 for utility/history ghosts
};

// COLORS_BY_CATEGORY is no longer used for dynamic fill, as colorClass is passed directly.
export const COLORS_BY_CATEGORY: { [category: string]: string } = { 
  head: "#5A5A5A",
  hand: "#5A5A5A",
  foot: "#5A5A5A",
  
  bicep: "#3A3A3A",
  forearm: "#3A3A3A",
  collar: "#3A3A3A",
  torso: "#3A3A3A",
  waist: "#3A3A3A",
  thigh: "#3A3A3A",
  shin: "#3A3A3A",

  default: COLORS.DEFAULT_FILL,
};

export const Bone: React.FC<BoneProps> = ({
  rotation,
  length, // This is now the final scaled kinematic length
  width = 15, // This is now the final scaled kinematic width
  variant = 'diamond',
  showPivots = true,
  visible = true,
  offset = { x: 0, y: 0 },
  children,
  drawsUpwards = false,
  colorClass = "fill-mono-dark",
  showLabel = false,
  label,
  boneKey,
  proportionKey,
  onAnchorMouseDown,
  onBodyMouseDown,
  isBeingDragged = false,
  isPausedAndPivotsVisible = false,
  patternFillId,
  isPinned = false,
  isGhost = false,
  ghostType = null,
  isSelected = false,
  isUnderTension = false,
  tension = 1.0,
  isAtLimit = false,
}) => {

  const getBonePath = (boneLength: number, boneWidth: number, variant: string, drawsUpwards: boolean): string => {
    const effectiveLength = drawsUpwards ? -boneLength : boneLength;
    const halfWidth = boneWidth / 2;

    switch (variant) {
      case 'head-wedge':
        const topWidth = boneWidth * 0.55;
        const baseWidth = boneWidth * 0.25;
        const headEffectiveLength = -boneLength;
        return `M ${-baseWidth / 2},0 L ${baseWidth / 2},0 L ${topWidth / 2},${headEffectiveLength} L ${-topWidth / 2},${headEffectiveLength} Z`;

      case 'collar-horizontal-oval-shape':
        const collarVisHeight = boneLength;
        const collarBaseWidth = boneWidth;
        const collarTopWidth = collarBaseWidth * 0.5; 
        return `M ${collarBaseWidth / 2},0 C ${collarBaseWidth * 0.3},${-collarVisHeight * 0.3} ${collarTopWidth * 0.7},${-collarVisHeight * 0.6} ${collarTopWidth / 2},${-collarVisHeight} L ${-collarTopWidth / 2},${-collarVisHeight} C ${-collarTopWidth * 0.7},${-collarVisHeight * 0.6} ${-collarBaseWidth * 0.3},${-collarVisHeight * 0.3} ${-collarBaseWidth / 2},0 Z`;

      case 'waist-teardrop-pointy-up':
        const wHeight = boneLength;
        const wWidth = boneWidth;
        const wTopWidthRatio = 0.6; // Waist top is 60% of its base
        return `M ${wWidth / 2},0 L ${wWidth * wTopWidthRatio / 2},${-wHeight} L ${-wWidth * wTopWidthRatio / 2},${-wHeight} L ${-wWidth / 2},0 Z`;

      case 'torso-teardrop-pointy-down':
        const tHeight = boneLength;
        const tWidth = boneWidth;
        const tBaseWidthRatio = 0.6; // Torso base matches waist top
        return `M ${tWidth * tBaseWidthRatio / 2},0 L ${tWidth / 2},${-tHeight} L ${-tWidth / 2},${-tHeight} L ${-tWidth * tBaseWidthRatio / 2},0 Z`;

      case 'deltoid-shape':
        const dHeight = boneLength;
        const shoulderWidth = boneWidth; 
        return `M ${shoulderWidth / 2} 0
                C ${shoulderWidth / 2} ${dHeight * 0.2} ${shoulderWidth * 1.2 / 2} ${dHeight * 0.4} ${shoulderWidth * 1.2 / 2} ${dHeight * 0.7}
                L 0 ${dHeight}
                L ${-shoulderWidth * 1.2 / 2} ${dHeight * 0.7}
                C ${-shoulderWidth * 1.2 / 2} ${dHeight * 0.4} ${-shoulderWidth / 2} ${dHeight * 0.2} ${-shoulderWidth / 2} 0 Z`;

      case 'limb-tapered':
        const taperedWidth = boneWidth;
        const taperedEndWidth = taperedWidth * 0.65;
        return `M ${taperedWidth / 2},0 L ${taperedEndWidth / 2},${effectiveLength} L ${-taperedEndWidth / 2},${effectiveLength} L ${-taperedWidth / 2},0 Z`;
      
      case 'foot-block-shape':
        const footBaseW = boneWidth;
        const footEndW = boneWidth * 1.4; // Widens towards ball of foot
        return `M ${footBaseW / 2},0 L ${footEndW / 2},${effectiveLength} L ${-footEndW / 2},${effectiveLength} L ${-footBaseW / 2},0 Z`;

      case 'toe-rounded-cap':
        const toeBaseW = boneWidth * 1.4; // Matches foot end for seamless alignment
        return `M ${toeBaseW / 2},0 L 0,${effectiveLength} L ${-toeBaseW / 2},0 Z`;

      case 'hand-foot-arrowhead-shape':
        const handFootWidth = boneWidth;
        const basePointX = handFootWidth * 0.15; // 30% total base width
        return `M ${-basePointX},0 L ${basePointX},0 L 0,${effectiveLength} Z`;

      default:
        const defaultWidth = boneWidth;
        const split = effectiveLength * 0.4;
        return `M 0 0 L ${defaultWidth / 2} ${split} L 0 ${effectiveLength} L ${-defaultWidth / 2} ${split} Z`;
    }
  };

  const visualEndPoint = drawsUpwards ? -length : length;
  const transform = (offset.x !== 0 || offset.y !== 0)
    ? `translate(${offset.x}, ${offset.y}) rotate(${rotation})`
    : `rotate(${rotation})`;

  const anchorCursorStyle = isPausedAndPivotsVisible && onAnchorMouseDown
    ? (isBeingDragged ? 'cursor-grabbing' : 'cursor-grab')
    : 'cursor-default';
    
  const bodyCursorStyle = isPausedAndPivotsVisible && onBodyMouseDown
    ? (isBeingDragged ? 'cursor-grabbing' : 'cursor-grab')
    : 'cursor-default';

  const isGhostly = isGhost;
  
  const ghostColor = ghostType === 'ik' 
      ? COLORS.GHOST_IK 
      : ghostType === 'fk' 
      ? COLORS.GHOST_FK 
      : COLORS.GHOST_STATIC; 

  const boneFill = isGhostly ? (isUnderTension ? 'url(#)' : ghostColor) : (patternFillId || "currentColor"); // Let animation control color
  const boneStrokeColor = isGhostly ? "none" : COLORS.RIDGE;
  const boneStrokeWidth = isGhostly ? 0 : 0.5;
  const boneStrokeDasharray = "none";
  let pathClassName = `${bodyCursorStyle}`;
  if (isGhostly && isUnderTension) {
      pathClassName += ' animate-pulse-red';
  }
  
  const boneWrapperStyle: React.CSSProperties = isSelected && !isGhostly ? {
    filter: `drop-shadow(0 0 4px ${COLORS.SELECTION_HIGHLIGHT})`,
    transition: 'filter 0.2s ease-in-out',
  } : {
    transition: 'filter 0.2s ease-in-out',
  };

  // --- Tension Visuals ---
  const tensionScale = isPinned ? Math.min(1 + (tension - 1) * 0.8, 2.5) : 1;
  const anchorRadius = 5 * tensionScale;
  const anchorFillColor = isPinned ? adjustBrightness(COLORS.ANCHOR_RED, 1 + Math.min(tension - 1, 1)) : COLORS.ANCHOR_RED;

  return (
    <g transform={transform} className={colorClass} style={boneWrapperStyle}>
      {visible && (
        <React.Fragment>
          <path
            d={getBonePath(length, width, variant, drawsUpwards)}
            fill={boneFill}
            stroke={boneStrokeColor}
            strokeWidth={boneStrokeWidth}
            strokeDasharray={boneStrokeDasharray}
            paintOrder="stroke"
            className={pathClassName}
            onMouseDown={(e) => {
                if (isPausedAndPivotsVisible && onBodyMouseDown && boneKey) {
                    onBodyMouseDown(boneKey, e.clientX, e);
                }
            }}
            onTouchStart={(e) => {
                if (isPausedAndPivotsVisible && onBodyMouseDown && boneKey) {
                    onBodyMouseDown(boneKey, e.touches[0].clientX, e);
                }
            }}
          />
          {/* Overlay line for axis */}
          {showPivots && !isGhostly && (
            <line x1="0" y1="0" x2="0" y2={visualEndPoint} stroke="rgba(150, 150, 150, 0.15)" strokeWidth="1" opacity={0.5} strokeLinecap="round" />
          )}
           {showLabel && label && (
            <text x={width / 2 + 5} y={visualEndPoint / 2} 
                  className="fill-mono-mid text-[7px] font-mono select-none opacity-40 tracking-tighter uppercase pointer-events-none"
                  data-is-label="true">
              {label}
            </text>
          )}
        </React.Fragment>
      )}

      <g transform={`translate(0, ${visualEndPoint})`}>{children}</g>

      {/* Anchor (red dot) at the start of the bone, always visible if showPivots */}
      {showPivots && !isGhostly && visible && boneKey && onAnchorMouseDown && (
        <g>
          <circle 
            cx="0" cy="0" r={anchorRadius} 
            fill={anchorFillColor} 
            stroke="white"
            strokeWidth="1"
            className={`drop-shadow-md ${anchorCursorStyle}`} 
            data-no-export="true"
            onMouseDown={(e) => isPausedAndPivotsVisible && onAnchorMouseDown(boneKey, e.clientX, e)}
            onTouchStart={(e) => isPausedAndPivotsVisible && onAnchorMouseDown(boneKey, e.touches[0].clientX, e)}
          />
          {isPinned && (
              <circle
                  cx="0" cy="0" r={10 * tensionScale}
                  fill="none"
                  stroke={isAtLimit ? undefined : COLORS.PIN_HIGHLIGHT}
                  strokeWidth="2"
                  data-no-export="true"
                  className={isAtLimit ? 'animate-strobe-red' : ''}
                  style={!isAtLimit ? { filter: `drop-shadow(0 0 3px ${COLORS.PIN_HIGHLIGHT})`} : {}}
              />
          )}
        </g>
      )}
    </g>
  );
};