import React, { useRef, useCallback } from 'react';
import { Vector2D } from '../types';

interface Props {
  position: Vector2D;
  onRotate: (deltaAngle: number) => void;
  onRotateStart: () => void;
  onRotateEnd: () => void;
}

export const RotationWheelControl: React.FC<Props> = ({ position, onRotate, onRotateStart, onRotateEnd }) => {
  const lastAngleRef = useRef<number | null>(null);
  const wheelRef = useRef<SVGGElement>(null);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (lastAngleRef.current === null) return;
    
    const svg = wheelRef.current?.ownerSVGElement;
    if (!svg) return;
    
    // Create a point for screen coordinates
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    
    // Transform client coords to SVG coords
    const svgP = pt.matrixTransform(svg.getScreenCTM()?.inverse());
    
    const currentAngle = Math.atan2(svgP.y - position.y, svgP.x - position.x) * 180 / Math.PI;
    let deltaAngle = currentAngle - lastAngleRef.current;

    // Handle angle wrapping
    if (deltaAngle > 180) deltaAngle -= 360;
    if (deltaAngle < -180) deltaAngle += 360;
    
    if (Math.abs(deltaAngle) > 0.1) { // Threshold to prevent tiny jitters
      onRotate(deltaAngle);
    }
    
    lastAngleRef.current = currentAngle;
  }, [position, onRotate]);
  
  const handleMouseUp = useCallback(() => {
    window.removeEventListener('mousemove', handleMouseMove);
    window.removeEventListener('mouseup', handleMouseUp);
    lastAngleRef.current = null;
    onRotateEnd();
  }, [handleMouseMove, onRotateEnd]);

  const handleMouseDown = useCallback((e: React.MouseEvent<SVGCircleElement>) => {
    e.stopPropagation();
    onRotateStart();
    
    const svg = e.currentTarget.ownerSVGElement;
    if (!svg) return;

    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const svgP = pt.matrixTransform(svg.getScreenCTM()?.inverse());
    
    lastAngleRef.current = Math.atan2(svgP.y - position.y, svgP.x - position.x) * 180 / Math.PI;

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
  }, [position, onRotateStart, handleMouseMove, handleMouseUp]);

  return (
    <g ref={wheelRef} transform={`translate(${position.x}, ${position.y})`}>
      <circle
        cx={0}
        cy={0}
        r={50}
        fill="rgba(8, 145, 178, 0.1)"
        stroke="rgba(8, 145, 178, 0.4)"
        strokeWidth={20}
        className="cursor-alias"
        onMouseDown={handleMouseDown}
      />
       <circle
        cx={0}
        cy={0}
        r={60}
        fill="transparent"
        stroke="rgba(8, 145, 178, 0.2)"
        strokeWidth={1}
        className="pointer-events-none"
      />
      <text
        x={0}
        y={5}
        textAnchor="middle"
        fill="white"
        fontSize="14"
        className="pointer-events-none select-none font-bold"
        stroke="#111827"
        strokeWidth="0.5"
        paintOrder="stroke"
      >⟳</text>
    </g>
  );
};