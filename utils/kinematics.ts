import { BASE_ROTATIONS, RIGGING } from '../constants';
import { PartName, Pose, Vector2D, AnchorName, WalkingEngineProportions } from '../types';

export const lerp = (start: number, end: number, t: number): number => start * (1 - t) + end * t;

// Added for smoother, robotic transitions
export const easeInOutQuint = (t: number): number => {
  return t < 0.5 ? 16 * t * t * t * t * t : 1 - Math.pow(-2 * t + 2, 5) / 2;
};

// Added for physically plausible impact simulation
export const easeInQuint = (t: number): number => t * t * t * t * t;


// This function calculates the shortest angular difference between two angles (in degrees).
// It's robust for angles in any range, including those outside [-180, 180].
export const getShortestAngleDiffDeg = (currentDeg: number, startDeg: number): number => {
  let diff = currentDeg - startDeg;

  // Normalize diff to [0, 360)
  diff = ((diff % 360) + 360) % 360; 
  
  // Then, adjust to [-180, 180]
  if (diff > 180) {
    diff -= 360;
  }
  return diff;
};

// NOTE: This function is currently not used in App.tsx for direct drag updates.
// It would be used for interpolating between two full poses over time.
export const lerpAngleShortestPath = (a: number, b: number, t: number): number => {
  // Use 'a' and 'b' directly for interpolation, but calculate shortest difference based on normalized angles.
  // The 'return a + ...' part needs 'a' as the starting point.

  // Normalize angles to [0, 360) for consistent difference calculation
  const normalizeAngle0to360 = (angle: number): number => {
    return ((angle % 360) + 360) % 360;
  };

  let startAngle = normalizeAngle0to360(a);
  let endAngle = normalizeAngle0to360(b);

  let delta = endAngle - startAngle;

  // Adjust delta to be within [-180, 180] for a shortest path
  if (delta > 180) {
    delta -= 360;
  } else if (delta < -180) {
    delta += 360;
  }
  
  // Apply this shortest delta from the original 'a'
  return a + delta * t;
};

const rad = (deg: number): number => deg * Math.PI / 180;
export const deg = (rad: number): number => rad * 180 / Math.PI; // Exported for Mannequin
const rotateVec = (x: number, y: number, angleDeg: number): Vector2D => {
  const r = rad(angleDeg);
  const c = Math.cos(r);
  const s = Math.sin(r);
  // Corrected Y calculation
  return { x: x * c - y * s, y: x * s + y * c }; 
};
const addVec = (v1: Vector2D, v2: Vector2D): Vector2D => ({ x: v1.x + v2.x, y: v1.y + v2.y });
export const distance = (v1: Vector2D, v2: Vector2D): number => {
  const dx = v2.x - v1.x;
  const dy = v2.y - v1.y;
  return Math.sqrt(dx * dx + dy * dy);
};


export const getTotalRotation = (key: string, pose: Pose): number => (BASE_ROTATIONS[key as keyof typeof BASE_ROTATIONS] || 0) + ((pose as any)[key] || 0);

/**
 * Helper to get scaled dimension based on raw value, base unit, and proportion state.
 */
export const getScaledDimension = (raw: number, baseUnitH: number, props: WalkingEngineProportions, key: keyof WalkingEngineProportions, axis: 'w' | 'h') => {
  return raw * baseUnitH * (props[key]?.[axis] || 1);
};


/**
 * Solves a 2-bone inverse kinematics problem.
 * @param target The world-space position the end-effector should reach.
 * @param root The world-space position of the chain's root joint.
 * @param len1 Length of the first bone (e.g., thigh).
 * @param len2 Length of the second bone (e.g., calf).
 * @param parentAngle The world-space angle of the parent of the root joint.
 * @param isLeft A hint to determine the "bend" direction (e.g., for knees/elbows).
 * @returns An object with angle1 (for root joint) and angle2 (for middle joint) in degrees, and a stretch factor.
 */
export const solveTwoBoneIK = (
    target: Vector2D,
    root: Vector2D,
    len1: number,
    len2: number,
    parentAngle: number,
    isLeft: boolean
): { angle1: number, angle2: number, stretch: number } | null => {
    const dist = distance(root, target);
    const maxDist = len1 + len2;
    let stretch = 1.0;
    let effectiveTarget = { ...target };
    let effectiveDist = dist;

    // Safeguard: If target is unreachable, calculate stretch and clamp target.
    if (dist > maxDist) {
        stretch = dist / maxDist;
        const ratio = maxDist / dist;
        effectiveTarget.x = root.x + (target.x - root.x) * ratio;
        effectiveTarget.y = root.y + (target.y - root.y) * ratio;
        effectiveDist = maxDist;
    }

    // Law of Cosines to find the angle of the second joint (knee/elbow).
    const cosAngle2 = (effectiveDist * effectiveDist - len1 * len1 - len2 * len2) / (2 * len1 * len2);
    
    // Clamp the value to prevent Math.acos from returning NaN due to floating point inaccuracies
    const clampedCosAngle2 = Math.max(-1, Math.min(1, cosAngle2));
    let angle2 = Math.acos(clampedCosAngle2);
    
    // Determine bend direction
    if (isLeft) { 
        // Left limbs bend counter-clockwise in our system
        angle2 = angle2;
    } else {
        // Right limbs bend clockwise
        angle2 = -angle2;
    }

    // Find the angle of the first joint (hip/shoulder).
    const angle_root_to_target = Math.atan2(effectiveTarget.y - root.y, effectiveTarget.x - root.x);
    const angle1_part2 = Math.atan2(len2 * Math.sin(angle2), len1 + len2 * Math.cos(angle2));
    let angle1 = angle_root_to_target - angle1_part2;
    
    // Convert world-space radians to local-space degrees
    const worldAngle1 = deg(angle1);
    
    const localAngle1 = worldAngle1 - 90 - parentAngle;
    const localAngle2 = deg(angle2);
    
    return {
        angle1: localAngle1,
        angle2: localAngle2,
        stretch: stretch,
    };
};