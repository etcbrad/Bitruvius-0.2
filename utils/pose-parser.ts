

import { Pose, Vector2D } from '../types';

// Short keys for URL/file serialization
export const SHORT_KEY_MAP: { [long: string]: string } = {
  root: 'r',
  bodyRotation: 'br',
  waist: 'w',      // New short key for waist
  torso: 't',
  collar: 'c',     // New short key for collar
  head: 'h',
  lShoulder: 'ls',
  lForearm: 'le',
  lWrist: 'lw',    // Now represents the hand
  rShoulder: 'rs',
  rForearm: 're',
  rWrist: 'rw',    // Now represents the hand
  lThigh: 'lt',
  lCalf: 'lc',
  lAnkle: 'la',    // Now represents the foot
  rThigh: 'rt',
  rCalf: 'rc',
  rAnkle: 'ra',    // Now represents the foot
};

// Long keys for internal use (reverse lookup)
export const LONG_KEY_MAP: { [short: string]: string } = Object.fromEntries(
  Object.entries(SHORT_KEY_MAP).map(([long, short]) => [short, long])
);

// Helper to round numbers for cleaner serialization
const round = (num: number): number => parseFloat(num.toFixed(2));

// Converts a pose object into a compact string representation
export const poseToString = (pose: Pose): string => {
  if (!pose) return '';

  const parts: string[] = [];
  // Define an ordered list of keys for consistent serialization
  const orderedKeys: Array<keyof Pose | 'root'> = [
    'root', 'bodyRotation', 'waist', 'torso', 'collar', 'head',
    'rShoulder', 'rForearm', 'rWrist', 'lShoulder', 'lForearm', 'lWrist',
    'rThigh', 'rCalf', 'rAnkle', 'lThigh', 'lCalf', 'lAnkle'
  ];

  for (const key of orderedKeys) {
    // Skip if key is undefined in the pose (e.e.g., offsets)
    if (pose[key] === undefined) continue;

    const shortKey = SHORT_KEY_MAP[key.toString()]; // Convert keyof Pose to string for lookup
    if (!shortKey) continue; // Skip if no short key mapping exists

    if (key === 'root') {
      const { x, y } = pose.root;
      parts.push(`${shortKey}:${round(x)},${round(y)}`);
    } else {
      const value = pose[key as keyof Pose]; // Type assertion for non-root keys
      if (typeof value === 'number') {
        parts.push(`${shortKey}:${round(value)}`);
      }
    }
  }
  return parts.join(';');
};

// Converts a string representation back into a pose object
export const stringToPose = (str: string): Partial<Pose> => {
  const pose: Partial<Pose> = {};
  if (typeof str !== 'string' || str.trim() === '') return pose;

  try {
    const parts = str.split(';');
    for (const part of parts) {
      const [shortKey, valueStr] = part.split(':');
      if (!shortKey || valueStr === undefined) continue;

      const longKey = LONG_KEY_MAP[shortKey];
      if (!longKey) continue;

      if (longKey === 'root') {
        const [xStr, yStr] = valueStr.split(',');
        const x = parseFloat(xStr);
        const y = parseFloat(yStr);
        if (!isNaN(x) && !isNaN(y)) {
          pose.root = { x, y } as Vector2D;
        }
      } else {
        const value = parseFloat(valueStr);
        if (!isNaN(value)) {
          (pose as any)[longKey] = value; // Use 'any' or more specific type assertion for dynamic assignment
        }
      }
    }
  } catch (error) {
    console.error("Error parsing pose string:", error);
    return {}; // Return empty pose on parsing error
  }
  return pose;
};