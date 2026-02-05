

// This file contains a static database of predefined poses for the mannequin.
// Each pose is represented as an object with an ID, category, name, source, and pose data.
// The pose data is stored as a compact string, which can be parsed into a Pose object.

// Note: The pose data strings have been updated to reflect the new part naming
// (e.g., 'w' for waist, 'c' for collar, 'lw'/'rw' for wrist/hand, 'la'/'ra' for ankle/foot),
// and the new proportional sizing and floor height.

import { SCALE_FACTOR } from './constants'; // Import SCALE_FACTOR
import { mirrorPose } from './utils/pose-mirror'; // Import mirrorPose

const scaleRoot = (rootString: string): string => {
  const parts = rootString.split(':');
  if (parts.length === 2 && parts[0] === 'r') {
    const [xStr, yStr] = parts[1].split(',');
    const x = parseFloat(xStr) * SCALE_FACTOR;
    const y = parseFloat(yStr) * SCALE_FACTOR;
    return `r:${x},${y}`;
  }
  return rootString;
};

const originalPoses = [
  // Updated T-Pose data string for new T_POSE_ROOT_Y
  { "id": "B01", "cat": "Base", "name": "T-Pose", "src": "Bitruvius", "data": scaleRoot("r:0,600") + ";br:0;w:0;t:0;c:0;h:0;ls:-90;le:0;lw:0;rs:90;re:0;rw:0;lt:0;lc:0;la:0;rt:0;rc:0;ra:0" },
  { "id": "A06", "cat": "Action", "name": "Shield", "src": "Manual", "data": scaleRoot("r:0,197.5") + ";br:0;w:0;t:0;c:0;h:187.81;rs:178.87;re:176.45;rw:62.24;ls:-178.18;le:177.87;lw:0;rt:-169.88;rc:180.62;ra:90;lt:-189.97;lc:179.19;la:90" },
  { "id": "D02", "cat": "Dance", "name": "Ballerina", "src": "Manual", "data": scaleRoot("r:0,385") + ";br:0;w:0;t:5;c:0;h:-10;ls:60;le:-45;lw:20;rs:-135;re:30;rw:-20;lt:-10;lc:100;la:90;rt:30;rc:0;ra:90" },
  { "id": "A08", "cat": "Action", "name": "Fly", "src": "User", "data": scaleRoot("r:0,197.5") + ";br:0;w:0;t:180.51;c:0;h:0;rs:-252.19;re:0;rw:0;ls:-108.15;le:0;lw:0;rt:0;rc:-179.21;ra:90;lt:0;lc:-179.23;la:90" },
  { "id": "S01", "cat": "Still", "name": "Tree Ornament", "src": "User", "data": scaleRoot("r:-67.04,95.43") + ";br:0;w:0;t:180.51;c:0;h:-184.08;rs:-252.19;re:71.74;rw:0;ls:-108.15;le:285.89;lw:0;rt:0;rc:-179.21;ra:90;lt:0;lc:-179.23;la:90" },
  { "id": "C01", "cat": "Character", "name": "Mustachioed", "src": "User", "data": scaleRoot("r:0,91.25") + ";br:0;w:0;t:180.56;c:0;h:0;rs:0;re:0;rw:0;ls:0;le:0;lw:0;rt:-80.31;rc:131.28;ra:90;lt:-275.29;lc:-132.04;la:90" },
  { "id": "C02", "cat": "Character", "name": "Lobster", "src": "User", "data": scaleRoot("r:0,91.25") + ";br:0;w:0;t:180.39;c:0;h:-190.78;rs:291.03;re:179.18;rw:0;ls:65.93;le:186.1;lw:0;rt:0;rc:0;ra:90;lt:0;lc:0;la:90" },
];

// Poses identified as "left-leaning" for mirroring
const posesToMirrorInfo = [
  { id: "A06", name: "Shield", cat: "Action" },
  { id: "D02", name: "Ballerina", cat: "Dance" },
  { id: "S01", name: "Tree Ornament", cat: "Still" },
];

const generatedMirroredPoses = posesToMirrorInfo.map(pInfo => {
  const originalPose = originalPoses.find(pose => pose.id === pInfo.id);
  if (!originalPose) {
    console.warn(`Original pose with ID ${pInfo.id} not found for mirroring.`);
    return null;
  }
  return mirrorPose(originalPose.data, pInfo.id, pInfo.name, pInfo.cat);
}).filter(Boolean); // Filter out any null entries if original pose wasn't found

export const POSE_LIBRARY_DB = [
  ...originalPoses,
  ...(generatedMirroredPoses as any), // Cast to any to safely spread (after filtering nulls)
];