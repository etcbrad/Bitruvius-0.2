
// utils/pose-mirror.ts

import { Pose, PartName, partNameToPoseKey } from '../types';
import { stringToPose, poseToString } from './pose-parser';

export const mirrorPose = (originalPoseString: string, originalId: string, originalName: string, originalCategory: string): { id: string; cat: string; name: string; src: string; data: string } => {
  const pose: Partial<Pose> = stringToPose(originalPoseString);
  const mirrored: Partial<Pose> = { ...pose };

  // Mirror root X position
  if (mirrored.root) {
    mirrored.root = { x: -mirrored.root.x, y: mirrored.root.y };
  }

  // Mirror body rotation
  if (typeof mirrored.bodyRotation === 'number') {
    mirrored.bodyRotation = -mirrored.bodyRotation;
  }

  // Swap and mirror limb rotations
  const swapAndMirror = (leftPart: PartName, rightPart: PartName) => {
    const leftKey = partNameToPoseKey[leftPart];
    const rightKey = partNameToPoseKey[rightPart];

    const leftRot = (pose as any)[leftKey];
    const rightRot = (pose as any)[rightKey];

    // Assign mirrored right rotation from original left rotation
    if (typeof leftRot === 'number') {
      (mirrored as any)[rightKey] = -leftRot;
    } else {
      delete (mirrored as any)[rightKey]; 
    }

    // Assign mirrored left rotation from original right rotation
    if (typeof rightRot === 'number') {
      (mirrored as any)[leftKey] = -rightRot;
    } else {
      delete (mirrored as any)[leftKey];
    }
  };

  swapAndMirror(PartName.LShoulder, PartName.RShoulder);
  // Using partNameToPoseKey for elbow/forearm and shin/calf to map correctly
  swapAndMirror(PartName.LElbow, PartName.RElbow);
  swapAndMirror(PartName.LWrist, PartName.RWrist);
  swapAndMirror(PartName.LThigh, PartName.RThigh);
  swapAndMirror(PartName.LSkin, PartName.RSkin);
  swapAndMirror(PartName.LAnkle, PartName.RAnkle);

  // For central parts like Waist, Torso, Collar, Head, their rotations are generally
  // relative to their parent and don't typically need to be inverted for mirroring,
  // as the `bodyRotation` handles the overall flip. If specific asymmetric twists
  // were intended to be mirrored, more complex logic would be needed here.

  return {
    id: `${originalId}_R`,
    cat: originalCategory, // Keep original category, or make it 'Mirrored'
    name: `RIGHT ${originalName}`, // Prepend "RIGHT" to the original name
    src: "Bitruvius Generated",
    data: poseToString(mirrored as Pose),
  };
};