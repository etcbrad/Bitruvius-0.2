
import { PartName, Pose, Vector2D, JointLimits, WalkingEnginePivotOffsets } from './types';

export const SCALE_FACTOR = 1; // This remains for overall SVG viewport scaling to maintain visual size.
export const BASE_HEAD_UNIT = 50; // The base unit for the internal anatomy proportions. 'H' from walking engine will substitute this.

// ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT: These are the *proportions* of each part
// relative to a single BASE_HEAD_UNIT (or the H from the walking engine).
export const ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT = {
  HEAD: 0.75, // Used for head.h - Adjusted to 3/4 height
  HEAD_WIDTH: (2 / 3) * 0.5, // Used for head.w - Top of head is as wide as top of collar. Collar top is 0.5 of its base.
  HEAD_NECK_GAP_OFFSET: 0.1,
  COLLAR: 0.5, // Used for collar.h - Adjusted to 0.5 for more human proportion
  COLLAR_WIDTH: (2 / 3), // Used for collar.w
  TORSO: 1.2, // Used for torso.h
  TORSO_WIDTH: 0.65, // Used for torso.w
  WAIST: 1.0, // Used for waist.h
  WAIST_WIDTH: 0.85, // Used for waist.w
  UPPER_ARM: 1.8, // Used for l_upper_arm.h, r_upper_arm.h
  LOWER_ARM: 1.4, // Used for l_lower_arm.h, r_lower_arm.h
  HAND: 0.8, // Used for l_hand.h, r_hand.h
  LEG_UPPER: 2.2, // Used for l_upper_leg.h, r_upper_leg.h
  LEG_LOWER: 1.8, // Used for l_lower_leg.h, r_lower_leg.h
  FOOT: 0.64, // Used for l_foot.h, r_foot.h (Ankle to ball of foot) - Adjusted for proportion
  TOE: 0.36, // Used for l_toe.h, r_toe.h (Ball of foot to toe tip) - Adjusted for proportion
  SHOULDER_WIDTH: 1.2, // Not directly used in props, but informs RIGGING
  HIP_WIDTH: 1.0, // Not directly used in props
  ROOT_SIZE: 0.25,
  LIMB_WIDTH_ARM: 0.22, // Used for l_upper_arm.w, r_upper_arm.w
  LIMB_WIDTH_FOREARM: 0.18, // Used for l_lower_arm.w, r_lower_arm.w
  LIMB_WIDTH_THIGH: 0.35, // Used for l_upper_leg.w, r_upper_leg.w
  LIMB_WIDTH_CALF: 0.28, // Used for l_lower_leg.w, r_lower_leg.w
  HAND_WIDTH: 0.2, // Used for l_hand.w, r_hand.w
  FOOT_WIDTH: 0.25, // Used for l_foot.w, r_foot.w
  TOE_WIDTH: 0.25, // Used for l_toe.w, r_toe.w
  EFFECTOR_WIDTH: 0.15,
};

// RIGGING values now reference ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT values directly.
export const RIGGING = {
  L_SHOULDER_X_OFFSET_FROM_COLLAR_CENTER: -ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.COLLAR_WIDTH / 2.1,
  R_SHOULDER_X_OFFSET_FROM_COLLAR_CENTER: ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.COLLAR_WIDTH / 2.1,
  SHOULDER_Y_OFFSET_FROM_COLLAR_END: ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.COLLAR,
  COLLAR_OFFSET_Y: ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.COLLAR * 0.15,
};

export const JOINT_KEYS: (keyof WalkingEnginePivotOffsets)[] = [
  'waist', 'torso', 'collar', 'neck',
  'l_shoulder', 'l_elbow', 'l_hand',
  'r_shoulder', 'r_elbow', 'r_hand',
  'l_hip', 'l_knee', 'l_foot', 'l_toe',
  'r_hip', 'r_knee', 'r_foot', 'r_toe'
];


// The actual Y position for the floor in the mannequin's local coordinate system (before overall translation).
// This is the sum of leg segments below the hip.
export const MANNEQUIN_LOCAL_FLOOR_Y = 
    ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.LEG_UPPER + 
    ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.LEG_LOWER;

export const GROUND_STRIP_HEIGHT_RAW_H_UNIT = 0.4; // Height of the visual ground strip in H units
export const GROUND_STRIP_COLOR = '#252525'; // Slightly lighter very dark grey for the ground strip


// BASE_ROTATIONS, RESET_POSE, and JOINT_LIMITS are no longer directly used by the walking engine's physics.
// Keeping them for potential future reference or if other parts of the system still rely on the Pose type structure.
type RotationValues = Omit<Pose, 'root' | 'offsets'>;

export const BASE_ROTATIONS: RotationValues = {
  bodyRotation: 0,
  torso: 0,
  waist: 0,
  collar: 0,
  head: 0,
  lShoulder: 0,
  lForearm: 0,
  lWrist: 0,
  rShoulder: 0,
  rForearm: 0,
  rWrist: 0,
  lThigh: 0,
  lCalf: 0,
  lAnkle: 0,
  rThigh: 0,
  rCalf: 0,
  rAnkle: 0,
};

export const RESET_POSE: Pose = {
  root: { x: 0, y: 0 }, 
  ...BASE_ROTATIONS,
  offsets: {
    [PartName.Collar]: {x: 0, y: RIGGING.COLLAR_OFFSET_Y}
  },
};

// Define joint rotation limits in degrees
export const JOINT_LIMITS: JointLimits = {
  // Spine (relative to parent)
  [PartName.Waist]: { min: -180, max: 180 }, 
  [PartName.Torso]: { min: -180, max: 180 },
  [PartName.Collar]: { min: -180, max: 180 },
  [PartName.Head]: { min: -180, max: 180 },

  // Right Arm (relative to parent)
  [PartName.RShoulder]: { min: -180, max: 180 }, 
  rForearm: { min: -180, max: 180 },         
  [PartName.RWrist]: { min: -180, max: 180 }, 

  // Left Arm (relative to parent)
  [PartName.LShoulder]: { min: -180, max: 180 }, 
  lForearm: { min: -180, max: 180 },          
  [PartName.LWrist]: { min: -180, max: 180 }, 

  // Right Leg (relative to parent)
  [PartName.RThigh]: { min: -180, max: 180 }, 
  rCalf: { min: -180, max: 180 },           
  [PartName.RAnkle]: { min: -180, max: 180 }, 
  // Left Leg (relative to parent)
  [PartName.LThigh]: { min: -180, max: 180 },
  lCalf: { min: -180, max: 180 },
  lAnkle: { min: -180, max: 180 },
};