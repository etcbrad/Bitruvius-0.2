import { WalkingEnginePivotOffsets } from '../types';

// Placeholder poses for the semantic command parser
const LUNGE_HEAVY: WalkingEnginePivotOffsets = {
    waist: 20, torso: 15, collar: 0, neck: -10,
    l_shoulder: -60, l_elbow: 45, l_hand: 0,
    r_shoulder: 120, r_elbow: 30, r_hand: 0,
    l_hip: -90, l_knee: 90, l_foot: 0, l_toe: 0,
    r_hip: 45, r_knee: -15, r_foot: 0, r_toe: 0,
};

const CROUCH_LIGHT: WalkingEnginePivotOffsets = {
    waist: 10, torso: 5, collar: 0, neck: 0,
    l_shoulder: -45, l_elbow: 30, l_hand: 0,
    r_shoulder: 45, r_elbow: 30, r_hand: 0,
    l_hip: -45, l_knee: 60, l_foot: -10, l_toe: 0,
    // FIX: Corrected duplicate 'l_toe' to 'r_toe' to fix both missing property and duplicate key errors.
    r_hip: -45, r_knee: 60, r_foot: -10, r_toe: 0,
};

const REACH_HIGH: WalkingEnginePivotOffsets = {
    waist: -10, torso: -15, collar: 0, neck: -10,
    l_shoulder: -160, l_elbow: -15, l_hand: 0,
    r_shoulder: 160, r_elbow: -15, r_hand: 0,
    l_hip: 0, l_knee: 0, l_foot: 20, l_toe: 30,
    r_hip: 0, r_knee: 0, r_foot: 20, r_toe: 30,
};

const GUARD_LEFT: WalkingEnginePivotOffsets = {
    waist: -30, torso: -10, collar: 0, neck: 20,
    l_shoulder: -90, l_elbow: 120, l_hand: 0,
    r_shoulder: -20, r_elbow: 90, r_hand: 0,
    l_hip: -30, l_knee: 45, l_foot: 0, l_toe: 0,
    r_hip: 15, r_knee: 15, r_foot: 0, r_toe: 0,
};

const IDLE_NEUTRAL: WalkingEnginePivotOffsets = {
    waist: 0, neck: 0, collar: 0, torso: 0,
    l_shoulder: -75, r_shoulder: 75,
    l_elbow: 0, r_elbow: 0,
    l_hand: 0, r_hand: 0,
    l_hip: 0, r_hip: 0,
    l_knee: 0, r_knee: 0,
    l_foot: 0, r_foot: 0,
    l_toe: 0, r_toe: 0
};

const commandMap: Record<string, WalkingEnginePivotOffsets> = {
    'LUNGE_HEAVY': LUNGE_HEAVY,
    'CROUCH_LIGHT': CROUCH_LIGHT,
    'REACH_HIGH': REACH_HIGH,
    'GUARD_LEFT': GUARD_LEFT,
    'IDLE_NEUTRAL': IDLE_NEUTRAL,
};

export const parseSemanticCommand = (command: string): WalkingEnginePivotOffsets | null => {
    const key = command.trim().toUpperCase();
    return commandMap[key] || null;
};

export const getAvailableCommands = (): string[] => {
    return Object.keys(commandMap);
};
