import React, { useMemo, useCallback, useEffect } from 'react';
import { Bone } from './Bone';
import { ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT, RIGGING } from '../constants';
import { WalkingEnginePose, WalkingEngineProportions, WalkingEnginePivotOffsets, Vector2D, MaskTransform, GlobalPositions } from '../types';
import { getScaledDimension as getKinematicDimension, getShortestAngleDiffDeg, deg, distance, lerpAngleShortestPath } from '../utils/kinematics';

interface MannequinProps {
  pose: WalkingEnginePose;
  pivotOffsets: WalkingEnginePivotOffsets;
  props: WalkingEngineProportions;
  showPivots: boolean;
  showLabels: boolean;
  baseUnitH: number;
  onAnchorMouseDown: (boneKey: keyof WalkingEnginePivotOffsets, clientX: number, e: React.MouseEvent | React.TouchEvent) => void;
  onBodyMouseDown: (boneKey: keyof WalkingEnginePivotOffsets, clientX: number, e: React.MouseEvent | React.TouchEvent) => void;
  draggingBoneKey: keyof WalkingEnginePivotOffsets | null;
  selectedBoneKeys: Set<keyof WalkingEnginePivotOffsets>;
  isPaused: boolean;
  maskImage?: string | null;
  maskTransform?: MaskTransform;
  isGhost?: boolean;
  overrideProps?: WalkingEngineProportions;
  onPositionsUpdate?: (positions: GlobalPositions) => void;
  activePins?: (keyof WalkingEnginePivotOffsets)[];
  onIKSolveUpdate?: (updates: Partial<WalkingEnginePivotOffsets>) => void;
  ghostType?: 'ik' | 'fk' | 'static' | null;
  ghostOpacity?: number;
  limbTensions?: Record<string, number>;
  pinsAtLimit?: Set<keyof WalkingEnginePivotOffsets>;
  isInteractive?: boolean;
  onClick?: () => void;
}

export const RENDER_ORDER: (keyof WalkingEngineProportions)[] = [
    'waist', 'torso', 'l_upper_leg', 'r_upper_leg', 'l_lower_leg', 'r_lower_leg', 'l_foot', 'r_foot', 'l_toe', 'r_toe', 
    'collar', 'head', 'l_upper_arm', 'r_upper_arm', 'l_lower_arm', 'r_lower_arm', 'l_hand', 'r_hand'
];

export const partDefinitions: Record<keyof WalkingEngineProportions, any> = {
    head: { rawH: ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.HEAD, rawW: ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.HEAD_WIDTH, variant: 'head-wedge', drawsUpwards: true, label: 'Head', boneKey: 'neck' },
    collar: { rawH: ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.COLLAR, rawW: ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.COLLAR_WIDTH, variant: 'collar-horizontal-oval-shape', drawsUpwards: true, label: 'Collar', boneKey: 'collar' },
    torso: { rawH: ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.TORSO, rawW: ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.TORSO_WIDTH, variant: 'torso-teardrop-pointy-down', drawsUpwards: true, label: 'Torso', boneKey: 'torso' },
    waist: { rawH: ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.WAIST, rawW: ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.WAIST_WIDTH, variant: 'waist-teardrop-pointy-up', drawsUpwards: true, label: 'Waist', boneKey: 'waist' },
    r_upper_arm: { rawH: ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.UPPER_ARM, rawW: ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.LIMB_WIDTH_ARM, variant: 'deltoid-shape', label: 'R.Bicep', boneKey: 'r_shoulder' },
    r_lower_arm: { rawH: ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.LOWER_ARM, rawW: ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.LIMB_WIDTH_FOREARM, variant: 'limb-tapered', label: 'R.Forearm', boneKey: 'r_elbow' },
    r_hand: { rawH: ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.HAND, rawW: ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.HAND_WIDTH, variant: 'hand-foot-arrowhead-shape', label: 'R.Hand', boneKey: 'r_hand' },
    l_upper_arm: { rawH: ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.UPPER_ARM, rawW: ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.LIMB_WIDTH_ARM, variant: 'deltoid-shape', label: 'L.Bicep', boneKey: 'l_shoulder' },
    l_lower_arm: { rawH: ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.LOWER_ARM, rawW: ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.LIMB_WIDTH_FOREARM, variant: 'limb-tapered', label: 'L.Forearm', boneKey: 'l_elbow' },
    l_hand: { rawH: ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.HAND, rawW: ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.HAND_WIDTH, variant: 'hand-foot-arrowhead-shape', label: 'L.Hand', boneKey: 'l_hand' },
    r_upper_leg: { rawH: ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.LEG_UPPER, rawW: ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.LIMB_WIDTH_THIGH, variant: 'limb-tapered', label: 'R.Thigh', boneKey: 'r_hip', tensionKey: 'r_leg' },
    r_lower_leg: { rawH: ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.LEG_LOWER, rawW: ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.LIMB_WIDTH_CALF, variant: 'limb-tapered', label: 'R.Calf', boneKey: 'r_knee', tensionKey: 'r_leg' },
    r_foot: { rawH: ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.FOOT, rawW: ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.FOOT_WIDTH, variant: 'foot-block-shape', label: 'R.Foot', boneKey: 'r_foot', tensionKey: 'r_leg', pinKey: 'r_foot' },
    r_toe: { rawH: ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.TOE, rawW: ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.TOE_WIDTH, variant: 'toe-rounded-cap', label: 'R.Toe', boneKey: 'r_toe' },
    l_upper_leg: { rawH: ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.LEG_UPPER, rawW: ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.LIMB_WIDTH_THIGH, variant: 'limb-tapered', label: 'L.Thigh', boneKey: 'l_hip', tensionKey: 'l_leg' },
    l_lower_leg: { rawH: ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.LEG_LOWER, rawW: ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.LIMB_WIDTH_CALF, variant: 'limb-tapered', label: 'L.Calf', boneKey: 'l_knee', tensionKey: 'l_leg' },
    l_foot: { rawH: ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.FOOT, rawW: ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.FOOT_WIDTH, variant: 'foot-block-shape', label: 'L.Foot', boneKey: 'l_foot', tensionKey: 'l_leg', pinKey: 'l_foot' },
    l_toe: { rawH: ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.TOE, rawW: ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.TOE_WIDTH, variant: 'toe-rounded-cap', label: 'L.Toe', boneKey: 'l_toe' },
};

const rotateVec = (vec: Vector2D, angleDeg: number): Vector2D => {
  const r = angleDeg * Math.PI / 180;
  const c = Math.cos(r);
  const s = Math.sin(r);
  return { x: vec.x * c - vec.y * s, y: vec.y * c + vec.x * s };
};
const addVec = (v1: Vector2D, v2: Vector2D): Vector2D => ({ x: v1.x + v2.x, y: v1.y + v2.y });

interface MannequinCalculations {
    transforms: GlobalPositions;
    finalProps: WalkingEngineProportions;
}

export const Mannequin: React.FC<MannequinProps> = ({
  pose, pivotOffsets, props, showPivots, showLabels, baseUnitH,
  onAnchorMouseDown, onBodyMouseDown, draggingBoneKey, selectedBoneKeys, isPaused,
  maskImage, maskTransform, isGhost = false, overrideProps, onPositionsUpdate, activePins = [],
  onIKSolveUpdate, ghostType, ghostOpacity = 0.6, limbTensions = {}, pinsAtLimit,
  isInteractive = false, onClick
}) => {
    const activeProps = useMemo(() => {
        return overrideProps ? overrideProps : JSON.parse(JSON.stringify(props));
    }, [props, overrideProps]);
    

    const getScaledDimension = useCallback((raw: number, key: keyof WalkingEngineProportions, axis: 'w' | 'h', currentProps: WalkingEngineProportions) => {
        return getKinematicDimension(raw, baseUnitH, currentProps, key, axis);
    }, [baseUnitH]);

    const calculations: MannequinCalculations = useMemo(() => {
        const trans: GlobalPositions = {};
        const currentRenderPivotOffsets = { ...pivotOffsets }; 
        const currentCalculatedProps: WalkingEngineProportions = JSON.parse(JSON.stringify(activeProps));
        const getRot = (key: keyof WalkingEnginePivotOffsets) => ((pose as any)[key] || 0) + ((currentRenderPivotOffsets as any)[key] || 0);

        const waistLen = getScaledDimension(ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.WAIST, 'waist', 'h', currentCalculatedProps);
        const torsoLen = getScaledDimension(ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.TORSO, 'torso', 'h', currentCalculatedProps);
        const collarLen = getScaledDimension(ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.COLLAR, 'collar', 'h', currentCalculatedProps);
        
        const actualWaistPos: Vector2D = { x: 0, y: 0 }; 
        const waistRot = getRot('waist');
        trans.waist = { position: actualWaistPos, rotation: waistRot };

        let torsoRot = waistRot + getRot('torso');
        trans.torso = { position: addVec(actualWaistPos, rotateVec({ x: 0, y: -waistLen }, waistRot)), rotation: torsoRot };

        let collarRot = torsoRot + getRot('collar');
        trans.collar = { position: addVec(trans.torso.position, rotateVec({ x: 0, y: -torsoLen }, torsoRot)), rotation: collarRot };
        
        let neckRot = collarRot + getRot('neck');
        trans.head = { position: addVec(trans.collar.position, rotateVec({ x: 0, y: -collarLen }, collarRot)), rotation: neckRot };

        ['r', 'l'].forEach(side => {
            const upArmKey = `${side}_upper_arm` as 'r_upper_arm' | 'l_upper_arm';
            const lowArmKey = `${side}_lower_arm` as 'r_lower_arm' | 'l_lower_arm';
            const handKey = `${side}_hand` as 'r_hand' | 'l_hand';
            const shoulderPivotKey = `${side}_shoulder` as 'r_shoulder' | 'l_shoulder';
            const elbowPivotKey = `${side}_elbow` as 'r_elbow' | 'l_elbow';
            const handPivotKey = `${side}_hand` as 'r_hand' | 'l_hand';

            const sx = (side === 'r' ? RIGGING.R_SHOULDER_X_OFFSET_FROM_COLLAR_CENTER : RIGGING.L_SHOULDER_X_OFFSET_FROM_COLLAR_CENTER) * baseUnitH;
            const collarEndPoint = addVec(trans.collar!.position, rotateVec({ x: 0, y: -collarLen }, collarRot));
            const shoulderRootPos = addVec(collarEndPoint, rotateVec({ x: sx, y: 0 }, collarRot));
            
            const upArmLen = getScaledDimension(ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.UPPER_ARM, upArmKey, 'h', currentCalculatedProps);
            const lowArmLen = getScaledDimension(ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.LOWER_ARM, lowArmKey, 'h', currentCalculatedProps);

            const shRot = collarRot + getRot(shoulderPivotKey); 
            const elRot = shRot + getRot(elbowPivotKey);
            const wristRot = elRot + getRot(handPivotKey);

            trans[upArmKey] = { position: shoulderRootPos, rotation: shRot };
            const elbowPos = addVec(shoulderRootPos, rotateVec({ x: 0, y: upArmLen }, shRot));
            trans[lowArmKey] = { position: elbowPos, rotation: elRot };
            const wristPos = addVec(elbowPos, rotateVec({ x: 0, y: lowArmLen }, elRot));
            trans[handKey] = { position: wristPos, rotation: wristRot };
        });

        ['r', 'l'].forEach(side => {
            const upLegKey = `${side}_upper_leg` as 'r_upper_leg' | 'l_upper_leg';
            const lowLegKey = `${side}_lower_leg` as 'r_lower_leg' | 'l_lower_leg';
            const footKey = `${side}_foot` as 'r_foot' | 'l_foot';
            const toeKey = `${side}_toe` as 'r_toe' | 'l_toe';
            const hipPivotKey = `${side}_hip` as 'r_hip' | 'l_hip';
            const kneePivotKey = `${side}_knee` as 'r_knee' | 'l_knee';
            const footPivotKey = `${side}_foot` as 'r_foot' | 'l_foot';
            const toePivotKey = `${side}_toe` as 'r_toe' | 'l_toe';

            const hipRootPos = actualWaistPos; 
            const thighLen = getScaledDimension(ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.LEG_UPPER, upLegKey, 'h', currentCalculatedProps);
            const calfLen = getScaledDimension(ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.LEG_LOWER, lowLegKey, 'h', currentCalculatedProps);
            const footLen = getScaledDimension(ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.FOOT, footKey, 'h', currentCalculatedProps);
            
            const parentGlobalAngleForLegs = waistRot;
            
            const hipRot = parentGlobalAngleForLegs + getRot(hipPivotKey); 
            const kneeRot = hipRot + getRot(kneePivotKey);
            const ankleRot = kneeRot + getRot(footPivotKey);
            const toeRot = ankleRot + getRot(toePivotKey);

            trans[upLegKey] = { position: hipRootPos, rotation: hipRot };
            const kneePos = addVec(hipRootPos, rotateVec({ x: 0, y: thighLen }, hipRot));
            trans[lowLegKey] = { position: kneePos, rotation: kneeRot };
            const anklePos = addVec(kneePos, rotateVec({ x: 0, y: calfLen }, kneeRot));
            trans[footKey] = { position: anklePos, rotation: ankleRot };
            const toePos = addVec(anklePos, rotateVec({ x: 0, y: footLen }, ankleRot));
            trans[toeKey] = { position: toePos, rotation: toeRot };
        });

        return { transforms: trans, finalProps: currentCalculatedProps };
    }, [pose, pivotOffsets, getScaledDimension, baseUnitH, activeProps]);

    useEffect(() => {
        if (onPositionsUpdate) {
            onPositionsUpdate(calculations.transforms);
        }
    }, [calculations.transforms, onPositionsUpdate]);
    
    const wrapperStyle: React.CSSProperties = {
      opacity: isGhost ? ghostOpacity : 1,
      pointerEvents: isInteractive ? 'auto' : (isGhost ? 'none' : 'auto'),
      cursor: isInteractive ? 'pointer' : 'default'
    };


    return (
        <g style={wrapperStyle} onClick={onClick}>
            {RENDER_ORDER.map(partKey => {
                const p = partDefinitions[partKey];
                const t = calculations.transforms[partKey];
                if (!p || !t) return null;

                let colorClass: string;
                if (isGhost) {
                    colorClass = "fill-none";
                } else if (partKey === 'collar') {
                    colorClass = 'fill-olive';
                } else {
                    colorClass = 'fill-mono-dark';
                }

                const headH = getScaledDimension(ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT.HEAD, 'head', 'h', calculations.finalProps);
                const isUnderTension = p.tensionKey ? (limbTensions[p.tensionKey] || 1) > 1.05 : false;
                const tension = p.tensionKey ? (limbTensions[p.tensionKey] || 1.0) : 1.0;
                const isAtLimit = p.pinKey ? pinsAtLimit?.has(p.pinKey) : false;

                return (
                    <g key={partKey} transform={`translate(${t.position.x}, ${t.position.y}) rotate(${t.rotation})`}>
                        <Bone 
                            rotation={0}
                            length={getScaledDimension(p.rawH, partKey, 'h', calculations.finalProps)}
                            width={getScaledDimension(p.rawW, partKey, 'w', calculations.finalProps)}
                            variant={p.variant}
                            drawsUpwards={p.drawsUpwards}
                            label={p.label}
                            boneKey={p.boneKey}
                            proportionKey={partKey}
                            showPivots={showPivots && !isGhost}
                            showLabel={showLabels && !isGhost}
                            onAnchorMouseDown={onAnchorMouseDown}
                            onBodyMouseDown={onBodyMouseDown}
                            isBeingDragged={!isGhost && draggingBoneKey === p.boneKey}
                            isSelected={p.boneKey ? selectedBoneKeys.has(p.boneKey) : false}
                            isPausedAndPivotsVisible={true} 
                            colorClass={colorClass}
                            isPinned={!isGhost && activePins.includes(p.pinKey)}
                            isGhost={isGhost}
                            ghostType={ghostType}
                            isUnderTension={isUnderTension}
                            tension={tension}
                            isAtLimit={isAtLimit}
                        />
                        {partKey === 'head' && maskImage && maskTransform && !isGhost && (
                          <g transform={`translate(${maskTransform.x}, ${maskTransform.y - headH/2}) rotate(${maskTransform.rotation}) scale(${maskTransform.scale})`}>
                            <image 
                              href={maskImage} 
                              x="-50" y="-50" width="100" height="100" 
                              preserveAspectRatio="xMidYMid meet"
                              className="pointer-events-none drop-shadow-lg"
                            />
                          </g>
                        )}
                    </g>
                );
            })}
        </g>
    );
};