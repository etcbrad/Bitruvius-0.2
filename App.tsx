import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { WalkingEnginePose, WalkingEnginePivotOffsets, WalkingEngineProportions, Vector2D, MaskTransform, GlobalPositions, PhysicsState, JointChainBehaviors, Keyframe } from './types';
import { ANATOMY_RAW_RELATIVE_TO_BASE_HEAD_UNIT, RIGGING, JOINT_KEYS } from './constants'; 
import { Mannequin, partDefinitions } from './components/Mannequin';
import { SystemLogger } from './components/SystemLogger';
import { Timeline } from './components/Timeline';
import { KeymapHelper } from './components/KeymapHelper';
import { getScaledDimension as getKinematicDimension, lerpAngleShortestPath, distance, solveTwoBoneIK } from './utils/kinematics';
import { useAnimationEngine } from './hooks/useAnimationEngine';
import * as gifenc from 'gifenc';

const T_POSE: WalkingEnginePivotOffsets = {
  waist: 0, neck: 0, collar: 0, torso: 0,
  l_shoulder: 0, r_shoulder: 0,
  l_elbow: 0, r_elbow: 0,
  l_hand: 0, r_hand: 0,
  l_hip: 0, r_hip: 0,
  l_knee: 0, r_knee: 0,
  l_foot: 0, r_foot: 0,
  l_toe: 0, r_toe: 0
};

const INITIAL_CHALLENGE_POSE: WalkingEnginePivotOffsets = {
  waist: 180, torso: 180, collar: 0, neck: 180,
  l_shoulder: -95, l_elbow: 180, l_hand: 180,
  r_shoulder: 95, r_elbow: 180, r_hand: 180,
  l_hip: 5, l_knee: 180, l_foot: 180, l_toe: 180,
  r_hip: -5, r_knee: 180, r_foot: 180, r_toe: 180
};

const RESTING_BASE_POSE: WalkingEnginePose = {
  waist: 0, neck: 0, collar: 0, torso: 0, 
  l_shoulder: 0, r_shoulder: 0, l_elbow: 0, r_elbow: 0, l_hand: 0, r_hand: 0, 
  l_hip: 0, r_hip: 0, l_knee: 0, r_knee: 0, l_foot: 0, r_foot: 0, l_toe: 0, r_toe: 0, 
  stride_phase: 0, y_offset: 0, x_offset: 0
};

const PROP_KEYS: (keyof WalkingEngineProportions)[] = [
  'head', 'collar', 'torso', 'waist',
  'l_upper_arm', 'l_lower_arm', 'l_hand',
  'r_upper_arm', 'r_lower_arm', 'r_hand',
  'l_upper_leg', 'l_lower_leg', 'l_foot', 'l_toe',
  'r_upper_leg', 'r_lower_leg', 'r_foot', 'r_toe'
];

const ATOMIC_PROPS = Object.fromEntries(PROP_KEYS.map(k => [k, { w: 1, h: 1 }])) as WalkingEngineProportions;

const PIVOT_TO_PART_MAP: Record<keyof WalkingEnginePivotOffsets, keyof WalkingEngineProportions> = {
  waist: 'waist', torso: 'torso', collar: 'collar', neck: 'head',
  l_shoulder: 'l_upper_arm', l_elbow: 'l_lower_arm', l_hand: 'l_hand',
  r_shoulder: 'r_upper_arm', r_elbow: 'r_lower_arm', r_hand: 'r_hand',
  l_hip: 'l_upper_leg', l_knee: 'l_lower_leg', l_foot: 'l_foot', l_toe: 'l_toe',
  r_hip: 'r_upper_leg', r_knee: 'r_lower_leg', r_foot: 'r_foot', r_toe: 'r_toe',
};

const JOINT_CHILD_MAP: Partial<Record<keyof WalkingEnginePivotOffsets, keyof WalkingEnginePivotOffsets>> = {
    waist: 'torso',
    torso: 'collar',
    collar: 'neck',
    l_shoulder: 'l_elbow',
    l_elbow: 'l_hand',
    r_shoulder: 'r_elbow',
    r_elbow: 'r_hand',
    l_hip: 'l_knee',
    l_knee: 'l_foot',
    l_foot: 'l_toe',
    r_hip: 'r_knee',
    r_knee: 'r_foot',
    r_foot: 'r_toe',
};
const JOINT_PARENT_MAP: Partial<Record<keyof WalkingEnginePivotOffsets, keyof WalkingEnginePivotOffsets>> = Object.fromEntries(
  Object.entries(JOINT_CHILD_MAP).map(([parent, child]) => [child, parent as keyof WalkingEnginePivotOffsets])
);

const easeOutExpo = (t: number): number => {
  return t === 1 ? 1 : 1 - Math.pow(2, -10 * t);
};
const easeOutCubic = (t: number): number => 1 - Math.pow(1 - t, 3);

interface HistoryState {
  pivotOffsets: WalkingEnginePivotOffsets;
  props: WalkingEngineProportions;
  timestamp: number;
  label?: string;
}

type SelectionScope = 'part' | 'hierarchy' | 'full';
const SELECTION_SCOPES: SelectionScope[] = ['part', 'hierarchy', 'full'];
type MotionStyle = 'standard' | 'clockwork' | 'lotte';
const FLOOR_Y = 600; // Global floor plane

export const App: React.FC = () => {
  const [showLabels, setShowLabels] = useState(false);
  const [baseH] = useState(150);
  const [isConsoleVisible, setIsConsoleVisible] = useState(false);
  const [isKeymapVisible, setIsKeymapVisible] = useState(false);
  const [activeControlTab, setActiveControlTab] = useState<'fk' | 'perf' | 'props' | 'animation'>('fk');
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [isCalibrated, setIsCalibrated] = useState(false);
  const [physicsState, setPhysicsState] = useState<PhysicsState>({ position: { x: 0, y: 0 }, velocity: { x: 0, y: 0 }, angularVelocity: 0, worldGravity: { x: 0, y: 9.8 } });
  const [bodyRotation, setBodyRotation] = useState(0);
  
  const [activePins, setActivePins] = useState<(keyof WalkingEnginePivotOffsets)[]>([]);
  const [pinTargetPositions, setPinTargetPositions] = useState<Record<string, Vector2D>>({});
  const [limbTensions, setLimbTensions] = useState<Record<string, number>>({});
  const [hardStopEnabled, setHardStopEnabled] = useState(true);
  const [pinsAtLimit, setPinsAtLimit] = useState<Set<keyof WalkingEnginePivotOffsets>>(new Set());
  
  const [allJointPositions, setAllJointPositions] = useState<GlobalPositions>({});
  const [onionSkinData, setOnionSkinData] = useState<HistoryState | null>(null);
  const [selectedBoneKey, setSelectedBoneKey] = useState<keyof WalkingEnginePivotOffsets | null>(null);
  const [selectionScope, setSelectionScope] = useState<SelectionScope>('part');
  const fkControlsRef = useRef<HTMLDivElement>(null);
  const [maskImage, setMaskImage] = useState<string | null>(null);
  const [maskTransform, setMaskTransform] = useState<MaskTransform>({ x: 0, y: 0, rotation: 0, scale: 1 });
  const [backgroundImage, setBackgroundImage] = useState<string | null>(null);
  const [backgroundTransform, setBackgroundTransform] = useState<MaskTransform>({ x: 0, y: 0, rotation: 0, scale: 1 });
  const [blendMode, setBlendMode] = useState('normal');
  const [pivotOffsets, setPivotOffsets] = useState<WalkingEnginePivotOffsets>(INITIAL_CHALLENGE_POSE);
  const [props, setProps] = useState<WalkingEngineProportions>(ATOMIC_PROPS);
  const [jointChainBehaviors, setJointChainBehaviors] = useState<JointChainBehaviors>({});
  const [previewPivotOffsets, setPreviewPivotOffsets] = useState<WalkingEnginePivotOffsets | null>(null);
  const [staticGhostPose, setStaticGhostPose] = useState<WalkingEnginePivotOffsets | null>(null);
  const [displayedPivotOffsets, setDisplayedPivotOffsets] = useState<WalkingEnginePivotOffsets>(INITIAL_CHALLENGE_POSE);
  const [predictiveGhostingEnabled, setPredictiveGhostingEnabled] = useState(true);
  const [showIntentPath, setShowIntentPath] = useState(true);
  const [jointFriction, setJointFriction] = useState(50);
  const [motionStyle, setMotionStyle] = useState<MotionStyle>('standard');
  const [targetFps, setTargetFps] = useState<number | null>(null);
  const [isTransitioning, setIsTransitioning] = useState(false);
  const transitionAnimationRef = useRef<number | null>(null);
  const transitionStartPoseRef = useRef<WalkingEnginePivotOffsets | null>(null);
  const transitionStartTimeRef = useRef<number | null>(null);
  const [keyframes, setKeyframes] = useState<Keyframe[]>([]);
  const [selectedKeyframeIndex, setSelectedKeyframeIndex] = useState<number | null>(null);
  const [draggingBoneKey, setDraggingBoneKey] = useState<keyof WalkingEnginePivotOffsets | null>(null);
  const [history, setHistory] = useState<HistoryState[]>([]);
  const [redoStack, setRedoStack] = useState<HistoryState[]>([]);
  const [recordingHistory, setRecordingHistory] = useState<HistoryState[]>([]);
  const [selectedLogIndex, setSelectedLogIndex] = useState<number | null>(null);
  const draggingBoneKeyRef = useRef<keyof WalkingEnginePivotOffsets | null>(null);
  const isSliderDraggingRef = useRef(false);
  const lastClientXRef = useRef(0);
  const isInteractingRef = useRef(false);
  const latestPivotOffsetsRef = useRef(pivotOffsets);
  const jointVelocitiesRef = useRef<WalkingEnginePivotOffsets>({...T_POSE});
  const prevPivotOffsetsForVelRef = useRef<WalkingEnginePivotOffsets>(pivotOffsets);

  const [showOnionSkins, setShowOnionSkins] = useState(true);
  const [onionSkinFrames, setOnionSkinFrames] = useState({ before: 1, after: 1 });
  const [isCapsLockOn, setIsCapsLockOn] = useState(false);
  const [exportTransparentBg, setExportTransparentBg] = useState(false);
  const [exportShowAnchors, setExportShowAnchors] = useState(true);
  const svgRef = useRef<SVGSVGElement>(null);
  const [isExportingGif, setIsExportingGif] = useState(false);
  const [gifExportProgress, setGifExportProgress] = useState(0);
  const [gifRenderPose, setGifRenderPose] = useState<WalkingEnginePivotOffsets | null>(null);
  
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => { if (e.key === 'CapsLock') setIsCapsLockOn(e.getModifierState('CapsLock')); };
    const handleKeyUp = (e: KeyboardEvent) => { if (e.key === 'CapsLock') setIsCapsLockOn(e.getModifierState('CapsLock')); };
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, []);

  const {
    isAnimating,
    animationTime,
    totalDuration,
    play: handlePlay,
    pause: handlePause,
    reset: resetAnimationEngine,
    setAnimationTime,
  } = useAnimationEngine(keyframes, setPivotOffsets);

  const addLog = (message: string) => { setRecordingHistory(prev => [...prev.slice(-99), { timestamp: Date.now(), label: message } as HistoryState]); };

  const handleExportPNG = useCallback(async () => {
    addLog("EXPORT: Generating PNG...");
    const svgElement = svgRef.current;
    if (!svgElement) {
        addLog("ERR: SVG element not found for export.");
        return;
    }

    const svgClone = svgElement.cloneNode(true) as SVGSVGElement;

    if (!exportShowAnchors) {
        svgClone.querySelectorAll('[data-no-export="true"]').forEach(el => el.remove());
    }

    if (!exportTransparentBg) {
        const bgRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        bgRect.setAttribute('width', '100%');
        bgRect.setAttribute('height', '100%');
        bgRect.setAttribute('fill', '#FFFFFF'); // 'paper' color
        bgRect.setAttribute('x', svgClone.viewBox.baseVal.x.toString());
        bgRect.setAttribute('y', svgClone.viewBox.baseVal.y.toString());
        svgClone.insertBefore(bgRect, svgClone.firstChild);
    }
    
    const rect = svgElement.getBoundingClientRect();
    const scale = 2;
    svgClone.setAttribute('width', `${rect.width * scale}`);
    svgClone.setAttribute('height', `${rect.height * scale}`);

    const svgXml = new XMLSerializer().serializeToString(svgClone);
    const svgDataUrl = `data:image/svg+xml;base64,${btoa(unescape(encodeURIComponent(svgXml)))}`;

    const img = new Image();
    
    img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = rect.width * scale;
        canvas.height = rect.height * scale;
        const ctx = canvas.getContext('2d');
        if (!ctx) {
            addLog("ERR: Could not get canvas context for export.");
            return;
        }
        
        ctx.drawImage(img, 0, 0);

        const pngUrl = canvas.toDataURL('image/png');
        const link = document.createElement('a');
        link.href = pngUrl;
        link.download = `bitruvian_export_${Date.now()}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        addLog("EXPORT: PNG download initiated.");
    };

    img.onerror = () => {
        addLog("ERR: Failed to load SVG image for export.");
    };

    img.src = svgDataUrl;
  }, [exportShowAnchors, exportTransparentBg, addLog]);

  const handleExportGIF = useCallback(async () => {
    if (keyframes.length < 2 || totalDuration <= 0) {
      addLog("ERR: Not enough keyframes to create a GIF.");
      return;
    }
    addLog("GIF EXPORT: Starting render...");
    setIsExportingGif(true);
    setGifExportProgress(0);
  
    const svgElement = svgRef.current;
    if (!svgElement) {
      addLog("ERR: SVG element not found.");
      setIsExportingGif(false);
      return;
    }
  
    const rect = svgElement.getBoundingClientRect();
    const scale = 1; // Use 1x for performance, can be increased for quality
    const width = rect.width * scale;
    const height = rect.height * scale;
  
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) {
      addLog("ERR: Canvas context failed.");
      setIsExportingGif(false);
      return;
    }
  
    const gif = gifenc.GIFEncoder();
    const FPS = 24;
    const delay = 1000 / FPS;
    let firstFrame = true;
  
    const captureFrame = async (time: number) => {
      if (time > totalDuration) {
        const buffer = await gif.finish();
        const blob = new Blob([buffer], { type: 'image/gif' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `bitruvian_anim_${Date.now()}.gif`;
        link.click();
        URL.revokeObjectURL(url);
        addLog("GIF EXPORT: Finished and downloaded.");
        setGifRenderPose(null);
        setIsExportingGif(false);
        return;
      }
  
      // Calculate pose at current time
      let startPose: WalkingEnginePivotOffsets, endPose: WalkingEnginePivotOffsets;
      let segmentStartTime: number, segmentDuration: number;
      
      let segmentFound = false;
      for (let i = 0; i < keyframes.length - 1; i++) {
          if (time >= keyframes[i].time && time < keyframes[i+1].time) {
              startPose = keyframes[i].pose; endPose = keyframes[i+1].pose;
              segmentStartTime = keyframes[i].time; segmentDuration = keyframes[i+1].time - keyframes[i].time;
              segmentFound = true; break;
          }
      }
      if (!segmentFound) {
          const lastKeyframe = keyframes[keyframes.length - 1];
          startPose = lastKeyframe.pose; endPose = keyframes[0].pose;
          segmentStartTime = lastKeyframe.time; segmentDuration = 1000;
      }
      
      const timeIntoSegment = time - segmentStartTime;
      const progress = segmentDuration > 0 ? timeIntoSegment / segmentDuration : 1;
      const currentPose = { ...startPose } as WalkingEnginePivotOffsets;
      JOINT_KEYS.forEach(k => {
          currentPose[k] = lerpAngleShortestPath(startPose[k], endPose[k], progress);
      });
  
      setGifRenderPose(currentPose);
      setGifExportProgress((time / totalDuration) * 100);
  
      await new Promise(resolve => requestAnimationFrame(resolve));
  
      const svgClone = svgElement.cloneNode(true) as SVGSVGElement;
      if (!exportShowAnchors) svgClone.querySelectorAll('[data-no-export="true"]').forEach(el => el.remove());
      if (!exportTransparentBg) {
        const bgRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        bgRect.setAttribute('width', '100%'); bgRect.setAttribute('height', '100%'); bgRect.setAttribute('fill', '#FFFFFF');
        bgRect.setAttribute('x', svgClone.viewBox.baseVal.x.toString()); bgRect.setAttribute('y', svgClone.viewBox.baseVal.y.toString());
        svgClone.insertBefore(bgRect, svgClone.firstChild);
      }
      svgClone.setAttribute('width', `${width}`);
      svgClone.setAttribute('height', `${height}`);
      
      const svgXml = new XMLSerializer().serializeToString(svgClone);
      const svgDataUrl = `data:image/svg+xml;base64,${btoa(unescape(encodeURIComponent(svgXml)))}`;
      
      const img = new Image();
      await new Promise<void>((resolve, reject) => {
        img.onload = () => {
            ctx.clearRect(0, 0, width, height);
            ctx.drawImage(img, 0, 0, width, height);
            const data = ctx.getImageData(0, 0, width, height).data;
            let palette, indexed;
            if (firstFrame) {
                palette = gifenc.quantize(data, 256, { format: 'rgba4444' });
                indexed = gifenc.applyPalette(data, palette, 'rgba4444');
                firstFrame = false;
            } else {
                palette = gifenc.quantize(data, 256, { format: 'rgba4444' });
                indexed = gifenc.applyPalette(data, palette, 'rgba4444');
            }
            gif.writeFrame(indexed, width, height, { palette, delay });
            resolve();
        };
        img.onerror = reject;
        img.src = svgDataUrl;
      });
  
      captureFrame(time + delay);
    };
  
    captureFrame(0);
  
  }, [keyframes, totalDuration, exportShowAnchors, exportTransparentBg, addLog]);

  const handleAddKeyframe = useCallback(() => {
    if (isAnimating || isTransitioning) return;
    const currentPose = { ...pivotOffsets };
    const newTime = keyframes.length > 0 ? keyframes[keyframes.length - 1].time + 1000 : 0;
    const newKeyframe: Keyframe = {
      id: `kf-${Date.now()}`,
      pose: currentPose,
      time: newTime,
    };
    setKeyframes(prev => [...prev, newKeyframe]);
    addLog(`ANIM: Keyframe ${keyframes.length + 1} added.`);
    recordSnapshot(`ADD_KEYFRAME_${keyframes.length + 1}`);
    setSelectedKeyframeIndex(keyframes.length);
  }, [pivotOffsets, keyframes, isAnimating, isTransitioning, addLog]);

  const handleFullAnimationReset = useCallback(() => {
      resetAnimationEngine();
      setKeyframes([]);
      setSelectedKeyframeIndex(null);
      addLog("ANIM: Timeline cleared.");
  }, [resetAnimationEngine]);
  
  const handleSelectKeyframe = useCallback((index: number) => {
      if (isAnimating) handlePause();
      setSelectedKeyframeIndex(index);
      setPivotOffsets(keyframes[index].pose);
      setAnimationTime(keyframes[index].time);
      addLog(`ANIM: Selected keyframe ${index + 1} for editing.`);
  }, [keyframes, isAnimating, handlePause, setAnimationTime]);

  const handleScrub = useCallback((time: number) => {
      if (isAnimating) handlePause();
      setSelectedKeyframeIndex(null);
      setAnimationTime(time);
      
      if (keyframes.length < 2) return;
      
      const totalDur = keyframes[keyframes.length - 1].time + 1000;
      if (totalDur <= 0) return;
      
      const effectiveTime = time % totalDur;
      let startPose: WalkingEnginePivotOffsets, endPose: WalkingEnginePivotOffsets;
      let segmentStartTime: number, segmentDuration: number;
      
      let segmentFound = false;
      for (let i = 0; i < keyframes.length - 1; i++) {
          if (effectiveTime >= keyframes[i].time && effectiveTime < keyframes[i+1].time) {
              startPose = keyframes[i].pose;
              endPose = keyframes[i+1].pose;
              segmentStartTime = keyframes[i].time;
              segmentDuration = keyframes[i+1].time - keyframes[i].time;
              segmentFound = true;
              break;
          }
      }

      if (!segmentFound) {
          const lastKeyframe = keyframes[keyframes.length - 1];
          startPose = lastKeyframe.pose;
          endPose = keyframes[0].pose;
          segmentStartTime = lastKeyframe.time;
          segmentDuration = 1000;
      }
      
      const timeIntoSegment = effectiveTime - segmentStartTime;
      const progress = segmentDuration > 0 ? timeIntoSegment / segmentDuration : 1;
      
      const nextPose = { ...startPose } as WalkingEnginePivotOffsets;
      JOINT_KEYS.forEach(k => {
          const start = startPose[k];
          const end = endPose[k];
          nextPose[k] = lerpAngleShortestPath(start, end, progress);
      });
      setPivotOffsets(nextPose);
  }, [isAnimating, handlePause, keyframes, setAnimationTime]);

  const handleAddTweenFrame = useCallback(() => {
    if (isAnimating || keyframes.length < 2) return;
    
    const newKeyframe: Keyframe = {
        id: `kf-tween-${Date.now()}`,
        pose: { ...pivotOffsets },
        time: animationTime
    };
    
    const newKeyframes = [...keyframes, newKeyframe].sort((a, b) => a.time - b.time);
    const newIndex = newKeyframes.findIndex(k => k.id === newKeyframe.id);
    
    setKeyframes(newKeyframes);
    setSelectedKeyframeIndex(newIndex);
    addLog(`ANIM: Added tween as new keyframe.`);
  }, [isAnimating, keyframes, animationTime, pivotOffsets]);
  
  const handleUpdateKeyframeTime = useCallback((index: number, newTime: number) => {
    setKeyframes(currentKeyframes => {
      const newKeyframes = [...currentKeyframes];
      if (newKeyframes[index]) {
        newKeyframes[index].time = newTime;
      }
      return newKeyframes;
    });
  }, []);
  
  const onionSkinPoses = useMemo(() => {
    if (!showOnionSkins || keyframes.length < 2) return [];

    const skins: { pose: WalkingEnginePivotOffsets, opacity: number, index: number }[] = [];
    const currentIdx = selectedKeyframeIndex;
    
    if (currentIdx !== null) {
        // Show before/after frames relative to the selected keyframe
        for (let i = 1; i <= onionSkinFrames.before; i++) {
            const idx = currentIdx - i;
            if (idx >= 0) {
                skins.push({ pose: keyframes[idx].pose, opacity: 0.4 - i * 0.1, index: idx });
            }
        }
        for (let i = 1; i <= onionSkinFrames.after; i++) {
            const idx = currentIdx + i;
            if (idx < keyframes.length) {
                skins.push({ pose: keyframes[idx].pose, opacity: 0.4 - i * 0.1, index: idx });
            }
        }
    }
    return skins;
  }, [showOnionSkins, keyframes, selectedKeyframeIndex, onionSkinFrames]);


  // Auto-save edits back to the selected keyframe
  useEffect(() => {
      if (selectedKeyframeIndex !== null && !isAnimating && !isTransitioning && !isInteractingRef.current) {
          setKeyframes(currentKeyframes => {
              const newKeyframes = [...currentKeyframes];
              if (newKeyframes[selectedKeyframeIndex]) {
                  newKeyframes[selectedKeyframeIndex] = {
                      ...newKeyframes[selectedKeyframeIndex],
                      pose: pivotOffsets
                  };
              }
              return newKeyframes;
          });
      }
  }, [pivotOffsets, selectedKeyframeIndex, isAnimating, isTransitioning]);

  const applyChainReaction = useCallback((startingKey: keyof WalkingEnginePivotOffsets, delta: number, initialOffsets: WalkingEnginePivotOffsets): WalkingEnginePivotOffsets => {
      const newOffsets = { ...initialOffsets };
      const queue: [keyof WalkingEnginePivotOffsets, number][] = [[startingKey, delta]];
      const visited = new Set<keyof WalkingEnginePivotOffsets>();
      visited.add(startingKey);
      while (queue.length > 0) {
          const [currentKey, currentDelta] = queue.shift()!;
          let children: (keyof WalkingEnginePivotOffsets)[] = [];
          if (currentKey === 'waist') children = ['torso', 'l_hip', 'r_hip'];
          else if (currentKey === 'torso') children = ['collar'];
          else if (currentKey === 'collar') children = ['neck', 'l_shoulder', 'r_shoulder'];
          else if (JOINT_CHILD_MAP[currentKey]) children = [JOINT_CHILD_MAP[currentKey]!];
          for (const childKey of children) {
              if (visited.has(childKey)) continue;
              const behavior = jointChainBehaviors[childKey] || {};
              const bendFactor = behavior.b ?? 0;
              const stretchFactor = behavior.s ?? 0;
              const totalFactor = bendFactor + stretchFactor;
              if (totalFactor !== 0) {
                  const childDelta = currentDelta * totalFactor;
                  newOffsets[childKey] = (newOffsets[childKey] || 0) + childDelta;
                  queue.push([childKey, childDelta]);
                  visited.add(childKey);
              }
          }
      }
      return newOffsets;
  }, [jointChainBehaviors]);
  
  const handleMaskUpload = (e: React.ChangeEvent<HTMLInputElement>) => { const file = e.target.files?.[0]; if (file) { const reader = new FileReader(); reader.onerror = () => addLog("ERR: Mask upload failed."); reader.onload = (readerEvent) => { const result = readerEvent.target?.result as string; if (result) { setMaskImage(result); addLog("IO: Mask image uploaded."); } }; reader.readAsDataURL(file); } };
  const handleBackgroundUpload = (e: React.ChangeEvent<HTMLInputElement>) => { const file = e.target.files?.[0]; if (file) { const reader = new FileReader(); reader.onerror = () => addLog("ERR: Background upload failed."); reader.onload = (readerEvent) => { const result = readerEvent.target?.result as string; if (result) { setBackgroundImage(result); addLog("IO: Background image uploaded."); } }; reader.readAsDataURL(file); } };
  const recordSnapshot = useCallback((label?: string) => { setRecordingHistory(prev => [...prev, { pivotOffsets: { ...pivotOffsets }, props: JSON.parse(JSON.stringify(props)), timestamp: Date.now(), label }]); }, [pivotOffsets, props]);
  const saveToHistory = useCallback(() => { setHistory(prev => [...prev.slice(-49), { pivotOffsets: { ...pivotOffsets }, props: JSON.parse(JSON.stringify(props)), timestamp: Date.now() }]); setRedoStack([]); }, [pivotOffsets, props]);
  const undo = useCallback(() => { if (history.length === 0 || isAnimating || isTransitioning) return; const previous = history[history.length - 1]; const current: HistoryState = { pivotOffsets: { ...pivotOffsets }, props: JSON.parse(JSON.stringify(props)), timestamp: Date.now() }; setRedoStack(prev => [current, ...prev]); setHistory(prev => prev.slice(0, -1)); setPivotOffsets(previous.pivotOffsets); setProps(previous.props); addLog("UNDO: System state reverted."); }, [history, pivotOffsets, props, isAnimating, isTransitioning]);
  const redo = useCallback(() => { if (redoStack.length === 0 || isAnimating || isTransitioning) return; const next = redoStack[0]; const current: HistoryState = { pivotOffsets: { ...pivotOffsets }, props: JSON.parse(JSON.stringify(props)), timestamp: Date.now() }; setHistory(prev => [...prev, current]); setRedoStack(prev => prev.slice(1)); setPivotOffsets(next.pivotOffsets); setProps(next.props); addLog("REDO: System state reapplied."); }, [redoStack, pivotOffsets, props, isAnimating, isTransitioning]);
  const handleLogClick = useCallback((log: HistoryState, index: number) => { setSelectedLogIndex(index); if (log.pivotOffsets && !isAnimating && !isTransitioning) { setPivotOffsets(log.pivotOffsets); if (log.props) setProps(log.props); } }, [isAnimating, isTransitioning]);
  useEffect(() => { const handleKeyDown = (e: KeyboardEvent) => { if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) { return; } if ((e.key === 'Delete' || e.key === 'Backspace') && selectedLogIndex !== null) { e.preventDefault(); const deletedLog = recordingHistory[selectedLogIndex]; setRecordingHistory(prev => prev.filter((_, i) => i !== selectedLogIndex)); setSelectedLogIndex(null); addLog(`LOG DELETED: "${deletedLog.label || `Pose @ ${new Date(deletedLog.timestamp).toLocaleTimeString()}`}" removed.`); return; } if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); if (e.shiftKey) redo(); else undo(); } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') { e.preventDefault(); redo(); } else if (e.key === 'Tab') { if (selectedBoneKey) { e.preventDefault(); const currentIndex = SELECTION_SCOPES.indexOf(selectionScope); const nextIndex = e.shiftKey ? (currentIndex - 1 + SELECTION_SCOPES.length) % SELECTION_SCOPES.length : (currentIndex + 1) % SELECTION_SCOPES.length; setSelectionScope(SELECTION_SCOPES[nextIndex]); } } }; window.addEventListener('keydown', handleKeyDown); return () => window.removeEventListener('keydown', handleKeyDown); }, [selectedLogIndex, recordingHistory, redo, undo, selectedBoneKey, selectionScope]);
  useEffect(() => { if (selectedBoneKey && fkControlsRef.current) { const element = fkControlsRef.current.querySelector(`[data-joint-key="${selectedBoneKey}"]`) as HTMLDivElement; if (element) { element.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); } } }, [selectedBoneKey]);
  const animatePoseTransition = useCallback((targetPose: Partial<WalkingEnginePivotOffsets>, duration: number = 700, onComplete?: () => void) => { if (transitionAnimationRef.current) { cancelAnimationFrame(transitionAnimationRef.current); } const startPose = { ...pivotOffsets }; transitionStartPoseRef.current = startPose; transitionStartTimeRef.current = performance.now(); setIsTransitioning(true); setStaticGhostPose(startPose); const localMotionStyle = motionStyle; const runClockworkJitter = (finalPose: WalkingEnginePivotOffsets) => { let frame = 0; const jitterFrames = 2; const jitterAmount = 1.5 * (1 - (jointFriction / 100)); const jitterLoop = () => { if (frame >= jitterFrames) { setPivotOffsets(finalPose); if (onComplete) onComplete(); return; } const jitteredPose = { ...finalPose }; JOINT_KEYS.forEach(key => { const rand = (Math.random() - 0.5) * 2; jitteredPose[key] += rand * jitterAmount; }); setPivotOffsets(jitteredPose); frame++; requestAnimationFrame(jitterLoop); }; requestAnimationFrame(jitterLoop); }; const animate = (now: number) => { const elapsed = now - transitionStartTimeRef.current!; const progress = Math.min(elapsed / duration, 1); let easedProgress; switch (localMotionStyle) { case 'lotte': easedProgress = easeOutCubic(progress); break; default: easedProgress = easeOutExpo(progress); } const newOffsets: WalkingEnginePivotOffsets = { ...startPose }; JOINT_KEYS.forEach(key => { const start = startPose[key] || 0; const end = targetPose[key] ?? start; let finalValue = lerpAngleShortestPath(start, end, easedProgress); if (localMotionStyle === 'clockwork') { finalValue = Math.round(finalValue / 5) * 5; } newOffsets[key] = finalValue; }); setPivotOffsets(newOffsets); if (progress < 1) { transitionAnimationRef.current = requestAnimationFrame(animate); } else { setIsTransitioning(false); transitionAnimationRef.current = null; setStaticGhostPose(null); const finalPose = { ...startPose, ...targetPose } as WalkingEnginePivotOffsets; if (localMotionStyle === 'clockwork') { runClockworkJitter(finalPose); } else { setPivotOffsets(finalPose); if (onComplete) onComplete(); } } }; transitionAnimationRef.current = requestAnimationFrame(animate); }, [pivotOffsets, motionStyle, jointFriction]);
  const startCalibration = useCallback(() => { if (isCalibrated || isCalibrating || isAnimating || isTransitioning) return; saveToHistory(); recordSnapshot("CALIBRATION_START"); setIsCalibrating(true); addLog("SEQUENCE: CALIBRATION START..."); animatePoseTransition(T_POSE, 500, () => { setDisplayedPivotOffsets(T_POSE); setIsCalibrating(false); setIsCalibrated(true); setIsConsoleVisible(true); recordSnapshot("CALIBRATION_END"); addLog("SEQUENCE: SYSTEM ALIGNED."); }); }, [isCalibrated, isCalibrating, isAnimating, isTransitioning, saveToHistory, recordSnapshot, animatePoseTransition]);
  const poseString = useMemo(() => { const poseData = JOINT_KEYS.map(k => `${k}:${Math.round(pivotOffsets[k])}`).join(';'); const propData = PROP_KEYS.map(k => `${k}:h${props[k].h.toFixed(2)},w${props[k].w.toFixed(2)}`).join(';'); return `POSE[${poseData}]|PROPS[${propData}]`; }, [pivotOffsets, props]);
  const handlePivotChange = useCallback((key: keyof WalkingEnginePivotOffsets, newValue: number) => { const updateFunc = predictiveGhostingEnabled ? setPreviewPivotOffsets : setPivotOffsets; updateFunc(currentOffsets => { const baseOffsets = currentOffsets || pivotOffsets; const delta = newValue - baseOffsets[key]; let newOffsets = { ...baseOffsets, [key]: newValue }; newOffsets = applyChainReaction(key, delta, newOffsets); return newOffsets; }); }, [pivotOffsets, applyChainReaction, predictiveGhostingEnabled]);

  const primaryPin = activePins.length > 0 ? activePins[0] : 'waist';

  const pinnedJointPosition = useMemo((): Vector2D => { const partKey = PIVOT_TO_PART_MAP[primaryPin]; if (!partKey || !allJointPositions[partKey]) { return { x: 0, y: 0 }; } return allJointPositions[partKey]!.position; }, [primaryPin, allJointPositions]);
  
  const handleInteractionMove = useCallback((clientX: number, clientY: number) => { if (isSliderDraggingRef.current || !isCalibrated || isAnimating || isTransitioning || !draggingBoneKeyRef.current) return; if (!isInteractingRef.current) { isInteractingRef.current = true; } setSelectedKeyframeIndex(null); const boneKey = draggingBoneKeyRef.current; const behaviors = jointChainBehaviors[boneKey] || {}; const parentKey = JOINT_PARENT_MAP[boneKey]; const svg = svgRef.current; if (!svg) return; const pt = svg.createSVGPoint(); pt.x = clientX; pt.y = clientY; const cursorSvgPos = pt.matrixTransform(svg.getScreenCTM()?.inverse()); const updatePoseState = (newState: WalkingEnginePivotOffsets) => { if (predictiveGhostingEnabled) { setPreviewPivotOffsets(newState); } else { setPivotOffsets(newState); } }; let baseOffsets = predictiveGhostingEnabled ? (previewPivotOffsets || pivotOffsets) : pivotOffsets; if (behaviors.l && parentKey) { const parentPartKey = PIVOT_TO_PART_MAP[parentKey]; const parentPos = allJointPositions[parentPartKey]?.position; if (parentPos) { const angle = Math.atan2(cursorSvgPos.y - parentPos.y, cursorSvgPos.x - parentPos.x) * 180 / Math.PI; const grandParentKey = JOINT_PARENT_MAP[parentKey]; const grandParentPartKey = grandParentKey ? PIVOT_TO_PART_MAP[grandParentKey] : undefined; const grandParentRot = grandParentPartKey ? (allJointPositions[grandParentPartKey]?.rotation || 0) : bodyRotation; const currentParentRot = baseOffsets[parentKey]; const newParentRot = angle - 90 - grandParentRot; const delta = newParentRot - currentParentRot; baseOffsets = { ...baseOffsets, [parentKey]: newParentRot }; baseOffsets = applyChainReaction(parentKey, delta, baseOffsets); } } else { const frictionFactor = 1 - (jointFriction / 125); const dragDelta = (clientX - lastClientXRef.current) * frictionFactor; const originalValue = baseOffsets[boneKey!]; const newValue = originalValue + dragDelta; baseOffsets = { ...baseOffsets, [boneKey!]: newValue }; baseOffsets = applyChainReaction(boneKey!, dragDelta, baseOffsets); } lastClientXRef.current = clientX; updatePoseState(baseOffsets); }, [isCalibrated, isAnimating, isTransitioning, jointFriction, applyChainReaction, jointChainBehaviors, allJointPositions, bodyRotation, predictiveGhostingEnabled, pivotOffsets, previewPivotOffsets]);
  
  useEffect(() => {
    if (!previewPivotOffsets || activePins.length === 0) {
        return;
    }
    const ikTargets = activePins.map(pin => ({ key: pin, target: pinTargetPositions[pin] }));
    const tempOffsets = { ...previewPivotOffsets };
    let ikApplied = false;

    ikTargets.forEach(({ key, target }) => {
        if (!target) return;
        const isLeft = key.startsWith('l_');
        let chain: (keyof WalkingEnginePivotOffsets)[] | null = null;
        if (key === 'l_foot' || key === 'r_foot') {
            chain = isLeft ? ['l_hip', 'l_knee'] : ['r_hip', 'r_knee'];
        }
        if (!chain) return;

        const [hipKey, kneeKey] = chain;
        const thighPart: keyof WalkingEngineProportions = isLeft ? 'l_upper_leg' : 'r_upper_leg';
        const calfPart: keyof WalkingEngineProportions = isLeft ? 'l_lower_leg' : 'r_lower_leg';
        const hipPos = allJointPositions[PIVOT_TO_PART_MAP[hipKey]]?.position;
        if (!hipPos) return;

        const thighLen = getKinematicDimension(partDefinitions[thighPart].rawH, baseH, props, thighPart, 'h');
        const calfLen = getKinematicDimension(partDefinitions[calfPart].rawH, baseH, props, calfPart, 'h');
        
        const parentOfHipKey = JOINT_PARENT_MAP[hipKey]!;
        const parentAngle = allJointPositions[PIVOT_TO_PART_MAP[parentOfHipKey]]?.rotation || 0;

        const ikResult = solveTwoBoneIK(target, hipPos, thighLen, calfLen, parentAngle, isLeft);

        if (ikResult) {
            ikApplied = true;
            tempOffsets[hipKey] = ikResult.angle1;
            tempOffsets[kneeKey] = ikResult.angle2;
        }
    });

    if (ikApplied) {
        setPreviewPivotOffsets(tempOffsets);
    }
}, [previewPivotOffsets, activePins, pinTargetPositions, allJointPositions, baseH, props, bodyRotation]);

useEffect(() => {
  let animationFrameId: number;
  const loop = () => {
    if (activePins.length > 0) {
      const newTensions: Record<string, number> = {};
      const newPinsAtLimit = new Set<keyof WalkingEnginePivotOffsets>();
      let totalPull = { x: 0, y: 0 };

      activePins.forEach(pinKey => {
        const partKey = PIVOT_TO_PART_MAP[pinKey];
        const currentPos = allJointPositions[partKey]?.position;
        const targetPos = pinTargetPositions[pinKey];

        if (currentPos && targetPos) {
          const dist = distance(currentPos, targetPos);
          if (dist > 0.1) {
            if (pinKey.includes('leg') || pinKey.includes('foot')) {
                const tensionKey = pinKey.startsWith('l_') ? 'l_leg' : 'r_leg';
                newTensions[tensionKey] = 1 + dist / 100;
            }
            const pullForce = Math.min(Math.pow(dist, 1.5) * 0.005, 5);
            const pullVector = {
                x: (targetPos.x - currentPos.x) / dist,
                y: (targetPos.y - currentPos.y) / dist
            };
            totalPull.x += pullVector.x * pullForce;
            totalPull.y += pullVector.y * pullForce;
          }
           if (hardStopEnabled && dist > 20) {
             newPinsAtLimit.add(pinKey);
             const correction = dist - 20;
             totalPull.x -= (targetPos.x - currentPos.x) / dist * correction * 0.5;
             totalPull.y -= (targetPos.y - currentPos.y) / dist * correction * 0.5;
           }
        }
      });

      setLimbTensions(newTensions);
      setPinsAtLimit(newPinsAtLimit);

      if (Math.abs(totalPull.x) > 0.01 || Math.abs(totalPull.y) > 0.01) {
        setPhysicsState(p => ({
            ...p,
            position: { x: p.position.x + totalPull.x, y: p.position.y + totalPull.y }
        }));
      }
    }
    animationFrameId = requestAnimationFrame(loop);
  };
  animationFrameId = requestAnimationFrame(loop);
  return () => cancelAnimationFrame(animationFrameId);
}, [activePins, pinTargetPositions, allJointPositions, hardStopEnabled]);

  const handleTogglePin = useCallback((boneKey: keyof WalkingEnginePivotOffsets, lockY?: number) => {
    setActivePins(prev => {
        const newPins = new Set(prev);
        if (newPins.has(boneKey)) {
            newPins.delete(boneKey);
            setPinTargetPositions(targets => {
                const newTargets = {...targets};
                delete newTargets[boneKey];
                return newTargets;
            });
            addLog(`PIN REMOVED: ${boneKey}`);
        } else {
            const partKey = PIVOT_TO_PART_MAP[boneKey];
            if (allJointPositions[partKey]) {
                newPins.add(boneKey);
                let targetPos = { ...allJointPositions[partKey]!.position };
                if (lockY !== undefined) {
                    targetPos.y = lockY;
                }
                setPinTargetPositions(targets => ({
                    ...targets,
                    [boneKey]: targetPos
                }));
                addLog(`PIN ADDED: ${boneKey}`);
            }
        }
        return Array.from(newPins);
    });
}, [allJointPositions]);

  const handleInteractionEnd = useCallback(() => { 
      isSliderDraggingRef.current = false; 

      if (draggingBoneKeyRef.current) {
        const boneKey = draggingBoneKeyRef.current;
        const partKey = PIVOT_TO_PART_MAP[boneKey];
        if (partKey && (partKey.includes('foot') || partKey.includes('toe'))) {
            const worldPos = allJointPositions[partKey];
            if (worldPos && Math.abs(worldPos.position.y - FLOOR_Y) < 10) {
                 handleTogglePin(boneKey, FLOOR_Y);
            }
        }
      }

      if (predictiveGhostingEnabled) { 
        if (draggingBoneKeyRef.current && previewPivotOffsets) { 
            recordSnapshot(`END_DRAG_${draggingBoneKeyRef.current}`); 
            const targetOffsets = { ...previewPivotOffsets }; 
            draggingBoneKeyRef.current = null; 
            isInteractingRef.current = false; 
            setPreviewPivotOffsets(null); 
            setStaticGhostPose(null); 
            const snapDuration = 50 + (jointFriction / 100) * 700; 
            animatePoseTransition(targetOffsets, snapDuration, () => { setPivotOffsets(targetOffsets); setDraggingBoneKey(null); }); 
        } else { 
            setPreviewPivotOffsets(null); 
            setDraggingBoneKey(null); 
            draggingBoneKeyRef.current = null; 
            isInteractingRef.current = false; 
            setStaticGhostPose(null); 
        } 
    } else { 
        if (draggingBoneKeyRef.current) { recordSnapshot(`END_DRAG_${draggingBoneKeyRef.current}`); } 
        draggingBoneKeyRef.current = null; 
        isInteractingRef.current = false; 
        setDraggingBoneKey(null); 
    } 
}, [recordSnapshot, previewPivotOffsets, jointFriction, animatePoseTransition, pivotOffsets, predictiveGhostingEnabled, allJointPositions, handleTogglePin]);

  const handleGlobalMouseMove = useCallback((e: MouseEvent) => { handleInteractionMove(e.clientX, e.clientY); }, [handleInteractionMove]);
  const handleGlobalTouchMove = useCallback((e: TouchEvent) => { if (e.touches[0]) { handleInteractionMove(e.touches[0].clientX, e.touches[0].clientY); } }, [handleInteractionMove]);
  const handleGlobalMouseUp = useCallback(() => { handleInteractionEnd(); }, [handleInteractionEnd]);
  const handleGlobalTouchEnd = useCallback(() => { handleInteractionEnd(); }, [handleInteractionEnd]);
  useEffect(() => { window.addEventListener('mousemove', handleGlobalMouseMove); window.addEventListener('mouseup', handleGlobalMouseUp); window.addEventListener('touchmove', handleGlobalTouchMove); window.addEventListener('touchend', handleGlobalTouchEnd); return () => { window.removeEventListener('mousemove', handleGlobalMouseMove); window.removeEventListener('mouseup', handleGlobalMouseUp); window.removeEventListener('touchmove', handleGlobalTouchMove); window.removeEventListener('touchend', handleGlobalTouchEnd); }; }, [handleGlobalMouseMove, handleGlobalMouseUp, handleGlobalTouchMove, handleGlobalTouchEnd]);
  const startDrag = useCallback((key: keyof WalkingEnginePivotOffsets, clientX: number) => { if (!isCalibrated || isAnimating || isTransitioning) return; saveToHistory(); recordSnapshot(`START_DRAG_${key}`); if (predictiveGhostingEnabled) { setPreviewPivotOffsets({ ...pivotOffsets }); setStaticGhostPose({ ...pivotOffsets }); } draggingBoneKeyRef.current = key; setDraggingBoneKey(key); lastClientXRef.current = clientX; }, [isCalibrated, isAnimating, isTransitioning, saveToHistory, recordSnapshot, pivotOffsets, predictiveGhostingEnabled]);
  
  const onAnchorMouseDown = useCallback((k: keyof WalkingEnginePivotOffsets, clientX: number, e: React.MouseEvent | React.TouchEvent) => { setSelectedKeyframeIndex(null); setSelectedBoneKey(k); setSelectionScope('part'); if (e.nativeEvent instanceof MouseEvent && e.shiftKey) { e.stopPropagation(); handleTogglePin(k); } else { startDrag(k, clientX); } }, [handleTogglePin, startDrag]);
  const handleBodyMouseDown = useCallback((k: keyof WalkingEnginePivotOffsets, clientX: number, e: React.MouseEvent | React.TouchEvent) => { setSelectedKeyframeIndex(null); setSelectedBoneKey(k); setSelectionScope('part'); const isShift = (e.nativeEvent instanceof MouseEvent && e.shiftKey) || (e.nativeEvent instanceof TouchEvent && e.shiftKey); if (isShift) { const childKey = JOINT_CHILD_MAP[k]; if (childKey) { startDrag(childKey, clientX); } else { startDrag(k, clientX); } } else { startDrag(k, clientX); } }, [startDrag]);
  const copyToClipboard = () => { navigator.clipboard.writeText(poseString); addLog("IO: State string copied to clipboard."); };
  const saveToFile = () => { const blob = new Blob([poseString], { type: 'text/plain' }); const url = URL.createObjectURL(blob); const link = document.createElement('a'); link.href = url; link.download = `bitruvian_pose_${Date.now()}.txt`; link.click(); addLog("IO: Pose exported to file."); };
  const updateProp = (key: keyof WalkingEngineProportions, axis: 'w' | 'h', val: number) => { if (isAnimating || isTransitioning) return; setProps(p => ({ ...p, [key]: { ...p[key], [axis]: val } })); };
  const resetProps = () => { if (isAnimating || isTransitioning) return; saveToHistory(); setProps(ATOMIC_PROPS); recordSnapshot("PROPS_RESET"); addLog("COMMAND: Anatomical proportions reset."); };
  const setFixedPose = (p: WalkingEnginePivotOffsets, name: string) => { if (isAnimating || isTransitioning) return; saveToHistory(); setPivotOffsets({ ...p }); recordSnapshot(`SET_POSE_${name.toUpperCase()}`); addLog(`COMMAND: Applied ${name} state.`); };
  const exportRecordingJSON = useCallback(() => { const dataStr = JSON.stringify(recordingHistory, null, 2); const blob = new Blob([dataStr], { type: 'application/json' }); const url = URL.createObjectURL(blob); const link = document.createElement('a'); link.href = url; link.download = `bitruvian_history_${Date.now()}.json`; link.click(); addLog("IO: Full rotation history exported as JSON."); }, [recordingHistory]);
  const clearHistory = () => { setRecordingHistory([]); addLog("COMMAND: Recording history cleared."); };
  const handleChainBehaviorToggle = (key: keyof WalkingEnginePivotOffsets, mode: 'b' | 's' | 'l') => { setJointChainBehaviors(prev => { const keyBehaviors = { ...(prev[key] || {}) }; let newModes: JointChainBehaviors[keyof WalkingEnginePivotOffsets]; if (mode === 'b') { const isCurrentlyActive = keyBehaviors.b != null && keyBehaviors.b !== 0; newModes = { ...keyBehaviors, b: isCurrentlyActive ? 0 : 1 }; } else if (mode === 's') { const isCurrentlyActive = keyBehaviors.s != null && keyBehaviors.s !== 0; newModes = { ...keyBehaviors, s: isCurrentlyActive ? 0 : -1 }; } else { newModes = { ...keyBehaviors, l: !keyBehaviors.l }; } if (newModes.b === 0) delete newModes.b; if (newModes.s === 0) delete newModes.s; return { ...prev, [key]: newModes }; }); };
  const handleChainBehaviorValueChange = (key: keyof WalkingEnginePivotOffsets, mode: 'b' | 's', value: string) => { const numValue = parseFloat(value); setJointChainBehaviors(prev => { const keyBehaviors = prev[key] || {}; return { ...prev, [key]: { ...keyBehaviors, [mode]: isNaN(numValue) ? 0 : numValue } }; }); };
  const blendModeOptions = ['normal', 'multiply', 'screen', 'overlay', 'darken', 'lighten', 'color-dodge', 'color-burn', 'hard-light', 'soft-light', 'difference', 'exclusion', 'hue', 'saturation', 'color', 'luminosity'];
  const ghostType = useMemo(() => { if (isTransitioning) return 'static'; if (previewPivotOffsets) return 'fk'; return null; }, [isTransitioning, previewPivotOffsets]);
  
  useEffect(() => { const newVels = { ...jointVelocitiesRef.current }; const finalPivotOffsets = previewPivotOffsets || pivotOffsets; JOINT_KEYS.forEach(key => { const delta = lerpAngleShortestPath(prevPivotOffsetsForVelRef.current[key], finalPivotOffsets[key], 1) - prevPivotOffsetsForVelRef.current[key]; newVels[key] = (newVels[key] * 0.2) + (delta * 0.8); }); jointVelocitiesRef.current = newVels; prevPivotOffsetsForVelRef.current = finalPivotOffsets; }, [previewPivotOffsets, pivotOffsets]);
  useEffect(() => { let animationFrameId: number; let lastFrameTime = performance.now(); const frameInterval = targetFps ? 1000 / targetFps : 0; const updateDisplayLoop = (currentTime: number) => { if (currentTime - lastFrameTime >= frameInterval) { lastFrameTime = currentTime - ((currentTime - lastFrameTime) % frameInterval); const poseForVelocityCalc = previewPivotOffsets || pivotOffsets; latestPivotOffsetsRef.current = poseForVelocityCalc; let poseToDisplay = { ...poseForVelocityCalc }; if (motionStyle === 'lotte') { const newVels = { ...jointVelocitiesRef.current }; const tailLengthDecayFactor = 1 - ((100 - jointFriction) / 100 * 0.2); JOINT_KEYS.forEach(key => { let totalNeighborVelocity = 0; const neighbors: (keyof WalkingEnginePivotOffsets)[] = []; const parent = JOINT_PARENT_MAP[key]; if (parent) neighbors.push(parent); if (key === 'collar') { neighbors.push('neck', 'l_shoulder', 'r_shoulder'); } else if (key === 'waist') { neighbors.push('torso', 'l_hip', 'r_hip'); } else if (JOINT_CHILD_MAP[key]) { neighbors.push(JOINT_CHILD_MAP[key]!); } neighbors.forEach(nKey => { totalNeighborVelocity += Math.abs(newVels[nKey]); }); const featherAmount = 0.5; if (totalNeighborVelocity > 0.2) { const rand = Math.random() - 0.5; const scaledFeather = Math.min(featherAmount, totalNeighborVelocity * 0.05) * rand; poseToDisplay[key] += scaledFeather; } newVels[key] *= tailLengthDecayFactor; if (Math.abs(newVels[key]) < 0.01) newVels[key] = 0; }); jointVelocitiesRef.current = newVels; } setDisplayedPivotOffsets(poseToDisplay); } animationFrameId = requestAnimationFrame(updateDisplayLoop); }; animationFrameId = requestAnimationFrame(updateDisplayLoop); return () => cancelAnimationFrame(animationFrameId); }, [targetFps, motionStyle, jointFriction, previewPivotOffsets, pivotOffsets]);
  const activeSelectionKeys = useMemo(() => { const selection = new Set<keyof WalkingEnginePivotOffsets>(); if (!selectedBoneKey) return selection; switch (selectionScope) { case 'part': selection.add(selectedBoneKey); break; case 'hierarchy': { selection.add(selectedBoneKey); let current = selectedBoneKey; while (JOINT_CHILD_MAP[current]) { const child = JOINT_CHILD_MAP[current]!; selection.add(child); current = child; } break; } case 'full': JOINT_KEYS.forEach(k => selection.add(k)); break; } return selection; }, [selectedBoneKey, selectionScope]);
  const displayPivotOffsetsForFKControls = isAnimating || isTransitioning ? displayedPivotOffsets : (previewPivotOffsets || pivotOffsets);
  const rotationControlLabel = primaryPin === 'waist' ? 'Body Rotation' : `Rotation @ ${primaryPin.replace(/_/g, ' ')}`;
  const frictionLabel = motionStyle === 'clockwork' ? 'Tick Rate' : motionStyle === 'lotte' ? 'Tail Length' : 'Joint Friction';
  const mannequinOffsets = useMemo(() => { if (isCalibrating) return pivotOffsets; if (motionStyle === 'lotte') return displayedPivotOffsets; if (predictiveGhostingEnabled && previewPivotOffsets) { return pivotOffsets; } return previewPivotOffsets || pivotOffsets; }, [isCalibrating, motionStyle, displayedPivotOffsets, pivotOffsets, previewPivotOffsets, predictiveGhostingEnabled]);
  const mannequinRenderPose = gifRenderPose || mannequinOffsets;

  return (
    <div className="flex h-full w-full bg-paper font-mono text-ink overflow-hidden select-none">
      {isConsoleVisible && (
        <div className="w-96 border-r border-ridge bg-mono-darker p-4 flex flex-col gap-4 custom-scrollbar overflow-y-auto z-50">
          <div className="flex justify-between items-center border-b border-ridge pb-2">
            <div className="flex items-center gap-2">
                <h1 className="text-2xl font-archaic tracking-widest text-ink uppercase italic">Bitruvius.Core</h1>
                <button onClick={() => setIsKeymapVisible(true)} title="8BitDo Controller Map" className="p-1 hover:bg-selection-super-light rounded transition-colors text-mono-mid hover:text-ink">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="6" y1="11" x2="10" y2="11"></line><line x1="8" y1="9" x2="8" y2="13"></line><line x1="15" y1="12" x2="15.01" y2="12"></line><line x1="18" y1="10" x2="18.01" y2="10"></line><path d="M17.32 5H6.68a4 4 0 0 0-3.97 3.59L2 15v4a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-4l-.71-6.41A4 4 0 0 0 17.32 5z"></path></svg>
                </button>
            </div>
            <div className="flex gap-1">
              <button onClick={undo} disabled={history.length === 0 || isAnimating || isTransitioning} title="Undo (Ctrl+Z)" className="p-1 hover:bg-selection-super-light disabled:opacity-20 rounded transition-colors">
                <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 10h10a8 8 0 0 1 8 8v2M3 10l6-6M3 10l6 6"/></svg>
              </button>
              <button onClick={redo} disabled={redoStack.length === 0 || isAnimating || isTransitioning} title="Redo (Ctrl+Y)" className="p-1 hover:bg-selection-super-light disabled:opacity-20 rounded transition-colors">
                <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 10H11a8 8 0 0 0-8 8v2M21 10l-6-6M21 10l-6 6"/></svg>
              </button>
            </div>
          </div>
          <div className="border-b border-ridge">
            <div className="flex">{(['fk', 'perf'] as const).map(tab => (<button key={tab} onClick={() => setActiveControlTab(tab)} className={`flex-1 text-sm py-2 font-bold transition-colors ${activeControlTab === tab ? 'bg-mono-dark text-selection border-b-2 border-selection' : 'text-mono-mid opacity-50'}`}>{tab.toUpperCase()}</button>))}</div>
            <div className="flex">{(['props', 'animation'] as const).map(tab => (<button key={tab} onClick={() => setActiveControlTab(tab)} className={`flex-1 text-sm py-2 font-bold transition-colors ${activeControlTab === tab ? 'bg-mono-dark text-selection border-b-2 border-selection' : 'text-mono-mid opacity-50'}`}>{tab.toUpperCase()}</button>))}</div>
          </div>
          <div className="flex-grow">
            {activeControlTab === 'fk' && (
              <div className="flex flex-col gap-4 pt-4 animate-in fade-in slide-in-from-left duration-200">
                <div>
                  <div className="text-xs font-bold text-mono-light uppercase border-b border-ridge pb-1">Ghosting</div>
                  <div className="flex flex-col gap-2 p-2 border border-ridge/50 rounded bg-white/30 mt-2">
                    <button onClick={() => { saveToHistory(); recordSnapshot(predictiveGhostingEnabled ? 'GHOST_OFF' : 'GHOST_ON'); setPredictiveGhostingEnabled(prev => !prev); }} className={`text-sm px-3 py-1 border transition-all ${predictiveGhostingEnabled ? 'bg-accent-green text-paper border-accent-green' : 'bg-paper/10 text-mono-mid border-ridge'}`} disabled={isTransitioning}> PREDICTIVE GHOST: {predictiveGhostingEnabled ? 'ON' : 'OFF'} </button>
                  </div>
                </div>
                <div className="flex flex-col gap-3">
                  <div className="flex flex-col gap-3">
                    <button onClick={() => setFixedPose(T_POSE, 'T-Pose')} className="text-sm px-3 py-2 border border-selection bg-selection text-paper font-bold hover:bg-selection-light transition-all uppercase tracking-widest text-center" disabled={isTransitioning}>ALIGN T-POSE</button>
                    <button onClick={() => { setActivePins([]); setPinTargetPositions({}); addLog("PINS CLEARED."); }} className={`text-sm px-3 py-1 border transition-all bg-paper/10 text-mono-mid border-ridge disabled:opacity-50`} disabled={activePins.length === 0}> CLEAR PINS ({activePins.length}) </button>
                  </div>
                </div>
                <div className="text-xs font-bold text-mono-light uppercase border-b border-ridge pb-1">Root Controls</div>
                 <div className="p-2 border border-ridge/50 rounded bg-white/30 space-y-2">
                    <div className="flex justify-between text-xs uppercase font-bold text-mono-light"><span>Root X</span><span>{physicsState.position.x.toFixed(0)}</span></div>
                    <input type="range" min="-500" max="500" step="1" value={physicsState.position.x} onChange={(e) => setPhysicsState(p => ({...p, position: { ...p.position, x: parseInt(e.target.value) }}))} onMouseDown={() => {saveToHistory(); recordSnapshot('START_ROOT_X');}} onMouseUp={() => recordSnapshot('END_ROOT_X')} className="w-full accent-selection h-1 cursor-ew-resize" disabled={isTransitioning}/>
                    <div className="flex justify-between text-xs uppercase font-bold text-mono-light"><span>Root Y</span><span>{physicsState.position.y.toFixed(0)}</span></div>
                    <input type="range" min="-700" max="700" step="1" value={physicsState.position.y} onChange={(e) => setPhysicsState(p => ({...p, position: { ...p.position, y: parseInt(e.target.value) }}))} onMouseDown={() => {saveToHistory(); recordSnapshot('START_ROOT_Y');}} onMouseUp={() => recordSnapshot('END_ROOT_Y')} className="w-full accent-selection h-1 cursor-ew-resize" disabled={isTransitioning}/>
                    <div className="flex justify-between text-xs uppercase font-bold text-mono-light"><span className="truncate">{rotationControlLabel}</span><span>{bodyRotation.toFixed(0)}°</span></div>
                    <input type="range" min="-180" max="180" step="1" value={bodyRotation} onChange={(e) => setBodyRotation(parseInt(e.target.value))} onMouseDown={() => {saveToHistory(); recordSnapshot('START_BODY_ROT');}} onMouseUp={() => recordSnapshot('END_BODY_ROT')} className="w-full accent-selection h-1 cursor-ew-resize" disabled={isTransitioning}/>
                 </div>
                <div className="text-xs font-bold text-mono-light uppercase border-b border-ridge pb-1">System Friction</div>
                <div className="p-2 border border-ridge/50 rounded bg-white/30 space-y-1">
                    <div className="flex justify-between text-xs uppercase text-mono-light"><span>{frictionLabel}</span><span>{jointFriction}%</span></div>
                    <input type="range" min="0" max="100" step="1" value={jointFriction} onChange={e => setJointFriction(parseInt(e.target.value))} className="w-full h-1 accent-selection cursor-ew-resize" />
                    <p className="text-[10px] text-mono-light italic text-center pt-1"> { motionStyle === 'clockwork' && "Controls jitter intensity at end of ticks."} { motionStyle === 'lotte' && "Controls duration of feathering effect."} { motionStyle === 'standard' && "Controls drag resistance and pose settling."} </p>
                </div>
                <div className="text-xs font-bold text-mono-light uppercase border-b border-ridge pb-1">Skeletal Rotations</div>
                <div ref={fkControlsRef} className="flex flex-col gap-2 pr-2 h-[400px] overflow-y-auto custom-scrollbar">
                  {JOINT_KEYS.map(k => {
                    const isSelected = k === selectedBoneKey;
                    const behaviors = jointChainBehaviors[k] || {};
                    const isBActive = behaviors.b != null && behaviors.b !== 0;
                    const isSActive = behaviors.s != null && behaviors.s !== 0;
                    return (
                    <div key={k} data-joint-key={k} className={`group p-1 rounded-sm transition-colors ${isSelected ? 'bg-selection-super-light' : ''}`}>
                      <div className="flex justify-between items-center text-sm uppercase font-bold text-mono-light group-hover:text-ink transition-colors mb-1">
                        <span className="truncate pr-2">{k.replace(/_/g, ' ')}</span>
                         <div className="flex items-center gap-2">
                            <div className="flex items-center border border-ridge rounded-sm overflow-hidden"> <button onClick={() => handleChainBehaviorToggle(k, 'b')} title="Bend: Child follows parent's rotation to flex" className={`w-5 h-5 text-xs transition-colors ${isBActive ? 'bg-accent-green text-paper' : 'bg-paper/10 text-mono-mid'}`}>B</button> <input type="number" value={isBActive ? behaviors.b : ''} onChange={(e) => handleChainBehaviorValueChange(k, 'b', e.target.value)} disabled={!isBActive} className="w-10 h-5 text-xs p-0 text-center bg-transparent text-mono-light outline-none disabled:text-mono-mid/50" step="0.1" /> </div>
                             <div className="flex items-center border border-ridge rounded-sm overflow-hidden"> <button onClick={() => handleChainBehaviorToggle(k, 's')} title="Stretch: Child counter-rotates to straighten" className={`w-5 h-5 text-xs transition-colors ${isSActive ? 'bg-accent-purple text-paper' : 'bg-paper/10 text-mono-mid'}`}>S</button> <input type="number" value={isSActive ? behaviors.s : ''} onChange={(e) => handleChainBehaviorValueChange(k, 's', e.target.value)} disabled={!isSActive} className="w-10 h-5 text-xs p-0 text-center bg-transparent text-mono-light outline-none disabled:text-mono-mid/50" step="0.1" /> </div>
                            <button onClick={() => handleChainBehaviorToggle(k, 'l')} title="Lead: Drag to rotate the parent bone" className={`w-5 h-5 text-xs rounded border transition-colors ${behaviors.l ? 'bg-accent-orange text-paper border-accent-orange' : 'bg-paper/10 text-mono-mid border-ridge'}`}>L</button>
                            <span className="w-12 text-right">{Math.round(displayPivotOffsetsForFKControls[k])}°</span>
                         </div>
                      </div>
                      <input type="range" min="-180" max="180" step="1" disabled={isAnimating || isTransitioning || activePins.includes(k)} value={displayPivotOffsetsForFKControls[k]} onMouseDown={() => { saveToHistory(); recordSnapshot(`START_RANGE_${k}`); isSliderDraggingRef.current = true; draggingBoneKeyRef.current = k; if (predictiveGhostingEnabled) { setPreviewPivotOffsets({ ...pivotOffsets }); setStaticGhostPose({ ...pivotOffsets }); } setSelectedKeyframeIndex(null); }} onChange={(e) => handlePivotChange(k, parseInt(e.target.value))} onMouseUp={handleInteractionEnd} className="w-full accent-selection h-1 cursor-ew-resize disabled:opacity-50" />
                    </div>
                  )})}
                </div>
              </div>
            )}
            {activeControlTab === 'perf' && ( <div className="flex flex-col gap-4 pt-4 animate-in fade-in slide-in-from-right duration-200"> <div className="text-xs font-bold text-mono-light uppercase border-b border-ridge pb-1">Environment</div> <div className="flex flex-col gap-2 p-2 border border-ridge/50 rounded bg-white/30"> <button onClick={() => setHardStopEnabled(p => !p)} className={`text-sm px-3 py-1 border transition-all ${hardStopEnabled ? 'bg-accent-red text-paper border-accent-red' : 'bg-paper/10 text-mono-mid border-ridge'}`} > HARD STOP: {hardStopEnabled ? 'ON' : 'OFF'} </button> <p className="text-[10px] text-mono-light italic text-center pt-1">When ON, prevents pins stretching beyond 20px.</p> </div> <div className="text-xs font-bold text-mono-light uppercase border-b border-ridge pb-1">Intent Preview</div> <div className="flex flex-col gap-2 p-2 border border-ridge/50 rounded bg-white/30"> <button onClick={() => { saveToHistory(); recordSnapshot(showIntentPath ? 'INTENT_PATH_OFF' : 'INTENT_PATH_ON'); setShowIntentPath(prev => !prev); }} className={`text-sm px-3 py-1 border transition-all ${showIntentPath ? 'bg-accent-green text-paper border-accent-green' : 'bg-paper/10 text-mono-mid border-ridge'}`} disabled={isTransitioning}> PREVIEW DEPTH: {showIntentPath ? '5 FRAMES' : 'OFF'} </button> </div> <div className="text-xs font-bold text-mono-light uppercase border-b border-ridge pb-1">Motion Style</div> <div className="p-2 border border-ridge/50 rounded bg-white/30 space-y-2"> <div className="grid grid-cols-3 gap-1"> {(['standard', 'clockwork', 'lotte'] as MotionStyle[]).map(style => ( <button key={style} onClick={() => { setMotionStyle(style); if(style !== 'standard') setTargetFps(12); else setTargetFps(null); }} className={`text-xs px-2 py-1 border uppercase font-bold transition-all ${motionStyle === style ? 'bg-selection text-paper border-selection' : 'bg-paper/10 text-mono-mid border-ridge'}`}> {style} </button> ))} </div> </div> <div className="text-xs font-bold text-mono-light uppercase border-b border-ridge pb-1">Rendering FPS</div> <div className="p-2 border border-ridge/50 rounded bg-white/30 space-y-2"> <div className="grid grid-cols-3 gap-1"> <button onClick={() => setTargetFps(null)} className={`text-xs px-2 py-1 border uppercase font-bold transition-all ${!targetFps ? 'bg-selection text-paper border-selection' : 'bg-paper/10 text-mono-mid border-ridge'}`}>Max</button> <button onClick={() => setTargetFps(24)} className={`text-xs px-2 py-1 border uppercase font-bold transition-all ${targetFps === 24 ? 'bg-selection text-paper border-selection' : 'bg-paper/10 text-mono-mid border-ridge'}`}>24</button> <button onClick={() => setTargetFps(12)} className={`text-xs px-2 py-1 border uppercase font-bold transition-all ${targetFps === 12 ? 'bg-selection text-paper border-selection' : 'bg-paper/10 text-mono-mid border-ridge'}`}>12</button> </div> </div> </div> )}
            {activeControlTab === 'props' && ( <div className="flex flex-col gap-4 animate-in fade-in slide-in-from-right duration-200"> <div className="text-xs font-bold text-mono-light uppercase border-b border-ridge pb-1 flex justify-between items-center"> <span>Anatomical Resizing</span> <button onClick={resetProps} disabled={isAnimating || isTransitioning} className="text-xs text-selection hover:underline disabled:opacity-50">RESET</button> </div> <div className="flex flex-col gap-4 pr-2 h-[400px] overflow-y-auto custom-scrollbar"> {PROP_KEYS.map(k => ( <div key={k} className="p-2 border border-ridge/50 rounded bg-white/30 space-y-2"> <div className="text-sm font-bold uppercase text-ink">{k.replace(/_/g, ' ')}</div> <div className="space-y-1"> <div className="flex justify-between text-xs uppercase text-mono-light"><span>Height Scale</span><span>{props[k].h.toFixed(2)}x</span></div> <input type="range" min="0.2" max="3" step="0.01" value={props[k].h} disabled={isAnimating || isTransitioning} onMouseDown={() => {saveToHistory(); recordSnapshot(`START_PROP_H_${k}`);}} onChange={e => updateProp(k, 'h', parseFloat(e.target.value))} onMouseUp={() => recordSnapshot(`END_PROP_H_${k}`)} className="w-full h-1 accent-mono-mid disabled:opacity-50" /> </div> <div className="space-y-1"> <div className="flex justify-between text-xs uppercase text-mono-light"><span>Width Scale</span><span>{props[k].w.toFixed(2)}x</span></div> <input type="range" min="0.2" max="3" step="0.01" value={props[k].w} disabled={isAnimating || isTransitioning} onMouseDown={() => {saveToHistory(); recordSnapshot(`START_PROP_W_${k}`);}} onChange={e => updateProp(k, 'w', parseFloat(e.target.value))} onMouseUp={() => recordSnapshot(`END_PROP_W_${k}`)} className="w-full h-1 accent-mono-mid disabled:opacity-50" /> </div> </div> ))} </div> </div> )}
            {activeControlTab === 'animation' && ( <div className="flex flex-col gap-4 animate-in fade-in duration-200"> <div className="text-xs font-bold text-mono-light uppercase border-b border-ridge pb-1">Animation Timeline</div> <Timeline keyframes={keyframes} onPlay={handlePlay} onPause={handlePause} onReset={handleFullAnimationReset} isAnimating={isAnimating} animationTime={animationTime} totalDuration={totalDuration} onAddKeyframe={handleAddKeyframe} onSelectKeyframe={handleSelectKeyframe} selectedKeyframeIndex={selectedKeyframeIndex} onScrub={handleScrub} onUpdateKeyframeTime={handleUpdateKeyframeTime} onAddTweenFrame={handleAddTweenFrame} /> <div className="text-xs font-bold text-mono-light uppercase border-b border-ridge pb-1 mt-2">Onion Skinning</div><div className="p-2 border border-ridge/50 rounded bg-white/30 space-y-2"><button onClick={() => setShowOnionSkins(p => !p)} className={`w-full text-sm px-3 py-1 border transition-all ${showOnionSkins ? 'bg-accent-green text-paper border-accent-green' : 'bg-paper/10 text-mono-mid border-ridge'}`}> ONION SKINS: {showOnionSkins ? 'ON' : 'OFF'} </button><div className="grid grid-cols-2 gap-4 pt-1"><div className="flex flex-col gap-1"><label className="text-xs uppercase font-bold text-mono-light">Before</label><input type="number" min="0" max="5" value={onionSkinFrames.before} onChange={e => setOnionSkinFrames(f => ({...f, before: parseInt(e.target.value) || 0}))} className="w-full bg-paper/50 border border-ridge/50 p-1 text-center rounded-sm" /></div><div className="flex flex-col gap-1"><label className="text-xs uppercase font-bold text-mono-light">After</label><input type="number" min="0" max="5" value={onionSkinFrames.after} onChange={e => setOnionSkinFrames(f => ({...f, after: parseInt(e.target.value) || 0}))} className="w-full bg-paper/50 border border-ridge/50 p-1 text-center rounded-sm" /></div></div> {isCapsLockOn && <div className="text-center text-accent-orange font-bold text-xs pt-1 animate-pulse tracking-widest">POSE-SELECT ACTIVE</div>}</div><div className="text-xs font-bold text-mono-light uppercase border-b border-ridge pb-1 mt-2">Export</div><div className="p-2 border border-ridge/50 rounded bg-white/30 space-y-3"><div className="flex items-center justify-between"><label htmlFor="transparent-bg" className="text-sm text-ink">Transparent Background</label><button id="transparent-bg" onClick={() => setExportTransparentBg(p => !p)} className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${exportTransparentBg ? 'bg-accent-green' : 'bg-gray-200'}`}><span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${exportTransparentBg ? 'translate-x-6' : 'translate-x-1'}`}/></button></div><div className="flex items-center justify-between"><label htmlFor="show-anchors" className="text-sm text-ink">Show Anchors</label><button id="show-anchors" onClick={() => setExportShowAnchors(p => !p)} className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${exportShowAnchors ? 'bg-accent-green' : 'bg-gray-200'}`}><span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${exportShowAnchors ? 'translate-x-6' : 'translate-x-1'}`}/></button></div><div className="grid grid-cols-2 gap-2"> <button onClick={handleExportPNG} disabled={isExportingGif} className="w-full text-sm px-3 py-2 border border-ridge font-bold bg-selection text-paper hover:bg-selection-light transition-colors disabled:opacity-50">EXPORT PNG</button> <button onClick={handleExportGIF} disabled={isExportingGif || keyframes.length < 2} className="w-full text-sm px-3 py-2 border border-ridge font-bold bg-selection text-paper hover:bg-selection-light transition-colors disabled:opacity-50">EXPORT GIF</button> </div> {isExportingGif && (<div className="space-y-1"><div className="text-xs text-center text-ink">Generating GIF... ({gifExportProgress.toFixed(0)}%)</div><div className="w-full bg-gray-200 rounded-full h-1.5"><div className="bg-accent-green h-1.5 rounded-full" style={{width: `${gifExportProgress}%`}}></div></div></div>)}</div></div> )}
          </div>
          <div className="flex flex-col gap-4 pt-4 border-t border-ridge"> <div className="text-xs font-bold text-mono-light uppercase border-b border-ridge pb-1">Serialization</div> <textarea readOnly value={poseString} className="w-full text-sm bg-white border border-ridge p-2 font-mono custom-scrollbar resize-none h-24" /> <div className="flex flex-col gap-2"> <button onClick={copyToClipboard} className="w-full text-sm px-3 py-2 border border-ridge font-bold bg-selection text-paper hover:bg-selection-light transition-colors">COPY STATE STRING</button> <button onClick={saveToFile} className="w-full text-sm px-3 py-2 border border-ridge font-bold text-mono-mid hover:bg-mono-dark transition-colors">EXPORT FILE</button> </div> </div>
          <SystemLogger logs={recordingHistory} isVisible={true} onExportJSON={exportRecordingJSON} onClearHistory={clearHistory} historyCount={recordingHistory.length} onLogMouseEnter={setOnionSkinData} onLogMouseLeave={() => setOnionSkinData(null)} onLogClick={handleLogClick} selectedLogIndex={selectedLogIndex} />
          <div id="mask-controls-placeholder" className="pt-4 border-t border-ridge bg-white/10 p-2 rounded"> <div className="text-xs font-bold text-mono-light uppercase mb-2">Mask Overlay</div> <input type="file" accept="image/*" onChange={handleMaskUpload} className="hidden" id="mask-upload" /> <label htmlFor="mask-upload" className="block text-center text-sm px-3 py-2 border border-ridge font-bold cursor-pointer hover:bg-mono-dark transition-colors mb-2 uppercase">{maskImage ? "Change Mask" : "Upload Mask"}</label> {maskImage && ( <div className="space-y-2 animate-in fade-in"><div className="grid grid-cols-2 gap-2"><div><div className="flex justify-between text-xs uppercase text-mono-light"><span>X Offset</span></div><input type="range" min="-100" max="100" value={maskTransform.x} onChange={e => setMaskTransform(t => ({...t, x: parseInt(e.target.value)}))} className="w-full h-1 accent-selection" /></div><div><div className="flex justify-between text-xs uppercase text-mono-light"><span>Y Offset</span></div><input type="range" min="-100" max="100" value={maskTransform.y} onChange={e => setMaskTransform(t => ({...t, y: parseInt(e.target.value)}))} className="w-full h-1 accent-selection" /></div></div><div><div className="flex justify-between text-xs uppercase text-mono-light"><span>Rotation</span><span>{maskTransform.rotation}°</span></div><input type="range" min="-180" max="180" value={maskTransform.rotation} onChange={e => setMaskTransform(t => ({...t, rotation: parseInt(e.target.value)}))} className="w-full h-1 accent-selection" /></div><div><div className="flex justify-between text-xs uppercase text-mono-light"><span>Scale</span><span>{maskTransform.scale.toFixed(2)}x</span></div><input type="range" min="0.01" max="10" step="0.01" value={maskTransform.scale} onChange={e => setMaskTransform(t => ({...t, scale: parseFloat(e.target.value)}))} className="w-full h-1 accent-selection" /></div><button onClick={() => setMaskImage(null)} className="w-full text-xs text-accent-red font-bold hover:underline py-1 uppercase">Remove Mask</button></div>)} </div>
          <div id="background-controls-placeholder" className="pt-4 border-t border-ridge bg-white/10 p-2 rounded"> <div className="text-xs font-bold text-mono-light uppercase mb-2">Background Image</div> <input type="file" accept="image/*" onChange={handleBackgroundUpload} className="hidden" id="background-upload" /> <label htmlFor="background-upload" className="block text-center text-sm px-3 py-2 border border-ridge font-bold cursor-pointer hover:bg-mono-dark transition-colors mb-2 uppercase">{backgroundImage ? "Change BG" : "Upload BG"}</label> {backgroundImage && ( <div className="space-y-2 animate-in fade-in"><div className="grid grid-cols-2 gap-2"><div><div className="flex justify-between text-xs uppercase text-mono-light"><span>X Offset</span></div><input type="range" min="-500" max="500" value={backgroundTransform.x} onChange={e => setBackgroundTransform(t => ({...t, x: parseInt(e.target.value)}))} className="w-full h-1 accent-selection" /></div><div><div className="flex justify-between text-xs uppercase text-mono-light"><span>Y Offset</span></div><input type="range" min="-500" max="500" value={backgroundTransform.y} onChange={e => setBackgroundTransform(t => ({...t, y: parseInt(e.target.value)}))} className="w-full h-1 accent-selection" /></div></div><div><div className="flex justify-between text-xs uppercase text-mono-light"><span>Rotation</span><span>{backgroundTransform.rotation}°</span></div><input type="range" min="-180" max="180" value={backgroundTransform.rotation} onChange={e => setBackgroundTransform(t => ({...t, rotation: parseInt(e.target.value)}))} className="w-full h-1 accent-selection" /></div><div><div className="flex justify-between text-xs uppercase text-mono-light"><span>Scale</span><span>{maskTransform.scale.toFixed(2)}x</span></div><input type="range" min="0.1" max="10" step="0.05" value={backgroundTransform.scale} onChange={e => setBackgroundTransform(t => ({...t, scale: parseFloat(e.target.value)}))} className="w-full h-1 accent-selection" /></div><div><div className="flex justify-between text-xs uppercase text-mono-light mb-1"><span>Blend Mode</span></div><select value={blendMode} onChange={e => setBlendMode(e.target.value)} className="w-full p-1 text-sm bg-white/50 border border-ridge/50 rounded-sm focus:ring-1 focus:ring-selection focus:border-selection outline-none">{blendModeOptions.map(mode => ( <option key={mode} value={mode} className="capitalize">{mode.replace(/-/g, ' ')}</option> ))}</select></div><button onClick={() => setBackgroundImage(null)} className="w-full text-xs text-accent-red font-bold hover:underline py-1 uppercase">Remove BG</button></div>)} </div>
        </div>
      )}
      {isKeymapVisible && <KeymapHelper onClose={() => setIsKeymapVisible(false)} />}
      <div className={`flex-1 relative flex items-center justify-center bg-paper p-8 overflow-hidden transition-all duration-500 ${isAnimating ? 'cursor-wait' : (!isCalibrated && !isCalibrating ? 'cursor-pointer group/stage' : '')}`} onClick={() => !isCalibrated && !isCalibrating && startCalibration()}>
        {isAnimating && <div className="absolute top-4 right-4 z-50 px-3 py-1 bg-selection text-paper text-sm font-bold tracking-[0.2em] animate-pulse rounded-sm border border-ridge/50">ANIMATING...</div>}
        {isTransitioning && <div className="absolute top-4 right-4 z-50 px-3 py-1 bg-accent-purple text-paper text-sm font-bold tracking-[0.2em] animate-pulse rounded-sm border border-ridge/50">TRANSITIONING...</div>}
        <button onClick={(e) => { e.stopPropagation(); setIsConsoleVisible(!isConsoleVisible); }} disabled={!isCalibrated} className={`absolute top-4 left-4 z-50 p-2 rounded-full transition-all shadow-sm border ${!isCalibrated ? 'bg-mono-dark text-mono-light opacity-30 cursor-not-allowed border-ridge' : 'bg-mono-darker/50 text-ink hover:bg-selection-super-light border-ridge'}`}> <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={isConsoleVisible ? "M6 18L18 6M6 6l12 12" : "M4 6h16M4 12h16M4 18h16"} /></svg> </button>
        {!isCalibrated && !isCalibrating && ( <div className="absolute inset-0 z-[100] flex flex-col items-center justify-end pb-16 md:pb-24 px-4 bg-paper/5 pointer-events-none animate-in fade-in duration-700"><h2 className="text-6xl md:text-8xl font-archaic text-ink tracking-tighter leading-none uppercase text-center animate-in slide-in-from-bottom-8 duration-1000">Bitruvian Posing Engine</h2></div>)}
        <svg ref={svgRef} viewBox="-500 -700 1000 1400" className={`w-full h-full drop-shadow-xl overflow-visible relative z-0 transition-all duration-300 ${!isCalibrated ? 'scale-110' : ''}`}>
          {backgroundImage && (<image id="background-image-renderer" href={backgroundImage} x="-500" y="-500" width="1000" height="1000" preserveAspectRatio="xMidYMid slice" transform={`translate(${backgroundTransform.x}, ${backgroundTransform.y}) rotate(${backgroundTransform.rotation}) scale(${backgroundTransform.scale})`} className="pointer-events-none" /> )}
          <g transform={`translate(${physicsState.position.x}, ${physicsState.position.y})`} className="relative z-10">
              <g transform={`rotate(${bodyRotation}, ${pinnedJointPosition.x}, ${pinnedJointPosition.y})`} className={backgroundImage ? `mix-blend-${blendMode}` : ''}>
                 {!isExportingGif && onionSkinPoses.map((skin, i) => (
                    <Mannequin 
                        key={`onion-${skin.index}-${i}`}
                        pose={RESTING_BASE_POSE}
                        pivotOffsets={skin.pose}
                        props={props}
                        isGhost={true}
                        ghostType={'static'}
                        ghostOpacity={skin.opacity}
                        showPivots={false} showLabels={false} baseUnitH={baseH}
                        onAnchorMouseDown={()=>{}} onBodyMouseDown={()=>{}}
                        draggingBoneKey={null} selectedBoneKeys={new Set()}
                        isPaused={true} activePins={[]} limbTensions={{}}
                        isInteractive={isCapsLockOn}
                        onClick={() => handleSelectKeyframe(skin.index)}
                    />
                 ))}
                 {!isExportingGif && staticGhostPose && ( <Mannequin pose={RESTING_BASE_POSE} pivotOffsets={staticGhostPose} props={props} isGhost={true} ghostType={'static'} ghostOpacity={0.4} showPivots={false} showLabels={false} baseUnitH={baseH} onAnchorMouseDown={()=>{}} onBodyMouseDown={()=>{}} draggingBoneKey={null} selectedBoneKeys={new Set()} isPaused={true} activePins={[]} limbTensions={limbTensions} pinsAtLimit={pinsAtLimit} /> )}
                 {!isExportingGif && predictiveGhostingEnabled && ghostType === 'fk' && previewPivotOffsets && ( <> {showIntentPath && staticGhostPose && draggingBoneKeyRef.current && Array.from({ length: 5 }).map((_, i) => { const t = (i + 1) / 5; const draggedKey = draggingBoneKeyRef.current!; const startValue = staticGhostPose[draggedKey]!; const endValue = previewPivotOffsets[draggedKey]!; const interpolatedValue = lerpAngleShortestPath(startValue, endValue, t); const delta = interpolatedValue - startValue; const basePoseForStep = { ...staticGhostPose, [draggedKey]: interpolatedValue }; const finalInterpolatedOffsets = applyChainReaction(draggedKey, delta, basePoseForStep); const opacity = 0.1 + t * 0.5; return ( <Mannequin key={`ghost-path-${i}`} pose={RESTING_BASE_POSE} pivotOffsets={finalInterpolatedOffsets} props={props} isGhost={true} ghostType={'fk'} ghostOpacity={opacity} showPivots={false} showLabels={false} baseUnitH={baseH} onAnchorMouseDown={()=>{}} onBodyMouseDown={()=>{}} draggingBoneKey={null} selectedBoneKeys={new Set()} isPaused={true} activePins={[]} limbTensions={limbTensions} pinsAtLimit={pinsAtLimit} /> ); })} {!showIntentPath && ( <Mannequin pose={RESTING_BASE_POSE} pivotOffsets={previewPivotOffsets} props={props} isGhost={true} ghostType={'fk'} showPivots={false} showLabels={false} baseUnitH={baseH} onAnchorMouseDown={()=>{}} onBodyMouseDown={()=>{}} draggingBoneKey={null} selectedBoneKeys={new Set()} isPaused={true} activePins={[]} limbTensions={limbTensions} pinsAtLimit={pinsAtLimit} /> )} </> )}
                 {!isExportingGif && onionSkinData && !previewPivotOffsets && !showOnionSkins && ( <Mannequin pose={RESTING_BASE_POSE} pivotOffsets={onionSkinData.pivotOffsets} props={onionSkinData.props} isGhost={true} ghostType={'static'} showPivots={false} showLabels={false} baseUnitH={baseH} onAnchorMouseDown={()=>{}} onBodyMouseDown={()=>{}} draggingBoneKey={null} selectedBoneKeys={new Set()} isPaused={true} activePins={[]} /> )}
                <Mannequin pose={RESTING_BASE_POSE} pivotOffsets={mannequinRenderPose} props={props} showPivots={isCalibrated && !isExportingGif} showLabels={showLabels} baseUnitH={baseH} onAnchorMouseDown={onAnchorMouseDown} onBodyMouseDown={handleBodyMouseDown} draggingBoneKey={draggingBoneKey} selectedBoneKeys={activeSelectionKeys} isPaused={true} maskImage={maskImage} maskTransform={maskTransform} onPositionsUpdate={setAllJointPositions} activePins={activePins} limbTensions={limbTensions} pinsAtLimit={pinsAtLimit} />
              </g>
          </g>
        </svg>
      </div>
    </div>
  );
};
