import { useState, useRef, useCallback, useMemo, useEffect } from 'react';
import { Keyframe, WalkingEnginePivotOffsets, EasingType } from '../types';
import { JOINT_KEYS } from '../constants';
import { lerpAngleShortestPath, easeOutCubic, easeInOutCubic, linear } from '../utils/kinematics';

const LOOP_DURATION = 1000; // Fixed duration for the loop from the last frame back to the first.

const easingFunctions: Record<EasingType, (t: number) => number> = {
    'linear': linear,
    'ease-out': easeOutCubic,
    'ease-in-out': easeInOutCubic,
};

export const useAnimationEngine = (
    keyframes: Keyframe[], 
    onPoseUpdate: (pose: WalkingEnginePivotOffsets) => void
) => {
    const [isAnimating, setIsAnimating] = useState(false);
    const [animationTime, setAnimationTime] = useState(0);
    
    const animationFrameId = useRef<number | null>(null);
    const lastTimestamp = useRef<number | null>(null);

    const totalDuration = useMemo(() => {
        if (keyframes.length < 2) return 0;
        return keyframes[keyframes.length - 1].time + LOOP_DURATION;
    }, [keyframes]);

    const runAnimationLoop = useCallback((timestamp: number) => {
        if (keyframes.length < 2 || totalDuration === 0) {
            setIsAnimating(false);
            return;
        }

        if (lastTimestamp.current === null) {
            lastTimestamp.current = timestamp;
        }
        const deltaTime = timestamp - lastTimestamp.current;
        lastTimestamp.current = timestamp;

        setAnimationTime(prevTime => {
            let newTime = prevTime + deltaTime;
            newTime %= totalDuration;
            
            let startKeyframe: Keyframe;
            let endPose: WalkingEnginePivotOffsets;
            let segmentStartTime: number;
            let segmentDuration: number;

            // Find the current segment
            let segmentFound = false;
            for (let i = 0; i < keyframes.length - 1; i++) {
                if (newTime >= keyframes[i].time && newTime < keyframes[i+1].time) {
                    startKeyframe = keyframes[i];
                    endPose = keyframes[i+1].pose;
                    segmentStartTime = keyframes[i].time;
                    segmentDuration = keyframes[i+1].time - keyframes[i].time;
                    segmentFound = true;
                    break;
                }
            }

            // If not found, it must be in the last segment (looping back to start)
            if (!segmentFound) {
                const lastKeyframe = keyframes[keyframes.length - 1];
                startKeyframe = lastKeyframe;
                endPose = keyframes[0].pose;
                segmentStartTime = lastKeyframe.time;
                segmentDuration = LOOP_DURATION;
            }
            
            const timeIntoSegment = newTime - segmentStartTime;
            const progress = segmentDuration > 0 ? timeIntoSegment / segmentDuration : 1;

            const easingType = startKeyframe!.easing || 'linear';
            const easingFunc = easingFunctions[easingType] || linear;
            const easedProgress = easingFunc(progress);
            
            const nextPose = { ...startKeyframe!.pose } as WalkingEnginePivotOffsets;
            JOINT_KEYS.forEach(k => {
                const start = startKeyframe!.pose[k];
                const end = endPose[k];
                nextPose[k] = lerpAngleShortestPath(start, end, easedProgress);
            });
            onPoseUpdate(nextPose);

            return newTime;
        });

        animationFrameId.current = requestAnimationFrame(runAnimationLoop);
    }, [keyframes, totalDuration, onPoseUpdate]);

    useEffect(() => {
        if (isAnimating) {
            lastTimestamp.current = null;
            animationFrameId.current = requestAnimationFrame(runAnimationLoop);
        } else {
            if (animationFrameId.current) {
                cancelAnimationFrame(animationFrameId.current);
            }
        }
        return () => {
            if (animationFrameId.current) {
                cancelAnimationFrame(animationFrameId.current);
            }
        };
    }, [isAnimating, runAnimationLoop]);

    const play = useCallback(() => {
        if (keyframes.length < 2) return;
        setIsAnimating(true);
    }, [keyframes.length]);
    
    const pause = useCallback(() => {
        setIsAnimating(false);
    }, []);

    const reset = useCallback(() => {
        setIsAnimating(false);
        setAnimationTime(0);
        if (keyframes.length > 0) {
            onPoseUpdate(keyframes[0].pose);
        }
    }, [keyframes, onPoseUpdate]);

    return {
        isAnimating,
        animationTime,
        totalDuration,
        play,
        pause,
        reset,
        setAnimationTime,
    };
};