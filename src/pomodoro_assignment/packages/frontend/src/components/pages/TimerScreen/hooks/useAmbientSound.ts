import { useEffect, useRef, useState, useCallback } from 'react';
import { Howl } from 'howler';

export type AmbientSoundType = 'forest' | 'ocean' | 'cafe' | 'rain' | 'none';

export interface UseAmbientSoundOptions {
  initialSound?: AmbientSoundType;
  initialVolume?: number;
  autoLoad?: boolean;
  loop?: boolean;
  fadeInDuration?: number;
  fadeOutDuration?: number;
}

export interface UseAmbientSoundReturn {
  // Playback controls
  play: (soundType?: AmbientSoundType) => void;
  pause: () => void;
  stop: (fadeOut?: boolean) => void;
  setVolume: (volume: number) => void;
  changeSound: (soundType: AmbientSoundType) => void;

  // State
  isPlaying: boolean;
  isLoading: boolean;
  currentSound: AmbientSoundType;
  volume: number;
  error: string | null;

  // Timer integration
  playWithTimer: (soundType: AmbientSoundType, timerMode?: boolean) => void;
  stopWithTimer: (fadeOut?: boolean) => void;
}

// Sound file paths - update these paths when real audio files are added
const SOUND_PATHS: Record<AmbientSoundType, string | null> = {
  forest: '/sounds/forest.mp3',
  ocean: '/sounds/ocean.mp3',
  cafe: '/sounds/cafe.mp3',
  rain: '/sounds/rain.mp3',
  none: null,
};

// Temporary sound generator using Web Audio API for demo purposes
const generateAmbientSound = (type: AmbientSoundType): AudioBuffer | null => {
  if (typeof window === 'undefined' || !window.AudioContext) return null;

  const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
  const sampleRate = audioContext.sampleRate;
  const duration = 3; // 3 seconds
  const length = sampleRate * duration;
  const buffer = audioContext.createBuffer(2, length, sampleRate);

  for (let channel = 0; channel < 2; channel++) {
    const channelData = buffer.getChannelData(channel);

    switch (type) {
      case 'rain':
        // Generate white noise for rain
        for (let i = 0; i < length; i++) {
          channelData[i] = (Math.random() - 0.5) * 0.1;
        }
        break;

      case 'ocean':
        // Generate low frequency waves for ocean
        for (let i = 0; i < length; i++) {
          const t = i / sampleRate;
          channelData[i] = Math.sin(2 * Math.PI * 0.1 * t) * 0.05 * Math.random();
        }
        break;

      case 'forest':
        // Generate chirping-like sounds for forest
        for (let i = 0; i < length; i++) {
          const t = i / sampleRate;
          const chirp = Math.sin(2 * Math.PI * (1000 + Math.random() * 2000) * t) * 0.02;
          channelData[i] = chirp * (Math.random() < 0.01 ? 1 : 0.1);
        }
        break;

      case 'cafe':
        // Generate brown noise for cafe ambiance
        for (let i = 0; i < length; i++) {
          const t = i / sampleRate;
          channelData[i] = (Math.random() - 0.5) * 0.15 * Math.exp(-t * 0.5);
        }
        break;

      default:
        return null;
    }
  }

  return buffer;
};

export const useAmbientSound = (options: UseAmbientSoundOptions = {}): UseAmbientSoundReturn => {
  const {
    initialSound = 'none',
    initialVolume = 0.5,
    autoLoad = true,
    loop = true,
    fadeInDuration = 1000,
    fadeOutDuration = 1000,
  } = options;

  // State management
  const [currentSound, setCurrentSound] = useState<AmbientSoundType>(initialSound);
  const [volume, setVolumeState] = useState(initialVolume);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Refs for audio instances
  const howlRef = useRef<Howl | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const bufferSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const gainNodeRef = useRef<GainNode | null>(null);
  const timerModeRef = useRef(false);

  // Initialize audio context for generated sounds
  useEffect(() => {
    if (typeof window !== 'undefined' && window.AudioContext) {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
      gainNodeRef.current = audioContextRef.current.createGain();
      gainNodeRef.current.connect(audioContextRef.current.destination);
      gainNodeRef.current.gain.value = volume;
    }

    return () => {
      if (bufferSourceRef.current) {
        bufferSourceRef.current.stop();
        bufferSourceRef.current.disconnect();
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);

  // Load and create Howl instance for a given sound type
  const createHowlInstance = useCallback((soundType: AmbientSoundType): Howl | null => {
    const soundPath = SOUND_PATHS[soundType];

    if (!soundPath || soundType === 'none') {
      return null;
    }

    try {
      return new Howl({
        src: [soundPath],
        loop,
        volume,
        html5: true,
        preload: true,
        onload: () => {
          setIsLoading(false);
          setError(null);
        },
        onloaderror: (id, error) => {
          setIsLoading(false);
          // Don't set error for file not found - we'll use generated sounds
          console.warn('Howl load error (expected, falling back to generated sounds):', id, error);
          // setError(null); // Clear any previous errors since we have fallback
        },
        onplayerror: (id, error) => {
          setError(`Failed to play sound: ${error}`);
          console.error('Howl play error:', id, error);
        },
        onplay: () => {
          setIsPlaying(true);
          setError(null);
        },
        onpause: () => {
          setIsPlaying(false);
        },
        onstop: () => {
          setIsPlaying(false);
        },
        onend: () => {
          if (!loop) {
            setIsPlaying(false);
          }
        },
      });
    } catch (err) {
      setError(`Failed to create audio instance: ${err}`);
      return null;
    }
  }, [loop, volume]);

  // Play generated sound using Web Audio API (fallback)
  const playGeneratedSound = useCallback((soundType: AmbientSoundType) => {
    if (!audioContextRef.current || !gainNodeRef.current || soundType === 'none') {
      return;
    }

    try {
      // Stop any existing sound
      if (bufferSourceRef.current) {
        bufferSourceRef.current.stop();
        bufferSourceRef.current.disconnect();
      }

      const buffer = generateAmbientSound(soundType);
      if (!buffer) {
        setError(`No sound available for type: ${soundType}`);
        return;
      }

      const source = audioContextRef.current.createBufferSource();
      source.buffer = buffer;
      source.loop = loop;
      source.connect(gainNodeRef.current);

      // Fade in effect
      gainNodeRef.current.gain.setValueAtTime(0, audioContextRef.current.currentTime);
      gainNodeRef.current.gain.linearRampToValueAtTime(
        volume,
        audioContextRef.current.currentTime + fadeInDuration / 1000
      );

      source.start(0);
      bufferSourceRef.current = source;
      setIsPlaying(true);
      setError(null);
    } catch (err) {
      setError(`Failed to play generated sound: ${err}`);
      console.error('Generated sound error:', err);
    }
  }, [volume, loop, fadeInDuration]);

  // Initialize sound on mount
  useEffect(() => {
    if (autoLoad && currentSound !== 'none') {
      // Try to create Howl instance first
      const howl = createHowlInstance(currentSound);
      if (howl) {
        howlRef.current = howl;
      } else {
        // Fallback to generated sound
        setIsLoading(false);
      }
    }
  }, [autoLoad, currentSound, createHowlInstance]);

  // Play function
  const play = useCallback((soundType?: AmbientSoundType) => {
    const soundToPlay = soundType || currentSound;

    if (soundToPlay === 'none') {
      stop();
      return;
    }

    setIsLoading(true);
    setError(null);

    // Try Howl first (for real audio files)
    if (SOUND_PATHS[soundToPlay]) {
      if (howlRef.current?.state() !== 'loaded') {
        const newHowl = createHowlInstance(soundToPlay);
        if (newHowl) {
          howlRef.current = newHowl;
        }
      }

      if (howlRef.current?.state() === 'loaded') {
        howlRef.current.volume(volume);
        howlRef.current.play();
        setCurrentSound(soundToPlay);
        console.log(`🎵 Playing real audio file: ${soundToPlay}`);
        return;
      }
    }

    // Fallback to generated sound
    console.log(`🎵 Using generated ambient sound: ${soundToPlay}`);
    playGeneratedSound(soundToPlay);
    setCurrentSound(soundToPlay);
  }, [currentSound, volume, createHowlInstance, playGeneratedSound]);

  // Pause function
  const pause = useCallback(() => {
    if (howlRef.current) {
      howlRef.current.pause();
    } else if (bufferSourceRef.current) {
      gainNodeRef.current?.gain.linearRampToValueAtTime(
        0,
        audioContextRef.current!.currentTime + 0.1
      );
      setTimeout(() => {
        if (bufferSourceRef.current) {
          try {
            bufferSourceRef.current.stop();
          } catch (e) {
            // Ignore errors from already stopped sources
          }
        }
      }, 100);
    }
    setIsPlaying(false);
  }, []);

  // Stop function with optional fade out
  const stop = useCallback((fadeOut = false) => {
    if (howlRef.current) {
      if (fadeOut && fadeOutDuration > 0) {
        howlRef.current.fade(volume, 0, fadeOutDuration);
        setTimeout(() => {
          howlRef.current?.stop();
        }, fadeOutDuration);
      } else {
        howlRef.current.stop();
      }
    } else if (bufferSourceRef.current && gainNodeRef.current && audioContextRef.current) {
      if (fadeOut) {
        gainNodeRef.current.gain.linearRampToValueAtTime(
          0,
          audioContextRef.current.currentTime + fadeOutDuration / 1000
        );
        setTimeout(() => {
          if (bufferSourceRef.current) {
            try {
              bufferSourceRef.current.stop();
              bufferSourceRef.current.disconnect();
              bufferSourceRef.current = null;
            } catch (e) {
              // Ignore errors from already stopped sources
            }
          }
        }, fadeOutDuration);
      } else {
        try {
          bufferSourceRef.current.stop();
          bufferSourceRef.current.disconnect();
          bufferSourceRef.current = null;
        } catch (e) {
          // Ignore errors from already stopped sources
        }
      }
    }
    setIsPlaying(false);
  }, [volume, fadeOutDuration]);

  // Volume control
  const setVolume = useCallback((newVolume: number) => {
    const clampedVolume = Math.max(0, Math.min(1, newVolume));
    setVolumeState(clampedVolume);

    if (howlRef.current) {
      howlRef.current.volume(clampedVolume);
    } else if (gainNodeRef.current && audioContextRef.current) {
      gainNodeRef.current.gain.linearRampToValueAtTime(
        clampedVolume,
        audioContextRef.current.currentTime + 0.1
      );
    }
  }, []);

  // Change sound function
  const changeSound = useCallback((soundType: AmbientSoundType) => {
    const wasPlaying = isPlaying;

    // Stop current sound
    stop(true);

    // Clean up existing instances
    if (howlRef.current) {
      howlRef.current.unload();
      howlRef.current = null;
    }

    // Create new instance and play if previously playing
    if (soundType !== 'none') {
      setCurrentSound(soundType);
      if (wasPlaying) {
        setTimeout(() => play(soundType), 200); // Small delay for smooth transition
      }
    } else {
      setCurrentSound('none');
    }
  }, [isPlaying, stop, play]);

  // Timer integration functions
  const playWithTimer = useCallback((soundType: AmbientSoundType, timerMode = false) => {
    timerModeRef.current = timerMode;
    play(soundType);
  }, [play]);

  const stopWithTimer = useCallback((fadeOut = true) => {
    if (timerModeRef.current) {
      stop(fadeOut);
      timerModeRef.current = false;
    }
  }, [stop]);

  return {
    // Playback controls
    play,
    pause,
    stop,
    setVolume,
    changeSound,

    // State
    isPlaying,
    isLoading,
    currentSound,
    volume,
    error,

    // Timer integration
    playWithTimer,
    stopWithTimer,
  };
};