import React, { useState } from 'react';
import styled from 'styled-components';

interface AmbientSettingsProps {
  soundEnabled: boolean;
  volume: number;
  currentSound: 'forest' | 'ocean' | 'cafe' | 'rain' | 'none';
  focusMode: boolean;
  notificationsEnabled: boolean;
  onSoundToggle?: (enabled: boolean) => void;
  onVolumeChange?: (volume: number) => void;
  onSoundChange?: (sound: string) => void;
  onFocusModeToggle?: (enabled: boolean) => void;
  onNotificationsToggle?: (enabled: boolean) => void;
  className?: string;
}

const SettingsContainer = styled.div<{ $compact: boolean }>`
  background: rgba(255, 255, 255, 0.8);
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  padding: ${({ $compact, theme }) => $compact ? theme.spacing.mobile.sm : theme.spacing.mobile.md};
  border: 1px solid rgba(127, 168, 112, 0.1);
  backdrop-filter: blur(10px);

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ $compact, theme }) => $compact ? theme.spacing.tablet.sm : theme.spacing.tablet.md};
    border-radius: ${({ theme }) => theme.borderRadius.xl};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: ${({ $compact }) => $compact ? '16px' : '24px'};
    border-radius: 20px;
  }
`;

const SettingsTitle = styled.h4`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: #2C3E50;
  margin: 0 0 ${({ theme }) => theme.spacing.mobile.sm} 0;
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.mobile.xs};
`;

const SettingsGrid = styled.div`
  display: grid;
  gap: ${({ theme }) => theme.spacing.mobile.sm};
`;

const SettingItem = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: ${({ theme }) => theme.spacing.mobile.xs} 0;
`;

const SettingLabel = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.mobile.xs};
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
  color: #8B7D7B;
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
`;

const SettingControl = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.mobile.sm};
`;

const ToggleSwitch = styled.button<{ $active: boolean }>`
  position: relative;
  width: 48px;
  height: 24px;
  background: ${({ $active }) => $active ? '#7FA870' : '#D4C4B0'};
  border: none;
  border-radius: 12px;
  cursor: pointer;
  transition: background 0.3s ease;
  padding: 2px;

  &::after {
    content: '';
    position: absolute;
    top: 2px;
    left: ${({ $active }) => $active ? '26px' : '2px'};
    width: 18px;
    height: 18px;
    background: white;
    border-radius: 50%;
    transition: left 0.3s ease;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
  }

  &:hover {
    opacity: 0.8;
  }
`;

const VolumeSlider = styled.input`
  width: 80px;
  height: 4px;
  -webkit-appearance: none;
  appearance: none;
  background: rgba(127, 168, 112, 0.2);
  border-radius: 2px;
  outline: none;

  &::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 16px;
    height: 16px;
    background: #7FA870;
    border-radius: 50%;
    cursor: pointer;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);

    &:hover {
      transform: scale(1.2);
    }
  }

  &::-moz-range-thumb {
    width: 16px;
    height: 16px;
    background: #7FA870;
    border-radius: 50%;
    cursor: pointer;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    border: none;

    &:hover {
      transform: scale(1.2);
    }
  }
`;

const SoundSelector = styled.select`
  padding: 6px 10px;
  border: 1px solid #D4C4B0;
  border-radius: 8px;
  background: white;
  color: #2C3E50;
  font-size: 12px;
  cursor: pointer;
  font-family: ${({ theme }) => theme.typography.fontFamily.primary};

  &:focus {
    outline: none;
    border-color: #7FA870;
  }
`;

const VolumeDisplay = styled.span`
  font-size: 11px;
  color: #8B7D7B;
  min-width: 30px;
  text-align: center;
`;

const PresetButtons = styled.div`
  display: flex;
  gap: 6px;
  margin-top: ${({ theme }) => theme.spacing.mobile.sm};
  flex-wrap: wrap;
`;

const PresetButton = styled.button<{ $active?: boolean }>`
  background: ${({ $active }) => $active ? '#7FA870' : 'rgba(127, 168, 112, 0.1)'};
  color: ${({ $active }) => $active ? 'white' : '#7FA870'};
  border: 1px solid rgba(127, 168, 112, 0.2);
  border-radius: 12px;
  padding: 4px 8px;
  font-size: 10px;
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover {
    background: #7FA870;
    color: white;
  }
`;

const soundOptions = [
  { value: 'none', label: 'Silent', icon: '🔇' },
  { value: 'forest', label: 'Forest', icon: '🌲' },
  { value: 'ocean', label: 'Ocean', icon: '🌊' },
  { value: 'cafe', label: 'Cafe', icon: '☕' },
  { value: 'rain', label: 'Rain', icon: '🌧️' },
];

export const AmbientSettings: React.FC<AmbientSettingsProps> = ({
  soundEnabled,
  volume,
  currentSound,
  focusMode,
  notificationsEnabled,
  onSoundToggle,
  onVolumeChange,
  onSoundChange,
  onFocusModeToggle,
  onNotificationsToggle,
  className,
}) => {
  const [localVolume, setLocalVolume] = useState(volume);

  const handleVolumeChange = (newVolume: number) => {
    setLocalVolume(newVolume);
    onVolumeChange?.(newVolume);
  };

  const presets = [
    { name: 'Focus', volume: 70, sound: 'forest', focusMode: true, notifications: false },
    { name: 'Relax', volume: 50, sound: 'ocean', focusMode: false, notifications: true },
    { name: 'Energize', volume: 80, sound: 'cafe', focusMode: false, notifications: true },
    { name: 'Sleep', volume: 30, sound: 'rain', focusMode: true, notifications: false },
  ];

  const applyPreset = (preset: typeof presets[0]) => {
    handleVolumeChange(preset.volume);
    onSoundChange?.(preset.sound);
    onFocusModeToggle?.(preset.focusMode);
    onNotificationsToggle?.(preset.notifications);
    if (!soundEnabled && preset.sound !== 'none') {
      onSoundToggle?.(true);
    }
  };

  return (
    <SettingsContainer className={className}>
      <SettingsTitle>🎵 Ambient Settings</SettingsTitle>

      <SettingsGrid>
        <SettingItem>
          <SettingLabel>
            🔔 Notifications
          </SettingLabel>
          <SettingControl>
            <ToggleSwitch
              $active={notificationsEnabled}
              onClick={() => onNotificationsToggle?.(!notificationsEnabled)}
            />
          </SettingControl>
        </SettingItem>

        <SettingItem>
          <SettingLabel>
            🔇 Focus Mode
          </SettingLabel>
          <SettingControl>
            <ToggleSwitch
              $active={focusMode}
              onClick={() => onFocusModeToggle?.(!focusMode)}
            />
          </SettingControl>
        </SettingItem>

        <SettingItem>
          <SettingLabel>
            🎵 Ambient Sound
          </SettingLabel>
          <SettingControl>
            <ToggleSwitch
              $active={soundEnabled}
              onClick={() => onSoundToggle?.(!soundEnabled)}
            />
          </SettingControl>
        </SettingItem>

        {soundEnabled && (
          <>
            <SettingItem>
              <SettingLabel>
                🌊 Sound Type
              </SettingLabel>
              <SettingControl>
                <SoundSelector
                  value={currentSound}
                  onChange={(e) => onSoundChange?.(e.target.value)}
                >
                  {soundOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.icon} {option.label}
                    </option>
                  ))}
                </SoundSelector>
              </SettingControl>
            </SettingItem>

            <SettingItem>
              <SettingLabel>
                🔊 Volume
              </SettingLabel>
              <SettingControl>
                <VolumeSlider
                  type="range"
                  min="0"
                  max="100"
                  value={localVolume}
                  onChange={(e) => handleVolumeChange(Number(e.target.value))}
                />
                <VolumeDisplay>{localVolume}%</VolumeDisplay>
              </SettingControl>
            </SettingItem>
          </>
        )}
      </SettingsGrid>

      <PresetButtons>
        {presets.map((preset) => (
          <PresetButton
            key={preset.name}
            onClick={() => applyPreset(preset)}
            $active={
              soundEnabled &&
              localVolume === preset.volume &&
              currentSound === preset.sound &&
              focusMode === preset.focusMode &&
              notificationsEnabled === preset.notifications
            }
          >
            {preset.name}
          </PresetButton>
        ))}
      </PresetButtons>
    </SettingsContainer>
  );
};

export type { AmbientSettingsProps };