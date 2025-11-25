import React, { useState, useEffect } from 'react';
import styled from 'styled-components';

interface FocusModeProps {
  isActive: boolean;
  onToggle?: (active: boolean) => void;
  blockNotifications?: boolean;
  enableAmbientSounds?: boolean;
  setAmbientSound?: (sound: string) => void;
  className?: string;
}

const FocusModeContainer = styled.div<{ $active: boolean }>`
  background: ${({ $active }) => $active ? 'rgba(127, 168, 112, 0.1)' : 'rgba(255, 255, 255, 0.8)'};
  border: ${({ $active }) => $active ? '2px solid #7FA870' : '1px solid rgba(127, 168, 112, 0.2)'};
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  padding: ${({ theme }) => theme.spacing.mobile.md};
  backdrop-filter: blur(10px);
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;

  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(127, 168, 112, 0.1), transparent);
    animation: ${({ $active }) => $active ? 'focusShimmer 3s infinite' : 'none'};
  }

  @keyframes focusShimmer {
    0% { left: -100%; }
    100% { left: 100%; }
  }
`;

const FocusModeHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: ${({ theme }) => theme.spacing.mobile.sm};
`;

const FocusModeTitle = styled.h4<{ $active: boolean }>`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ $active }) => $active ? '#7FA870' : '#2C3E50'};
  margin: 0;
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.mobile.xs};
`;

const FocusModeToggle = styled.button<{ $active: boolean }>`
  background: ${({ $active }) => $active ? '#7FA870' : 'transparent'};
  color: ${({ $active }) => $active ? 'white' : '#7FA870'};
  border: 2px solid #7FA870;
  border-radius: 20px;
  padding: 6px 14px;
  font-size: 12px;
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover {
    transform: translateY(-1px);
    box-shadow: ${({ $active }) => $active ? '0 4px 12px rgba(127, 168, 112, 0.3)' : '0 2px 8px rgba(127, 168, 112, 0.2)'};
  }
`;

const FocusModeFeatures = styled.div<{ $active: boolean }>`
  display: grid;
  gap: ${({ theme }) => theme.spacing.mobile.sm};
  opacity: ${({ $active }) => $active ? 1 : 0.6};
  transition: opacity 0.3s ease;
`;

const FocusFeature = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.mobile.sm};
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
  color: #8B7D7B;
`;

const FeatureIcon = styled.span<{ $enabled: boolean }>`
  opacity: ${({ $enabled }) => $enabled ? 1 : 0.3};
  filter: grayscale(({ $enabled }) => $enabled ? 0 : 1});
`;

const FeatureText = styled.span<{ $enabled: boolean }>`
  color: ${({ $enabled }) => $enabled ? '#2C3E50' : '#A8968E'};
  font-weight: ${({ $enabled }) => $enabled ? 'medium' : 'normal'};
`;

const FeatureToggle = styled.button<{ $enabled: boolean }>`
  margin-left: auto;
  background: ${({ $enabled }) => $enabled ? '#7FA870' : '#D4C4B0'};
  color: white;
  border: none;
  border-radius: 12px;
  width: 40px;
  height: 20px;
  position: relative;
  cursor: pointer;
  transition: background 0.3s ease;

  &::after {
    content: '';
    position: absolute;
    top: 2px;
    left: ${({ $enabled }) => $enabled ? '22px' : '2px'};
    width: 16px;
    height: 16px;
    background: white;
    border-radius: 50%;
    transition: left 0.3s ease;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
  }
`;

const FocusModeStatus = styled.div<{ $active: boolean }>`
  text-align: center;
  margin-top: ${({ theme }) => theme.spacing.mobile.sm};
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.xs};
  color: ${({ $active }) => $active ? '#7FA870' : '#A8968E'};
  font-style: ${({ $active }) => $active ? 'normal' : 'italic'};
  font-weight: ${({ $active }) => $active ? 'medium' : 'normal'};
`;

const ambientSounds = [
  { id: 'forest', name: 'Forest', icon: '🌲' },
  { id: 'ocean', name: 'Ocean', icon: '🌊' },
  { id: 'rain', name: 'Rain', icon: '🌧️' },
  { id: 'cafe', name: 'Cafe', icon: '☕' },
  { id: 'white_noise', name: 'White Noise', icon: '📻' },
];

export const FocusMode: React.FC<FocusModeProps> = ({
  isActive,
  onToggle,
  blockNotifications = true,
  enableAmbientSounds = false,
  setAmbientSound,
  className,
}) => {
  const [notificationsBlocked, setNotificationsBlocked] = useState(blockNotifications);
  const [ambientSoundEnabled, setAmbientSoundEnabled] = useState(enableAmbientSounds);
  const [selectedSound, setSelectedSound] = useState('forest');

  useEffect(() => {
    setNotificationsBlocked(blockNotifications);
    setAmbientSoundEnabled(enableAmbientSounds);
  }, [blockNotifications, enableAmbientSounds]);

  const handleToggle = () => {
    const newActive = !isActive;
    onToggle?.(newActive);
  };

  const handleNotificationToggle = () => {
    const newBlocked = !notificationsBlocked;
    setNotificationsBlocked(newBlocked);
  };

  const handleAmbientToggle = () => {
    const newEnabled = !ambientSoundEnabled;
    setAmbientSoundEnabled(newEnabled);
    if (newEnabled) {
      setAmbientSound?.(selectedSound);
    }
  };

  const handleSoundChange = (soundId: string) => {
    setSelectedSound(soundId);
    if (ambientSoundEnabled) {
      setAmbientSound?.(soundId);
    }
  };

  const selectedSoundData = ambientSounds.find(sound => sound.id === selectedSound);

  return (
    <FocusModeContainer $active={isActive} className={className}>
      <FocusModeHeader>
        <FocusModeTitle $active={isActive}>
          🔇 Focus Mode
        </FocusModeTitle>
        <FocusModeToggle $active={isActive} onClick={handleToggle}>
          {isActive ? 'Active' : 'Activate'}
        </FocusModeToggle>
      </FocusModeHeader>

      <FocusModeFeatures $active={isActive}>
        <FocusFeature>
          <FeatureIcon $enabled={notificationsBlocked}>
            🔔
          </FeatureIcon>
          <FeatureText $enabled={notificationsBlocked}>
            Block Notifications
          </FeatureText>
          <FeatureToggle
            $enabled={notificationsBlocked}
            onClick={handleNotificationToggle}
          />
        </FocusFeature>

        <FocusFeature>
          <FeatureIcon $enabled={ambientSoundEnabled}>
            🎵
          </FeatureIcon>
          <FeatureText $enabled={ambientSoundEnabled}>
            Ambient Sounds
          </FeatureText>
          <FeatureToggle
            $enabled={ambientSoundEnabled}
            onClick={handleAmbientToggle}
          />
        </FocusFeature>

        {ambientSoundEnabled && (
          <FocusFeature>
            <FeatureIcon $enabled={true}>
              {selectedSoundData?.icon}
            </FeatureIcon>
            <select
              value={selectedSound}
              onChange={(e) => handleSoundChange(e.target.value)}
              style={{
                marginLeft: 'auto',
                padding: '4px 8px',
                border: '1px solid #D4C4B0',
                borderRadius: '6px',
                fontSize: '11px',
                background: 'white',
                cursor: 'pointer',
              }}
            >
              {ambientSounds.map((sound) => (
                <option key={sound.id} value={sound.id}>
                  {sound.icon} {sound.name}
                </option>
              ))}
            </select>
          </FocusFeature>
        )}
      </FocusModeFeatures>

      <FocusModeStatus $active={isActive}>
        {isActive
          ? '🎯 Deep focus enabled - minimize distractions for optimal productivity'
          : '💡 Enable focus mode to block notifications and minimize distractions'
        }
      </FocusModeStatus>
    </FocusModeContainer>
  );
};

export type { FocusModeProps };