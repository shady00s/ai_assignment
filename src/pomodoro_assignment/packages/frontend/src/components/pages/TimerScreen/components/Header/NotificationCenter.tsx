import React from 'react';
import styled from 'styled-components';

interface Notification {
  id: string;
  title: string;
  message: string;
  type: 'info' | 'success' | 'warning' | 'error' | 'achievement';
  timestamp: string;
  read: boolean;
}

interface NotificationCenterProps {
  notifications?: Notification[];
  onClose?: () => void;
  className?: string;
}

const NotificationCenterContainer = styled.div`
  position: absolute;
  top: 100%;
  right: 0;
  margin-top: ${({ theme }) => theme.spacing.sm};
  width: 320px;
  max-height: 400px;
  background: white;
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
  border: 1px solid rgba(127, 168, 112, 0.1);
  overflow: hidden;
  z-index: 1000;

  ${({ theme }) => theme.mediaQueries.tablet} {
    width: 360px;
    max-height: 480px;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    width: 400px;
    max-height: 520px;
  }
`;

const Header = styled.div`
  padding: ${({ theme }) => theme.spacing.mobile.md};
  border-bottom: 1px solid rgba(127, 168, 112, 0.1);
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: linear-gradient(135deg, #F8F9FA 0%, #E8D8C8 100%);
`;

const Title = styled.h3`
  margin: 0;
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.md};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: #2C3E50;
  font-family: ${({ theme }) => theme.typography.fontFamily.primary};
`;

const CloseButton = styled.button`
  background: none;
  border: none;
  font-size: 18px;
  cursor: pointer;
  color: #8B7D7B;
  padding: ${({ theme }) => theme.spacing.xs};
  border-radius: ${({ theme }) => theme.borderRadius.sm};
  transition: all 0.2s ease;

  &:hover {
    background: rgba(127, 168, 112, 0.1);
    color: #2C3E50;
  }
`;

const NotificationList = styled.div`
  max-height: 320px;
  overflow-y: auto;
  padding: ${({ theme }) => theme.spacing.mobile.sm};
`;

const NotificationItem = styled.div<{ $read: boolean; $type: string }>`
  padding: ${({ theme }) => theme.spacing.mobile.md};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  margin-bottom: ${({ theme }) => theme.spacing.sm};
  border-left: 4px solid ${({ $type, theme }) => {
    switch ($type) {
      case 'success': return '#7FA870';
      case 'warning': return '#F4A261';
      case 'error': return '#C85A5A';
      case 'achievement': return '#E9C46A';
      default: return '#7A8B7F';
    }
  }};
  background: ${({ $read }) => $read ? 'rgba(127, 168, 112, 0.05)' : 'rgba(127, 168, 112, 0.1)'};
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover {
    background: rgba(127, 168, 112, 0.15);
    transform: translateX(2px);
  }
`;

const NotificationTitle = styled.div`
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
  color: #2C3E50;
  margin-bottom: ${({ theme }) => theme.spacing.xs};
`;

const NotificationMessage = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
  color: #8B7D7B;
  line-height: 1.4;
`;

const NotificationTime = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.xs};
  color: #A8968E;
  margin-top: ${({ theme }) => theme.spacing.xs};
`;

const EmptyState = styled.div`
  text-align: center;
  padding: ${({ theme }) => theme.spacing.mobile.xl};
  color: #A8968E;
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
`;

export const NotificationCenter: React.FC<NotificationCenterProps> = ({
  notifications = [],
  onClose,
  className,
}) => {
  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`;
    return `${Math.floor(diffMins / 1440)}d ago`;
  };

  return (
    <NotificationCenterContainer className={className}>
      <Header>
        <Title>Notifications</Title>
        <CloseButton onClick={onClose}>×</CloseButton>
      </Header>

      <NotificationList>
        {notifications.length === 0 ? (
          <EmptyState>
            <div>🔔</div>
            <div>No new notifications</div>
          </EmptyState>
        ) : (
          notifications.map((notification) => (
            <NotificationItem
              key={notification.id}
              $read={notification.read}
              $type={notification.type}
            >
              <NotificationTitle>{notification.title}</NotificationTitle>
              <NotificationMessage>{notification.message}</NotificationMessage>
              <NotificationTime>{formatTime(notification.timestamp)}</NotificationTime>
            </NotificationItem>
          ))
        )}
      </NotificationList>
    </NotificationCenterContainer>
  );
};

export type { NotificationCenterProps };