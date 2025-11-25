import React from 'react';
import styled from 'styled-components';

interface MobileLayoutProps {
  zenGarden: React.ReactNode;
  currentTask: React.ReactNode;
  sessionControls: React.ReactNode;
  analytics: React.ReactNode;
  wellness: React.ReactNode;
  className?: string;
}

const MobileContainer = styled.div`
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background: linear-gradient(135deg, #F8F9FA 0%, #E8D8C8 100%);
  font-family: ${({ theme }) => theme.typography.fontFamily.primary};
`;

const HeaderSection = styled.header`
  padding: ${({ theme }) => theme.spacing.mobile.sm} ${({ theme }) => theme.spacing.mobile.md};
  background: rgba(255, 255, 255, 0.8);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid rgba(127, 168, 112, 0.1);
  position: sticky;
  top: 0;
  z-index: 100;
`;

const MainContent = styled.main`
  flex: 1;
  padding: ${({ theme }) => theme.spacing.mobile.md};
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.mobile.lg};
`;

const ZenGardenSection = styled.section`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.mobile.md};
`;

const CurrentTaskSection = styled.section`
  background: rgba(255, 255, 255, 0.6);
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  padding: ${({ theme }) => theme.spacing.mobile.md};
  border: 1px solid rgba(127, 168, 112, 0.1);
`;

const ControlsSection = styled.section`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.mobile.sm};
`;

const AnalyticsSection = styled.section`
  display: grid;
  grid-template-columns: 1fr;
  gap: ${({ theme }) => theme.spacing.mobile.md};
`;

const WellnessSection = styled.section`
  background: rgba(255, 255, 255, 0.6);
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  padding: ${({ theme }) => theme.spacing.mobile.md};
  border: 1px solid rgba(127, 168, 112, 0.1);
`;

export const MobileLayout: React.FC<MobileLayoutProps> = ({
  zenGarden,
  currentTask,
  sessionControls,
  analytics,
  wellness,
  className,
}) => {
  return (
    <MobileContainer className={className}>
      <MainContent>
        <ZenGardenSection>
          {zenGarden}
        </ZenGardenSection>

        <CurrentTaskSection>
          {currentTask}
        </CurrentTaskSection>

        <ControlsSection>
          {sessionControls}
        </ControlsSection>

        <AnalyticsSection>
          {analytics}
        </AnalyticsSection>

        <WellnessSection>
          {wellness}
        </WellnessSection>
      </MainContent>
    </MobileContainer>
  );
};

export type { MobileLayoutProps };