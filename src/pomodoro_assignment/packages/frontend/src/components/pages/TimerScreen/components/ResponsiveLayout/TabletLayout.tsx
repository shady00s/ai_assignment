import React from 'react';
import styled from 'styled-components';

interface TabletLayoutProps {
  zenGarden: React.ReactNode;
  currentTask: React.ReactNode;
  sessionControls: React.ReactNode;
  analytics: React.ReactNode;
  wellness: React.ReactNode;
  className?: string;
}

const TabletContainer = styled.div`
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background: linear-gradient(135deg, #F8F9FA 0%, #E8D8C8 100%);

  /* Dark mode styles */
  .dark-mode & {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%) !important;
  }
  font-family: ${({ theme }) => theme.typography.fontFamily.primary};
`;

const HeaderSection = styled.header`
  padding: ${({ theme }) => theme.spacing.tablet.md} ${({ theme }) => theme.spacing.tablet.lg};
  background: rgba(255, 255, 255, 0.9);

  /* Dark mode styles */
  .dark-mode & {
    background: rgba(30, 41, 59, 0.9) !important;
    border-bottom: 1px solid rgba(127, 168, 112, 0.2) !important;
  }
  backdrop-filter: blur(10px);
  border-bottom: 1px solid rgba(127, 168, 112, 0.1);
  position: sticky;
  top: 0;
  z-index: 100;
`;

const MainContent = styled.main`
  flex: 1;
  padding: ${({ theme }) => theme.spacing.tablet.lg};
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.tablet.lg};
`;

const TopSection = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: ${({ theme }) => theme.spacing.tablet.lg};
  align-items: start;
`;

const ZenGardenSection = styled.section`
  grid-column: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.tablet.md};
`;

const SidePanelSection = styled.section`
  grid-column: 2;
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.tablet.md};
`;

const CurrentTaskSection = styled.div`
  background: rgba(255, 255, 255, 0.7);

  /* Dark mode styles */
  .dark-mode & {
    background: rgba(30, 41, 59, 0.7) !important;
    border: 1px solid rgba(127, 168, 112, 0.2) !important;
  }
  border-radius: ${({ theme }) => theme.borderRadius.xl};
  padding: ${({ theme }) => theme.spacing.tablet.md};
  border: 1px solid rgba(127, 168, 112, 0.1);
`;

const WellnessSection = styled.div`
  background: rgba(255, 255, 255, 0.7);

  /* Dark mode styles */
  .dark-mode & {
    background: rgba(30, 41, 59, 0.7) !important;
    border: 1px solid rgba(127, 168, 112, 0.2) !important;
  }
  border-radius: ${({ theme }) => theme.borderRadius.xl};
  padding: ${({ theme }) => theme.spacing.tablet.md};
  border: 1px solid rgba(127, 168, 112, 0.1);
`;

const MiddleSection = styled.section`
  display: flex;
  justify-content: center;
  padding: ${({ theme }) => theme.spacing.tablet.lg} 0;
`;

const ControlsSection = styled.div`
  display: flex;
  justify-content: center;
  gap: ${({ theme }) => theme.spacing.tablet.lg};
`;

const BottomSection = styled.section`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: ${({ theme }) => theme.spacing.tablet.lg};
`;

const AnalyticsLeftSection = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.tablet.md};
`;

const AnalyticsRightSection = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.tablet.md};
`;

export const TabletLayout: React.FC<TabletLayoutProps> = ({
  zenGarden,
  currentTask,
  sessionControls,
  analytics,
  wellness,
  className,
}) => {
  return (
    <TabletContainer className={className}>
      <MainContent>
        <TopSection>
          <ZenGardenSection>
            {zenGarden}
          </ZenGardenSection>

          <SidePanelSection>
            <CurrentTaskSection>
              {currentTask}
            </CurrentTaskSection>

            <WellnessSection>
              {wellness}
            </WellnessSection>
          </SidePanelSection>
        </TopSection>

        <MiddleSection>
          <ControlsSection>
            {sessionControls}
          </ControlsSection>
        </MiddleSection>

        <BottomSection>
          <AnalyticsLeftSection>
            {analytics}
          </AnalyticsLeftSection>

          <AnalyticsRightSection>
            {/* Additional analytics components can go here */}
          </AnalyticsRightSection>
        </BottomSection>
      </MainContent>
    </TabletContainer>
  );
};

export type { TabletLayoutProps };