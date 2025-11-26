import React from 'react';
import styled from 'styled-components';

interface DesktopLayoutProps {
  zenGarden: React.ReactNode;
  currentTask: React.ReactNode;
  sessionControls: React.ReactNode;
  analytics: React.ReactNode;
  wellness: React.ReactNode;
  className?: string;
}

const DesktopContainer = styled.div`
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background: linear-gradient(135deg, #F8F9FA 0%, #E8D8C8 100%);
  font-family: ${({ theme }) => theme.typography.fontFamily.primary};

  /* Dark mode styles */
  .dark-mode & {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%) !important;
  }
`;

const HeaderSection = styled.header`
  padding: ${({ theme }) => theme.spacing.lg} ${({ theme }) => theme.spacing.xl};
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(15px);
  border-bottom: 1px solid rgba(127, 168, 112, 0.1);
  position: sticky;
  top: 0;
  z-index: 100;

  /* Dark mode styles */
  .dark-mode & {
    background: rgba(30, 41, 59, 0.95) !important;
    border-bottom: 1px solid rgba(127, 168, 112, 0.2) !important;
  }
`;

const MainContent = styled.main`
  flex: 1;
  padding: ${({ theme }) => theme.spacing.xl};
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.xl};
  max-width: 1400px;
  margin: 0 auto;
  width: 100%;
`;

const TopSection = styled.div`
  display: grid;
  grid-template-columns: 1fr 2fr 1fr;
  gap: ${({ theme }) => theme.spacing.xl};
  align-items: start;
`;

const LeftPanelSection = styled.section`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.lg};
`;

const CenterSection = styled.section`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.xl};
`;

const RightPanelSection = styled.section`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.lg};
`;

const ZenGardenSection = styled.div`
  background: rgba(255, 255, 255, 0.8);
  border-radius: 32px;
  padding: 40px;
  border: 1px solid rgba(127, 168, 112, 0.1);
  backdrop-filter: blur(10px);
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.08);

  /* Dark mode styles */
  .dark-mode & {
    background: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid rgba(127, 168, 112, 0.2) !important;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4) !important;
  }
`;

const CurrentTaskSection = styled.div`
  background: rgba(255, 255, 255, 0.8);
  border-radius: 24px;
  padding: ${({ theme }) => theme.spacing.lg};
  border: 1px solid rgba(127, 168, 112, 0.1);
  backdrop-filter: blur(10px);

  /* Dark mode styles */
  .dark-mode & {
    background: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid rgba(127, 168, 112, 0.2) !important;
  }
`;

const WellnessSection = styled.div`
  background: rgba(255, 255, 255, 0.8);
  border-radius: 24px;
  padding: ${({ theme }) => theme.spacing.lg};
  border: 1px solid rgba(127, 168, 112, 0.1);
  backdrop-filter: blur(10px);

  /* Dark mode styles */
  .dark-mode & {
    background: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid rgba(127, 168, 112, 0.2) !important;
  }
`;

const ControlsSection = styled.div`
  display: flex;
  justify-content: center;
  gap: ${({ theme }) => theme.spacing.xl};
  padding: ${({ theme }) => theme.spacing.lg} 0;
`;

const BottomSection = styled.section`
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: ${({ theme }) => theme.spacing.xl};
`;

const AnalyticsLeftSection = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.lg};
`;

const AnalyticsCenterSection = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.lg};
`;

const AnalyticsRightSection = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.lg};
`;

// Responsive adjustments for very large screens
const DesktopContainerResponsive = styled(DesktopContainer)`
  @media (min-width: 1600px) {
    max-width: 1600px;
    margin: 0 auto;
  }
`;

export const DesktopLayout: React.FC<DesktopLayoutProps> = ({
  header,
  zenGarden,
  currentTask,
  sessionControls,
  analytics,
  wellness,
  className,
}) => {
  return (
    <DesktopContainerResponsive className={className}>


      <MainContent>
        <TopSection>
          <LeftPanelSection>
            <CurrentTaskSection>
              {currentTask}
            </CurrentTaskSection>
             <AnalyticsLeftSection>
            {/* Left analytics components */}
            {analytics}
          </AnalyticsLeftSection>
          </LeftPanelSection>

          <CenterSection>
            <ZenGardenSection>
              {zenGarden}
            </ZenGardenSection>

            <ControlsSection>
              {sessionControls}
            </ControlsSection>
          </CenterSection>

          <RightPanelSection>
            <WellnessSection>
              {wellness}
            </WellnessSection>
          </RightPanelSection>
        </TopSection>

        <BottomSection>
         

          <AnalyticsCenterSection>
            {/* Center analytics components - Team stats, challenges */}
          </AnalyticsCenterSection>

          <AnalyticsRightSection>
            {/* Right analytics components - Achievements, trends */}
          </AnalyticsRightSection>
        </BottomSection>
      </MainContent>
    </DesktopContainerResponsive>
  );
};

export type { DesktopLayoutProps };