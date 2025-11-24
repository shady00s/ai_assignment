import React from 'react';
import styled from 'styled-components';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell
} from 'recharts';
import { formatMinutes, calculatePercentage } from '../../utils/dataFormatters';
import { Card } from '@/components/atoms';

export interface WeeklyData {
  day: string;
  focusTime: number;
  goal: number;
  completed: boolean;
}

interface WeeklyBarChartProps {
  weeklyData: WeeklyData[];
  dailyGoal: number;
}

const ChartContainer = styled(Card)`
  padding: ${({ theme }) => theme.spacing.lg};
  min-height: 300px;
`;

const ChartHeader = styled.div`
  margin-bottom: ${({ theme }) => theme.spacing.lg};

  h3 {
    color: ${({ theme }) => theme.colors.neutral[500]};
    font-size: ${({ theme }) => theme.typography.fontSize.xl};
    font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
    margin: 0 0 ${({ theme }) => theme.spacing.xs} 0;
  }

  p {
    color: ${({ theme }) => theme.colors.neutral[400]};
    font-size: ${({ theme }) => theme.typography.fontSize.sm};
    margin: 0;
  }
`;

const ChartWrapper = styled.div`
  width: 100%;
  height: 250px;

  ${({ theme }) => theme.mediaQueries.tablet} {
    height: 300px;
  }
`;

const CustomTooltip = styled.div`
  background-color: ${({ theme }) => theme.colors.neutral[50]};
  border: 1px solid ${({ theme }) => theme.colors.neutral[200]};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  padding: ${({ theme }) => theme.spacing.sm};
  box-shadow: ${({ theme }) => theme.shadows.md};

  .tooltip-title {
    font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
    color: ${({ theme }) => theme.colors.neutral[500]};
    margin-bottom: 2px;
  }

  .tooltip-value {
    color: ${({ theme }) => theme.colors.primary.main};
    font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  }

  .tooltip-goal {
    color: ${({ theme }) => theme.colors.neutral[400]};
    font-size: ${({ theme }) => theme.typography.fontSize.sm};
  }
`;

const CustomTooltipContent: React.FC<any> = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <CustomTooltip>
        <div className="tooltip-title">{label}</div>
        <div className="tooltip-value">Focus: {formatMinutes(data.focusTime)}</div>
        <div className="tooltip-goal">Goal: {formatMinutes(data.goal)}</div>
        <div className="tooltip-goal">
          Progress: {calculatePercentage(data.focusTime, data.goal)}%
        </div>
      </CustomTooltip>
    );
  }
  return null;
};

const getBarColor = (focusTime: number, goal: number): string => {
  const percentage = calculatePercentage(focusTime, goal);
  if (percentage >= 100) return '#7FA870'; // Success green
  if (percentage >= 75) return '#7A8B7F';  // Primary moss green
  if (percentage >= 50) return '#F4A261';  // Warning amber
  return '#C85A5A';                       // Error red
};

export const WeeklyBarChart: React.FC<WeeklyBarChartProps> = ({
  weeklyData,
  dailyGoal
}) => {
  const transformedData = weeklyData.map(day => ({
    ...day,
    percentage: calculatePercentage(day.focusTime, day.goal)
  }));

  return (
    <ChartContainer>
      <ChartHeader>
        <h3>Weekly Overview</h3>
        <p>Your focus time progress over the past week</p>
      </ChartHeader>

      <ChartWrapper>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={transformedData}
            margin={{
              top: 20,
              right: 30,
              left: 20,
              bottom: 5
            }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="#E5E5E5"
              opacity={0.5}
            />
            <XAxis
              dataKey="day"
              tick={{
                fill: '#8B7D7B',
                fontSize: 12
              }}
              axisLine={{
                stroke: '#D4C5B9'
              }}
            />
            <YAxis
              tick={{
                fill: '#8B7D7B',
                fontSize: 12
              }}
              axisLine={{
                stroke: '#D4C5B9'
              }}
              label={{
                value: 'Minutes',
                angle: -90,
                position: 'insideLeft',
                style: {
                  fill: '#8B7D7B',
                  fontSize: 12
                }
              }}
            />
            <Tooltip content={<CustomTooltipContent />} />
            <Bar
              dataKey="focusTime"
              radius={[8, 8, 0, 0]}
              maxBarSize={60}
            >
              {transformedData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={getBarColor(entry.focusTime, entry.goal)}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartWrapper>
    </ChartContainer>
  );
};