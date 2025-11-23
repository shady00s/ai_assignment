import React from 'react';
import styled from 'styled-components';

interface InputProps {
  type?: 'text' | 'email' | 'password' | 'number' | 'tel' | 'url';
  placeholder?: string;
  value?: string;
  defaultValue?: string;
  onChange?: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onFocus?: (event: React.FocusEvent<HTMLInputElement>) => void;
  onBlur?: (event: React.FocusEvent<HTMLInputElement>) => void;
  disabled?: boolean;
  required?: boolean;
  error?: boolean;
  helperText?: string;
  label?: string;
  id?: string;
  name?: string;
  autoComplete?: string;
  autoFocus?: boolean;
  maxLength?: number;
  minLength?: number;
  pattern?: string;
  className?: string;
  fullWidth?: boolean;
  size?: 'small' | 'medium' | 'large';
}

const InputWrapper = styled.div<{ $fullWidth?: boolean }>`
  display: flex;
  flex-direction: column;
  width: ${({ $fullWidth }) => $fullWidth ? '100%' : 'auto'};
`;

const Label = styled.label<{ $size?: 'small' | 'medium' | 'large' }>`
  font-size: ${({ theme, $size }) => {
    switch ($size) {
      case 'small': return theme.typography.fontSize.sm;
      case 'large': return theme.typography.fontSize.lg;
      default: return theme.typography.fontSize.base;
    }
  }};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  color: ${({ theme }) => theme.colors.neutral[500]};
  margin-bottom: ${({ theme }) => theme.spacing.xs};
`;

const StyledInput = styled.input<{
  $error?: boolean;
  $size?: 'small' | 'medium' | 'large';
}>`
  ${({ theme, $size, $error }) => ({
    ...theme.components.Input,
    padding: (() => {
      switch ($size) {
        case 'small': return `${theme.spacing.xs} ${theme.spacing.sm}`;
        case 'large': return `${theme.spacing.md} ${theme.spacing.lg}`;
        default: return `${theme.spacing.sm} ${theme.spacing.md}`;
      }
    })(),
    fontSize: (() => {
      switch ($size) {
        case 'small': return theme.typography.fontSize.sm;
        case 'large': return theme.typography.fontSize.lg;
        default: return theme.typography.fontSize.base;
      }
    })(),
    minHeight: (() => {
      switch ($size) {
        case 'small': return '32px';
        case 'large': return '48px';
        default: return '40px';
      }
    })(),
    border: `1px solid ${$error ? theme.colors.error : theme.colors.neutral[300]}`,
    '&:focus': {
      outline: 'none',
      borderColor: $error ? theme.colors.error : theme.colors.primary.main,
      boxShadow: `0 0 0 2px ${($error ? theme.colors.error : theme.colors.primary.light)}33`,
    },
  })};

  &:disabled {
    background-color: ${({ theme }) => theme.colors.neutral[100]};
    cursor: not-allowed;
    opacity: 0.7;
  }

  &::placeholder {
    color: ${({ theme }) => theme.colors.neutral[300]};
  }

  ${({ theme }) => theme.mediaQueries.mobile} {
    min-height: 44px; /* Touch-friendly minimum size */
  }
`;

const HelperText = styled.span<{ $error?: boolean }>`
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  color: ${({ theme, $error }) => $error ? theme.colors.error : theme.colors.neutral[400]};
  margin-top: ${({ theme }) => theme.spacing.xs};
`;

export const Input: React.FC<InputProps> = ({
  type = 'text',
  placeholder,
  value,
  defaultValue,
  onChange,
  onFocus,
  onBlur,
  disabled = false,
  required = false,
  error = false,
  helperText,
  label,
  id,
  name,
  autoComplete,
  autoFocus = false,
  maxLength,
  minLength,
  pattern,
  className,
  fullWidth = false,
  size = 'medium',
}) => {
  const inputId = id || name || `input-${Math.random().toString(36).substr(2, 9)}`;

  return (
    <InputWrapper $fullWidth={fullWidth} className={className}>
      {label && (
        <Label htmlFor={inputId} $size={size}>
          {label}
          {required && ' *'}
        </Label>
      )}
      <StyledInput
        type={type}
        id={inputId}
        name={name}
        placeholder={placeholder}
        value={value}
        defaultValue={defaultValue}
        onChange={onChange}
        onFocus={onFocus}
        onBlur={onBlur}
        disabled={disabled}
        required={required}
        autoComplete={autoComplete}
        autoFocus={autoFocus}
        maxLength={maxLength}
        minLength={minLength}
        pattern={pattern}
        $error={error}
        $size={size}
      />
      {helperText && (
        <HelperText $error={error}>
          {helperText}
        </HelperText>
      )}
    </InputWrapper>
  );
};

export type { InputProps };