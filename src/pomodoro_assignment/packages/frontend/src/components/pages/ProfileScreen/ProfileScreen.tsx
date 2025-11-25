import React, { useState, useMemo } from 'react';
import styled from 'styled-components';
import { useGetProfileQuery, useUpdateProfileMutation, usePatchProfileMutation, useGetTeamsQuery } from '../../../store/api/apiSlice';
import { Card } from '../../atoms/Card';
import { Button } from '../../atoms/Button';
import { Input } from '../../atoms/Input';
import { LoadingSpinner } from '../../atoms/LoadingSpinner';
import { ErrorMessage } from '../../atoms/ErrorMessage';

const ProfileContainer = styled.div`
  padding: ${({ theme }) => theme.spacing.mobile.sm};
  max-width: 100%;
  margin: 0 auto;
  min-height: 100vh;

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ theme }) => theme.spacing.tablet.md};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: ${({ theme }) => theme.spacing.desktop.lg};
    max-width: 1200px;
  }
`;

const ProfileHeader = styled.header`
  margin-bottom: ${({ theme }) => theme.spacing.xl};
  text-align: center;

  h1 {
    color: ${({ theme }) => theme.colors.neutral[500]};
    font-size: ${({ theme }) => theme.typography.fontSize['2xl']};
    font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
    margin-bottom: ${({ theme }) => theme.spacing.sm};

    ${({ theme }) => theme.mediaQueries.tablet} {
      font-size: ${({ theme }) => theme.typography.fontSize.tablet['3xl']};
    }

    ${({ theme }) => theme.mediaQueries.desktop} {
      font-size: ${({ theme }) => theme.typography.fontSize.desktop['3xl']};
    }
  }
`;

const ProfileGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr;
  gap: ${({ theme }) => theme.spacing.lg};

  ${({ theme }) => theme.mediaQueries.desktop} {
    grid-template-columns: repeat(2, 1fr);
  }
`;

const FullWidthSection = styled.div`
  grid-column: 1 / -1;
`;

const SectionTitle = styled.h3`
  color: ${({ theme }) => theme.colors.neutral[500]};
  font-size: ${({ theme }) => theme.typography.fontSize.lg};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  margin-bottom: ${({ theme }) => theme.spacing.md};
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.sm};
`;

const UserInfoSection = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.md};
  margin-bottom: ${({ theme }) => theme.spacing.md};

  ${({ theme }) => theme.mediaQueries.tablet} {
    gap: ${({ theme }) => theme.spacing.lg};
  }
`;

const UserAvatar = styled.div`
  width: 80px;
  height: 80px;
  border-radius: 50%;
  background: linear-gradient(135deg, ${({ theme }) => theme.colors.primary.main}, ${({ theme }) => theme.colors.primary.dark});
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 32px;
  color: white;
  font-weight: bold;

  ${({ theme }) => theme.mediaQueries.tablet} {
    width: 100px;
    height: 100px;
    font-size: 40px;
  }
`;

const UserDetails = styled.div`
  flex: 1;
`;

const UserName = styled.h4`
  color: ${({ theme }) => theme.colors.neutral[600]};
  font-size: ${({ theme }) => theme.typography.fontSize.xl};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  margin-bottom: ${({ theme }) => theme.spacing.xs};
`;

const UserEmail = styled.p`
  color: ${({ theme }) => theme.colors.neutral[400]};
  font-size: ${({ theme }) => theme.typography.fontSize.base};
  margin-bottom: ${({ theme }) => theme.spacing.sm};
`;

const UserStats = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: ${({ theme }) => theme.spacing.md};
  margin-top: ${({ theme }) => theme.spacing.sm};
`;

const StatItem = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.xs};
  color: ${({ theme }) => theme.colors.neutral[500]};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
`;

const PreferencesGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: ${({ theme }) => theme.spacing.md};
`;

const ToggleGroup = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.sm};
  margin-bottom: ${({ theme }) => theme.spacing.md};
  
`;

const Toggle = styled.input.attrs({ type: 'checkbox' })`
  width: 20px;
  height: 20px;
  accent-color: ${({ theme }) => theme.colors.primary.main};
  background-color: ${({ theme }) => theme.colors.neutral[100]};
  border: 2px solid ${({ theme }) => theme.colors.neutral[300]};
  border-radius: 4px;
  cursor: pointer;
  color: white;
  &:checked {
    background-color: ${({ theme }) => theme.colors.primary.main};
    border-color: ${({ theme }) => theme.colors.primary.main};
  }

  &:checked::before {
    content: '✓';
    color: white;
    display: block;
    text-align: center;
    line-height: 16px;
    font-size: 14px;
    font-weight: bold;
  }

  &:focus {
    outline: 2px solid ${({ theme }) => theme.colors.primary.light};
    outline-offset: 2px;
  }
`;

const ToggleLabel = styled.label`
  color: ${({ theme }) => theme.colors.neutral[500]};
  font-size: ${({ theme }) => theme.typography.fontSize.base};
  cursor: pointer;
  user-select: none;
`;

const ActionButtons = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing.md};
  margin-top: ${({ theme }) => theme.spacing.lg};
  flex-wrap: wrap;
`;

const ButtonGroup = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing.sm};
  margin-top: ${({ theme }) => theme.spacing.md};
  flex-wrap: wrap;
`;

const SuccessMessage = styled.div`
  background-color: ${({ theme }) => theme.colors.success}20;
  color: ${({ theme }) => theme.colors.success};
  padding: ${({ theme }) => theme.spacing.sm} ${({ theme }) => theme.spacing.md};
  border-radius: 8px;
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  margin-top: ${({ theme }) => theme.spacing.md};
`;

const FormContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.md};
`;

const FormRow = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.sm};
`;

const FormLabel = styled.label`
  color: ${({ theme }) => theme.colors.neutral[500]};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
`;

const FormInput = styled.input`
  padding: ${({ theme }) => theme.spacing.sm} ${({ theme }) => theme.spacing.md};
  border: 1px solid ${({ theme }) => theme.colors.neutral[300]};
  border-radius: 6px;
  font-size: ${({ theme }) => theme.typography.fontSize.base};
  color: #000000;
  background-color: #ffffff;

  &:focus {
    outline: none;
    border-color: ${({ theme }) => theme.colors.primary.main};
    box-shadow: 0 0 0 2px ${({ theme }) => theme.colors.primary.main}30;
  }

  &.error {
    border-color: ${({ theme }) => theme.colors.error};
  }

  &::placeholder {
    color: #666666;
  }
`;

const FormSelect = styled.select`
  padding: ${({ theme }) => theme.spacing.sm} ${({ theme }) => theme.spacing.md};
  border: 1px solid ${({ theme }) => theme.colors.neutral[300]};
  border-radius: 6px;
  font-size: ${({ theme }) => theme.typography.fontSize.base};
  color: #000000;
  background-color: #ffffff;

  &:focus {
    outline: none;
    border-color: ${({ theme }) => theme.colors.primary.main};
    box-shadow: 0 0 0 2px ${({ theme }) => theme.colors.primary.main}30;
  }

  option {
    color: #000000;
    background-color: #ffffff;
  }
`;

const FormHelper = styled.p`
  color: ${({ theme }) => theme.colors.neutral[400]};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  margin-top: ${({ theme }) => theme.spacing.xs};
`;

const FormErrorMessage = styled.p`
  color: ${({ theme }) => theme.colors.error};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  margin-top: ${({ theme }) => theme.spacing.xs};
`;

export const ProfileScreen: React.FC = () => {
  const [showPreferencesSuccess, setShowPreferencesSuccess] = useState(false);
  const [showNotificationsSuccess, setShowNotificationsSuccess] = useState(false);
  const [showWellnessSuccess, setShowWellnessSuccess] = useState(false);
  const [showProfileSuccess, setShowProfileSuccess] = useState(false);
  const [isUpdatingPreferences, setIsUpdatingPreferences] = useState(false);
  const [isUpdatingNotifications, setIsUpdatingNotifications] = useState(false);
  const [isUpdatingWellness, setIsUpdatingWellness] = useState(false);
  const [isEditingProfile, setIsEditingProfile] = useState(false);
  const [isUpdatingProfile, setIsUpdatingProfile] = useState(false);

  const {
    data: userProfile,
    isLoading: profileLoading,
    error: profileError,
  } = useGetProfileQuery();

  const [updateProfile] = useUpdateProfileMutation();
  const [patchProfile] = usePatchProfileMutation();
  const { data: teams } = useGetTeamsQuery();

  // Initialize local state using useMemo to avoid setState in effect
  const localPreferences = useMemo(() => {
    if (userProfile?.preferences) {
      const prefs = userProfile.preferences;
      return {
        workDuration: prefs.workDuration ?? 25,
        shortBreakDuration: prefs.shortBreakDuration ?? 6,
        longBreakDuration: prefs.longBreakDuration ?? 15,
        longBreakInterval: prefs.longBreakInterval ?? 4,
        autoStartBreaks: prefs.autoStartBreaks ?? false,
        autoStartWork: prefs.autoStartWork ?? false,
        soundEnabled: prefs.soundEnabled ?? true,
        volume: prefs.volume ?? 70,
        ambientSound: (prefs.ambientSound as 'forest' | 'ocean' | 'cafe' | 'rain' | 'none') ?? 'forest',
        darkMode: prefs.darkMode ?? false,
      };
    }
    return {
      workDuration: 25,
      shortBreakDuration: 6,
      longBreakDuration: 15,
      longBreakInterval: 4,
      autoStartBreaks: false,
      autoStartWork: false,
      soundEnabled: true,
      volume: 70,
      ambientSound: 'forest' as 'forest' | 'ocean' | 'cafe' | 'rain' | 'none',
      darkMode: false,
    };
  }, [userProfile?.preferences]);

  const localNotifications = useMemo(() => {
    if (userProfile?.preferences?.notifications) {
      const notifs = userProfile.preferences.notifications;
      return {
        achievements: notifs.achievements ?? true,
        teamUpdates: notifs.teamUpdates ?? true,
        weeklyReports: notifs.weeklyReports ?? true,
        deadlineReminders: notifs.deadlineReminders ?? true,
        wellnessReminders: notifs.wellnessReminders ?? true,
      };
    }
    return {
      achievements: true,
      teamUpdates: true,
      weeklyReports: true,
      deadlineReminders: true,
      wellnessReminders: true,
    };
  }, [userProfile?.preferences?.notifications]);

  const localWellness = useMemo(() => {
    if (userProfile?.preferences?.wellness) {
      const wellness = userProfile.preferences.wellness;
      return {
        mindfulnessReminders: wellness.mindfulnessReminders ?? false,
        hydrationReminders: wellness.hydrationReminders ?? false,
        movementBreaks: wellness.movementBreaks ?? false,
        eyeRest: wellness.eyeRest ?? true,
        endOfDay: wellness.endOfDay ?? false,
      };
    }
    return {
      mindfulnessReminders: false,
      hydrationReminders: false,
      movementBreaks: false,
      eyeRest: true,
      endOfDay: false,
    };
  }, [userProfile?.preferences?.wellness]);

  const [preferencesState, setPreferencesState] = useState(localPreferences);
  const [notificationsState, setNotificationsState] = useState(localNotifications);
  const [wellnessState, setWellnessState] = useState(localWellness);

  // Form state for editable user info
  const [tempFormData, setTempFormData] = useState({
    firstName: userProfile?.firstName || '',
    lastName: userProfile?.lastName || '',
    email: userProfile?.email || '',
    username: userProfile?.email?.split('@')[0] || '',
    avatar: userProfile?.avatar || '',
    teamId: userProfile?.teamId || '',
  });

  const [errors, setErrors] = useState({
    firstName: '',
    lastName: '',
    email: '',
    username: '',
  });

  // Update state when memoized values change
  React.useEffect(() => {
    setPreferencesState(localPreferences);
  }, [localPreferences]);

  React.useEffect(() => {
    setNotificationsState(localNotifications);
  }, [localNotifications]);

  React.useEffect(() => {
    setWellnessState(localWellness);
  }, [localWellness]);

  // Update form data when userProfile changes
  React.useEffect(() => {
    if (userProfile) {
      setTempFormData({
        firstName: userProfile.firstName || '',
        lastName: userProfile.lastName || '',
        email: userProfile.email || '',
        username: userProfile.email?.split('@')[0] || '',
        avatar: userProfile.avatar || '',
        teamId: userProfile.teamId || '',
      });
    }
  }, [userProfile]);

  // Validation functions
  const validateEmail = (email: string) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  };

  const validateUsername = (username: string) => {
    return username.length >= 3 && /^[a-zA-Z0-9_]+$/.test(username);
  };

  const validateForm = () => {
    const newErrors = {
      firstName: '',
      lastName: '',
      email: '',
      username: '',
    };

    let isValid = true;

    if (!tempFormData.firstName.trim()) {
      newErrors.firstName = 'First name is required';
      isValid = false;
    }

    if (!tempFormData.lastName.trim()) {
      newErrors.lastName = 'Last name is required';
      isValid = false;
    }

    if (!tempFormData.email.trim()) {
      newErrors.email = 'Email is required';
      isValid = false;
    } else if (!validateEmail(tempFormData.email)) {
      newErrors.email = 'Please enter a valid email address';
      isValid = false;
    }

    if (!tempFormData.username.trim()) {
      newErrors.username = 'Username is required';
      isValid = false;
    } else if (!validateUsername(tempFormData.username)) {
      newErrors.username = 'Username must be at least 3 characters and contain only letters, numbers, and underscores';
      isValid = false;
    }

    setErrors(newErrors);
    return isValid;
  };

  // Handler functions
  const handleEditProfile = () => {
    setIsEditingProfile(true);
  };

  const handleCancelEdit = () => {
    setIsEditingProfile(false);
    // Reset form data to original values
    if (userProfile) {
      setTempFormData({
        firstName: userProfile.firstName || '',
        lastName: userProfile.lastName || '',
        email: userProfile.email || '',
        username: userProfile.email?.split('@')[0] || '',
        avatar: userProfile.avatar || '',
        teamId: userProfile.teamId || '',
      });
    }
    setErrors({
      firstName: '',
      lastName: '',
      email: '',
      username: '',
    });
  };

  const handleSaveProfile = async () => {
    if (!validateForm()) {
      return;
    }

    setIsUpdatingProfile(true);
    try {
      await patchProfile({
        firstName: tempFormData.firstName,
        lastName: tempFormData.lastName,
        email: tempFormData.email,
        avatar: tempFormData.avatar || undefined,
        teamId: tempFormData.teamId || undefined,
      }).unwrap();

      setShowProfileSuccess(true);
      setIsEditingProfile(false);
      setTimeout(() => setShowProfileSuccess(false), 3000);
    } catch (error) {
      console.error('Failed to update profile:', error);
      alert('Failed to update profile. Please try again.');
    } finally {
      setIsUpdatingProfile(false);
    }
  };


  const handleUpdatePreferences = async () => {
    setIsUpdatingPreferences(true);
    try {
      await updateProfile({
        preferences: {
          ...preferencesState,
          notifications: notificationsState,
          wellness: wellnessState,
        }
      }).unwrap();

      setShowPreferencesSuccess(true);
      setTimeout(() => setShowPreferencesSuccess(false), 3000);
    } catch (error) {
      console.error('Failed to update preferences:', error);
      alert('Failed to update preferences. Please try again.');
    } finally {
      setIsUpdatingPreferences(false);
    }
  };

  const handleUpdateNotifications = async () => {
    setIsUpdatingNotifications(true);
    try {
      await updateProfile({
        preferences: {
          ...preferencesState,
          notifications: notificationsState,
          wellness: wellnessState,
        }
      }).unwrap();

      setShowNotificationsSuccess(true);
      setTimeout(() => setShowNotificationsSuccess(false), 3000);
    } catch (error) {
      console.error('Failed to update notifications:', error);
      alert('Failed to update notifications. Please try again.');
    } finally {
      setIsUpdatingNotifications(false);
    }
  };

  const handleUpdateWellness = async () => {
    setIsUpdatingWellness(true);
    try {
      await updateProfile({
        preferences: {
          ...preferencesState,
          notifications: notificationsState,
          wellness: wellnessState,
        }
      }).unwrap();

      setShowWellnessSuccess(true);
      setTimeout(() => setShowWellnessSuccess(false), 3000);
    } catch (error) {
      console.error('Failed to update wellness settings:', error);
      alert('Failed to update wellness settings. Please try again.');
    } finally {
      setIsUpdatingWellness(false);
    }
  };

  
  const handleExportData = async () => {
    try {
      // TODO: Implement actual data export functionality
      console.log('Export data clicked');
      alert('Data export feature coming soon! You will be able to download all your data.');
    } catch (error) {
      console.error('Failed to export data:', error);
      alert('Failed to export data. Please try again.');
    }
  };

  const handleDeleteAccount = async () => {
    const confirmed = window.confirm(
      'Are you sure you want to delete your account? This action cannot be undone and will permanently delete all your data.'
    );

    if (confirmed) {
      const doubleConfirmed = window.confirm(
        'This is your final warning. Are you absolutely sure you want to delete your account?'
      );

      if (doubleConfirmed) {
        try {
          // TODO: Implement actual account deletion API call
          console.log('Delete account confirmed');
          alert('Account deletion feature coming soon! Please contact support to delete your account.');
        } catch (error) {
          console.error('Failed to delete account:', error);
          alert('Failed to delete account. Please contact support.');
        }
      }
    }
  };

  const handleSignOut = async () => {
    try {
      // TODO: Implement actual sign out functionality
      console.log('Sign out clicked');

      // For now, just clear local storage and reload
      localStorage.clear();
      window.location.href = '/auth';
    } catch (error) {
      console.error('Failed to sign out:', error);
      alert('Failed to sign out. Please try again.');
    }
  };

  if (profileLoading) {
    return (
      <ProfileContainer>
        <LoadingSpinner size="large" centered />
      </ProfileContainer>
    );
  }

  if (profileError) {
    return (
      <ProfileContainer>
        <ErrorMessage
          message="Failed to load profile data. Please try again later."
          variant="card"
        />
      </ProfileContainer>
    );
  }

  const userName = `${userProfile?.firstName || ''} ${userProfile?.lastName || ''}`.trim() || 'User';
  const userEmail = userProfile?.email || 'user@example.com';
  const userLevel = userProfile?.level || 1;
  const userXP = userProfile?.xp || 0;
  const userStreak = userProfile?.streak || 0;
  const userTotalFocusTime = userProfile?.totalFocusTime || 0;
  const userTasksCompleted = userProfile?.tasksCompleted || 0;
  const userQualityScore = userProfile?.qualityScore || 0;
  const userWellnessScore = userProfile?.wellnessScore || 0;
  const userAvatar = userProfile?.avatar;
  const userTeamId = userProfile?.teamId;
  const createdAt = userProfile?.createdAt;

  // Get team name from teams data
  const getTeamName = (teamId: string | undefined) => {
    if (!teamId || !teams) return 'No Team';
    const team = teams.find(t => t.id === teamId);
    return team ? team.name : 'Unknown Team';
  };

  const userTeamName = getTeamName(userTeamId);

  // Format focus time from minutes to hours
  const focusTimeHours = Math.floor(userTotalFocusTime / 60);
  const focusTimeMinutes = userTotalFocusTime % 60;
  const formattedFocusTime = focusTimeHours > 0
    ? `${focusTimeHours}h ${focusTimeMinutes}m`
    : `${focusTimeMinutes}m`;

  // Format date
  const joinDate = createdAt ? new Date(createdAt).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric'
  }) : 'Unknown';

  return (
    <ProfileContainer>
      <ProfileHeader>
        <h1>My Profile & Settings</h1>
      </ProfileHeader>

      <ProfileGrid>
        {/* User Info Section */}
        <Card>
          <SectionTitle>👤 User Info</SectionTitle>

          {isEditingProfile ? (
            <FormContainer>
              <FormRow>
                <FormLabel htmlFor="firstName">First Name</FormLabel>
                <FormInput
                  id="firstName"
                  type="text"
                  value={tempFormData.firstName}
                  onChange={(e) => setTempFormData(prev => ({ ...prev, firstName: e.target.value }))}
                  className={errors.firstName ? 'error' : ''}
                />
                {errors.firstName && (
                  <FormErrorMessage>{errors.firstName}</FormErrorMessage>
                )}
              </FormRow>

              <FormRow>
                <FormLabel htmlFor="lastName">Last Name</FormLabel>
                <FormInput
                  id="lastName"
                  type="text"
                  value={tempFormData.lastName}
                  onChange={(e) => setTempFormData(prev => ({ ...prev, lastName: e.target.value }))}
                  className={errors.lastName ? 'error' : ''}
                />
                {errors.lastName && (
                  <FormErrorMessage>{errors.lastName}</FormErrorMessage>
                )}
              </FormRow>

              <FormRow>
                <FormLabel htmlFor="email">Email Address</FormLabel>
                <FormInput
                  id="email"
                  type="email"
                  value={tempFormData.email}
                  onChange={(e) => setTempFormData(prev => ({ ...prev, email: e.target.value }))}
                  className={errors.email ? 'error' : ''}
                />
                {errors.email && (
                  <FormErrorMessage>{errors.email}</FormErrorMessage>
                )}
              </FormRow>

              <FormRow>
                <FormLabel htmlFor="username">Username</FormLabel>
                <FormInput
                  id="username"
                  type="text"
                  value={tempFormData.username}
                  onChange={(e) => setTempFormData(prev => ({ ...prev, username: e.target.value }))}
                  className={errors.username ? 'error' : ''}
                />
                {errors.username && (
                  <FormErrorMessage>{errors.username}</FormErrorMessage>
                )}
              </FormRow>

              <FormRow>
                <FormLabel htmlFor="avatar">Avatar URL</FormLabel>
                <FormInput
                  id="avatar"
                  type="url"
                  value={tempFormData.avatar || ''}
                  onChange={(e) => setTempFormData(prev => ({ ...prev, avatar: e.target.value }))}
                  placeholder="https://example.com/avatar.jpg"
                />
                <FormHelper>
                  Enter a URL for your profile picture
                  {tempFormData.avatar && (
                    <div style={{ marginTop: '8px' }}>
                      <img
                        src={tempFormData.avatar}
                        alt="Avatar preview"
                        style={{
                          width: '50px',
                          height: '50px',
                          borderRadius: '50%',
                          objectFit: 'cover',
                          border: '2px solid #e5e7eb'
                        }}
                        onError={(e) => {
                          e.currentTarget.style.display = 'none';
                        }}
                      />
                    </div>
                  )}
                </FormHelper>
              </FormRow>

              <FormRow>
                <FormLabel htmlFor="teamId">Team Assignment</FormLabel>
                <FormSelect
                  id="teamId"
                  value={tempFormData.teamId || ''}
                  onChange={(e) => setTempFormData(prev => ({ ...prev, teamId: e.target.value || undefined }))}
                >
                  <option value="">No Team</option>
                  {teams?.map(team => (
                    <option key={team.id} value={team.id}>
                      {team.name}
                    </option>
                  ))}
                </FormSelect>
                <FormHelper>Select your team assignment</FormHelper>
              </FormRow>

              {showProfileSuccess && (
                <SuccessMessage>
                  Profile updated successfully!
                </SuccessMessage>
              )}

              <ActionButtons>
                <Button
                  onClick={handleSaveProfile}
                  disabled={isUpdatingProfile}
                >
                  {isUpdatingProfile ? 'Saving...' : 'Save Changes'}
                </Button>
                <Button
                  variant="secondary"
                  onClick={handleCancelEdit}
                  disabled={isUpdatingProfile}
                >
                  Cancel Changes
                </Button>
              </ActionButtons>
            </FormContainer>
          ) : (
            <>
              <UserInfoSection>
                <UserAvatar>
                  {userAvatar ? (
                    <img
                      src={userAvatar}
                      alt={`${userName}'s avatar`}
                      style={{
                        width: '100%',
                        height: '100%',
                        borderRadius: '50%',
                        objectFit: 'cover'
                      }}
                      onError={(e) => {
                        // Fallback to initials if image fails to load
                        e.currentTarget.style.display = 'none';
                        e.currentTarget.parentElement!.textContent = userName.charAt(0).toUpperCase();
                      }}
                    />
                  ) : (
                    userName.charAt(0).toUpperCase()
                  )}
                </UserAvatar>
                <UserDetails>
                  <UserName>{userName}</UserName>
                  <UserEmail>📧 {userEmail}</UserEmail>
                  <UserStats>
                    <StatItem>
                      <span>🏢</span>
                      <span>Level {userLevel}</span>
                    </StatItem>
                    <StatItem>
                      <span>⭐</span>
                      <span>{userXP.toLocaleString()} XP</span>
                    </StatItem>
                    <StatItem>
                      <span>👥</span>
                      <span>{userTeamName}</span>
                    </StatItem>
                    <StatItem>
                      <span>🔥</span>
                      <span>{userStreak} day streak</span>
                    </StatItem>
                  </UserStats>
                  <UserStats>
                    <StatItem>
                      <span>📝</span>
                      <span>{userTasksCompleted} tasks</span>
                    </StatItem>
                    <StatItem>
                      <span>⏱️</span>
                      <span>{formattedFocusTime} focus time</span>
                    </StatItem>
                    <StatItem>
                      <span>💯</span>
                      <span>Quality: {userQualityScore.toFixed(1)}/5.0</span>
                    </StatItem>
                    <StatItem>
                      <span>🌿</span>
                      <span>Wellness: {userWellnessScore.toFixed(1)}/10</span>
                    </StatItem>
                  </UserStats>
                  <UserStats>
                    <StatItem>
                      <span>📅</span>
                      <span>Joined {joinDate}</span>
                    </StatItem>
                  </UserStats>
                </UserDetails>
              </UserInfoSection>
              <Button variant="secondary" onClick={handleEditProfile}>Edit Profile</Button>
            </>
          )}
        </Card>

        {/* Preferences Section */}
        <Card>
          <SectionTitle>⚙️ Preferences</SectionTitle>
          <PreferencesGrid>
            <Input
              type="number"
              label="Work Duration"
              value={preferencesState.workDuration.toString()}
              onChange={(e) => setPreferencesState(prev => ({
                ...prev,
                workDuration: parseInt(e.target.value) || 25
              }))}
                            helperText="minutes"
            />
            <Input
              type="number"
              label="Short Break"
              value={preferencesState.shortBreakDuration.toString()}
              onChange={(e) => setPreferencesState(prev => ({
                ...prev,
                shortBreakDuration: parseInt(e.target.value) || 6
              }))}
                            helperText="minutes"
            />
            <Input
              type="number"
              label="Long Break"
              value={preferencesState.longBreakDuration.toString()}
              onChange={(e) => setPreferencesState(prev => ({
                ...prev,
                longBreakDuration: parseInt(e.target.value) || 15
              }))}
                            helperText="minutes"
            />
            <Input
              type="number"
              label="Long Break Interval"
              value={preferencesState.longBreakInterval.toString()}
              onChange={(e) => setPreferencesState(prev => ({
                ...prev,
                longBreakInterval: parseInt(e.target.value) || 4
              }))}
                            helperText="sessions"
            />
          </PreferencesGrid>

          <ToggleGroup>
            <Toggle
              id="autoStartBreaks"
              checked={preferencesState.autoStartBreaks}
              onChange={(e) => setPreferencesState(prev => ({
                ...prev,
                autoStartBreaks: e.target.checked
              }))}
            />
            <ToggleLabel htmlFor="autoStartBreaks">
              Auto-start breaks
            </ToggleLabel>
          </ToggleGroup>

          <ToggleGroup>
            <Toggle
              id="autoStartWork"
              checked={preferencesState.autoStartWork}
              onChange={(e) => setPreferencesState(prev => ({
                ...prev,
                autoStartWork: e.target.checked
              }))}
            />
            <ToggleLabel htmlFor="autoStartWork">
              Auto-start work sessions
            </ToggleLabel>
          </ToggleGroup>

          {showPreferencesSuccess && (
            <SuccessMessage>
              Preferences saved successfully!
            </SuccessMessage>
          )}

          <ActionButtons>
            <Button
              onClick={handleUpdatePreferences}
              disabled={isUpdatingPreferences}
            >
              {isUpdatingPreferences ? 'Saving...' : 'Save Changes'}
            </Button>
          </ActionButtons>
        </Card>

        {/* Notifications Section */}
        <Card>
          <SectionTitle>🔔 Notifications</SectionTitle>
          <ToggleGroup>
            <Toggle
              id="achievements"
              checked={notificationsState.achievements}
              onChange={(e) => setNotificationsState(prev => ({
                ...prev,
                achievements: e.target.checked
              }))}
            />
            <ToggleLabel htmlFor="achievements">
              🏆 Achievements
            </ToggleLabel>
          </ToggleGroup>

          <ToggleGroup>
            <Toggle
              id="teamUpdates"
              checked={notificationsState.teamUpdates}
              onChange={(e) => setNotificationsState(prev => ({
                ...prev,
                teamUpdates: e.target.checked
              }))}
            />
            <ToggleLabel htmlFor="teamUpdates">
              👥 Team Updates
            </ToggleLabel>
          </ToggleGroup>

          <ToggleGroup>
            <Toggle
              id="weeklyReports"
              checked={notificationsState.weeklyReports}
              onChange={(e) => setNotificationsState(prev => ({
                ...prev,
                weeklyReports: e.target.checked
              }))}
            />
            <ToggleLabel htmlFor="weeklyReports">
              📅 Weekly Reports
            </ToggleLabel>
          </ToggleGroup>

          <ToggleGroup>
            <Toggle
              id="deadlineReminders"
              checked={notificationsState.deadlineReminders}
              onChange={(e) => setNotificationsState(prev => ({
                ...prev,
                deadlineReminders: e.target.checked
              }))}
            />
            <ToggleLabel htmlFor="deadlineReminders">
              ⏰ Deadline Reminders
            </ToggleLabel>
          </ToggleGroup>

          <ToggleGroup>
            <Toggle
              id="wellnessReminders"
              checked={notificationsState.wellnessReminders}
              onChange={(e) => setNotificationsState(prev => ({
                ...prev,
                wellnessReminders: e.target.checked
              }))}
            />
            <ToggleLabel htmlFor="wellnessReminders">
              🧘 Wellness Reminders
            </ToggleLabel>
          </ToggleGroup>

          {showNotificationsSuccess && (
            <SuccessMessage>
              Notification settings updated successfully!
            </SuccessMessage>
          )}

          <Button
            variant="secondary"
            onClick={handleUpdateNotifications}
            disabled={isUpdatingNotifications}
          >
            {isUpdatingNotifications ? 'Updating...' : 'Update Notifications'}
          </Button>
        </Card>

        {/* Wellness Section */}
        <Card>
          <SectionTitle>🌿 Wellness Settings</SectionTitle>
          <ToggleGroup>
            <Toggle
              id="mindfulnessReminders"
              checked={wellnessState.mindfulnessReminders}
              onChange={(e) => setWellnessState(prev => ({
                ...prev,
                mindfulnessReminders: e.target.checked
              }))}
            />
            <ToggleLabel htmlFor="mindfulnessReminders">
              Mindfulness Reminders
            </ToggleLabel>
          </ToggleGroup>

          <ToggleGroup>
            <Toggle
              id="hydrationReminders"
              checked={wellnessState.hydrationReminders}
              onChange={(e) => setWellnessState(prev => ({
                ...prev,
                hydrationReminders: e.target.checked
              }))}
            />
            <ToggleLabel htmlFor="hydrationReminders">
              Hydration Reminders
            </ToggleLabel>
          </ToggleGroup>

          <ToggleGroup>
            <Toggle
              id="movementBreaks"
              checked={wellnessState.movementBreaks}
              onChange={(e) => setWellnessState(prev => ({
                ...prev,
                movementBreaks: e.target.checked
              }))}
            />
            <ToggleLabel htmlFor="movementBreaks">
              Movement Breaks
            </ToggleLabel>
          </ToggleGroup>

          <ToggleGroup>
            <Toggle
              id="eyeRest"
              checked={wellnessState.eyeRest}
              onChange={(e) => setWellnessState(prev => ({
                ...prev,
                eyeRest: e.target.checked
              }))}
            />
            <ToggleLabel htmlFor="eyeRest">
              Eye Rest Breaks
            </ToggleLabel>
          </ToggleGroup>

          <ToggleGroup>
            <Toggle
              id="endOfDay"
              checked={wellnessState.endOfDay}
              onChange={(e) => setWellnessState(prev => ({
                ...prev,
                endOfDay: e.target.checked
              }))}
            />
            <ToggleLabel htmlFor="endOfDay">
              End of Day Summary
            </ToggleLabel>
          </ToggleGroup>

          {showWellnessSuccess && (
            <SuccessMessage>
              Wellness settings updated successfully!
            </SuccessMessage>
          )}

          <Button
            variant="secondary"
            onClick={handleUpdateWellness}
            disabled={isUpdatingWellness}
          >
            {isUpdatingWellness ? 'Updating...' : 'Update Wellness Settings'}
          </Button>
        </Card>

        {/* Account Section */}
        <FullWidthSection>
          <Card>
            <SectionTitle>⚙️ Account Management</SectionTitle>
            <ButtonGroup>
              <Button variant="secondary" onClick={handleExportData}>
                📊 Export My Data
              </Button>
              <Button variant="secondary" onClick={handleDeleteAccount}>
                🗑️ Delete Account
              </Button>
            </ButtonGroup>

            <ButtonGroup>
              <Button variant="secondary">
                🌙 Theme: Light
              </Button>
              <Button variant="secondary">
                🌍 Language: English
              </Button>
            </ButtonGroup>

            <ButtonGroup>
              <Button variant="primary" onClick={handleSignOut}>
                🚪 Sign Out
              </Button>
            </ButtonGroup>
          </Card>
        </FullWidthSection>
      </ProfileGrid>
    </ProfileContainer>
  );
};