import { fetchBaseQuery, FetchBaseQueryError } from '@reduxjs/toolkit/query';
import { RootState } from '../index';
import { setTokens, clearAuth } from '../slices/authSlice';

/**
 * Custom base query that handles automatic token refresh
 */
export const baseQueryWithReauth = fetchBaseQuery({
  baseUrl: '/api',
  prepareHeaders: (headers, { getState }) => {
    const token = (getState() as RootState).auth.token;
    if (token) {
      headers.set('authorization', `Bearer ${token}`);
    }
    return headers;
  },
});

/**
 * Wrapper around baseQueryWithReauth that handles token refresh
 */
export const baseQueryWithAuth = async (args: any, api: any, extraOptions: any) => {
  let result = await baseQueryWithReauth(args, api, extraOptions);

  // Handle token expiration
  if (result.error && result.error.status === 401) {
    console.log('🔄 Token expired, attempting to refresh...');

    // Try to get a new token
    const refreshResult = await baseQueryWithReauth(
      {
        url: 'auth/refresh',
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${(api.getState() as RootState).auth.refreshToken}`,
        },
      },
      api,
      extraOptions
    );

    if (refreshResult.data) {
      // Token refresh successful, update the tokens in the store
      const { token, refreshToken } = refreshResult.data as { token: string; refreshToken: string };
      api.dispatch(setTokens({ token, refreshToken }));

      // Retry the original request with the new token
      result = await baseQueryWithReauth(args, api, extraOptions);
      console.log('✅ Token refreshed successfully, retrying original request');
    } else {
      // Token refresh failed, clear auth state
      console.log('❌ Token refresh failed, clearing auth state');
      api.dispatch(clearAuth());

      // Return a more descriptive error
      result = {
        error: {
          status: 401,
          data: {
            message: 'Authentication expired. Please log in again.',
          },
        },
      } as FetchBaseQueryError;
    }
  }

  return result;
};