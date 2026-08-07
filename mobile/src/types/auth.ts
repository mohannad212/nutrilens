// Mirrors backend/app/schemas/auth.py field-for-field (03-api-specification.md §2).
// Keep in sync manually until codegen is worth the setup for a project this size.

export type HealthGoal = 'weight_loss' | 'maintenance' | 'weight_gain';

export interface RegisterRequest {
  email: string;
  username: string;
  password: string;
  health_goal: HealthGoal;
}

export interface RegisterResponse {
  user_id: number;
  username: string;
  email: string;
  health_goal: HealthGoal;
  created_at: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface LoginUser {
  user_id: number;
  username: string;
  health_goal: HealthGoal;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  user: LoginUser;
}
