export type User = {
  id: string;
  email: string;
  fullName?: string;
  role: 'admin' | 'teacher';
};

const TOKEN_KEY = 'vultus-token';
const USER_KEY = 'vultus-user';

export const getToken = () => localStorage.getItem(TOKEN_KEY);

export const setSession = (token: string, user: User) => {
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USER_KEY, JSON.stringify(user));
};

export const updateStoredUser = (user: User) => {
  localStorage.setItem(USER_KEY, JSON.stringify(user));
};

export const clearSession = () => {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
};

export const getUser = (): User | null => {
  const raw = localStorage.getItem(USER_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as User;
  } catch {
    return null;
  }
};
