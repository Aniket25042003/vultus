import { render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { it, expect } from 'vitest';
import LandingPage from '../pages/LandingPage';

it('renders landing headline', () => {
  render(
    <BrowserRouter>
      <LandingPage />
    </BrowserRouter>
  );

  expect(screen.getByText(/Let campus spaces/i)).toBeInTheDocument();
});
