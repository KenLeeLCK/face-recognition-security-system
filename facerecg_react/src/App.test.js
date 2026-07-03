import { render, screen } from '@testing-library/react';
import App from './App';

test('renders face recognition app title', () => {
  render(<App />);
  const titleElement = screen.getByRole('heading', { name: /face recognition/i });
  expect(titleElement).toBeInTheDocument();
});
