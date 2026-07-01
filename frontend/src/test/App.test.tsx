import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from '../App'
import type { AdminTab } from '../pages/adminAnnotations/model'

type RouteThemeProps = {
  themeMode?: 'dark' | 'light'
  onToggleTheme?: () => void
}

type AdminRouteProps = RouteThemeProps & {
  initialTab?: AdminTab
}

vi.mock('../pages/WorkspacePage', () => ({
  default: ({ themeMode = 'dark', onToggleTheme = () => undefined }: RouteThemeProps) => (
    <div>
      <div data-testid="workspace-page">Workspace route</div>
      <button type="button" onClick={onToggleTheme} aria-label="toggle workspace theme">
        {themeMode}
      </button>
    </div>
  ),
}))

vi.mock('../pages/AnnotationPage', () => ({
  default: ({ themeMode = 'dark', onToggleTheme = () => undefined }: RouteThemeProps) => (
    <div>
      <div data-testid="annotation-page">Annotation route</div>
      <button type="button" onClick={onToggleTheme} aria-label="toggle annotation theme">
        {themeMode}
      </button>
    </div>
  ),
}))

vi.mock('../pages/AdminAnnotationsPage', () => ({
  default: ({
    themeMode = 'dark',
    onToggleTheme = () => undefined,
    initialTab = 'review',
  }: AdminRouteProps) => (
    <div>
      <div data-testid="admin-annotations-page">Admin annotation route</div>
      <div data-testid="admin-initial-tab">{initialTab}</div>
      <button type="button" onClick={onToggleTheme} aria-label="toggle admin theme">
        {themeMode}
      </button>
    </div>
  ),
}))

function renderAt(path: string) {
  window.history.pushState({}, '', path)
  return render(<App />)
}

afterEach(() => {
  window.history.pushState({}, '', '/')
  window.localStorage.clear()
  delete document.documentElement.dataset.theme
})

describe('App route contracts', () => {
  it('renders the shared app shell and workspace route at /', () => {
    const { container } = renderAt('/')

    expect(container.querySelector('main')).not.toBeNull()
    expect(screen.getByTestId('workspace-page')).toBeInTheDocument()
  })

  it('keeps the annotation page on the separate /annotate/:id route', () => {
    renderAt('/annotate/alpha-run')

    expect(screen.getByTestId('annotation-page')).toBeInTheDocument()
    expect(screen.queryByTestId('workspace-page')).not.toBeInTheDocument()
  })

  it('wires the canonical admin annotation route with shared theme props', async () => {
    const user = userEvent.setup()
    const { container } = renderAt('/admin/annotations/review')

    expect(screen.getByTestId('admin-annotations-page')).toBeInTheDocument()
    expect(screen.getByTestId('admin-initial-tab')).toHaveTextContent('review')
    expect(screen.queryByTestId('workspace-page')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /retrain/i })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'toggle admin theme' })).toHaveTextContent('dark')

    await user.click(screen.getByRole('button', { name: 'toggle admin theme' }))

    await waitFor(() => expect(document.documentElement.dataset.theme).toBe('light'))
    expect(container.querySelector('.app-shell')).toHaveAttribute('data-theme', 'light')
  })


  it('redirects the legacy plural admin URL to the review tab', async () => {
    renderAt('/admin/annotations')

    await waitFor(() => expect(window.location.pathname).toBe('/admin/annotations/review'))
    expect(screen.getByTestId('admin-annotations-page')).toBeInTheDocument()
    expect(screen.getByTestId('admin-initial-tab')).toHaveTextContent('review')
  })

  it('passes valid admin tabs through to the admin page contract', () => {
    const { unmount } = renderAt('/admin/annotations/dataset')

    expect(screen.getByTestId('admin-annotations-page')).toBeInTheDocument()
    expect(screen.getByTestId('admin-initial-tab')).toHaveTextContent('dataset')

    unmount()
    renderAt('/admin/annotations/training')

    expect(screen.getByTestId('admin-annotations-page')).toBeInTheDocument()
    expect(screen.getByTestId('admin-initial-tab')).toHaveTextContent('training')
  })

  it('redirects invalid admin tabs to the review tab', async () => {
    renderAt('/admin/annotations/not-a-tab')

    await waitFor(() => expect(window.location.pathname).toBe('/admin/annotations/review'))
    expect(screen.getByTestId('admin-annotations-page')).toBeInTheDocument()
    expect(screen.getByTestId('admin-initial-tab')).toHaveTextContent('review')
  })

  it('redirects the singular admin annotation URL to the working admin route', async () => {
    renderAt('/admin/annotation')

    await waitFor(() => expect(window.location.pathname).toBe('/admin/annotations/review'))
    expect(screen.getByTestId('admin-annotations-page')).toBeInTheDocument()
  })

  it('redirects the legacy dashboard route to the workspace without merging pages', async () => {
    renderAt('/dashboard')

    await waitFor(() => expect(window.location.pathname).toBe('/'))
    expect(screen.getByTestId('workspace-page')).toBeInTheDocument()
  })

  it('preserves legacy /analysis/:id handoff as a workspace query redirect', async () => {
    renderAt('/analysis/alpha run')

    await waitFor(() => {
      expect(window.location.pathname).toBe('/')
      expect(window.location.search).toBe('?analysis=alpha%20run')
    })
    expect(screen.getByTestId('workspace-page')).toBeInTheDocument()
  })

  it('applies and persists the shared dark/light theme from app state', async () => {
    const user = userEvent.setup()
    const { container } = renderAt('/')

    await waitFor(() => expect(document.documentElement.dataset.theme).toBe('dark'))
    expect(container.querySelector('.app-shell')).toHaveAttribute('data-theme', 'dark')

    await user.click(screen.getByRole('button', { name: 'toggle workspace theme' }))

    await waitFor(() => expect(document.documentElement.dataset.theme).toBe('light'))
    expect(window.localStorage.getItem('clinic-codex-theme')).toBe('light')
    expect(container.querySelector('.app-shell')).toHaveAttribute('data-theme', 'light')
  })

  it('hydrates the shared theme from localStorage', async () => {
    window.localStorage.setItem('clinic-codex-theme', 'light')

    const { container } = renderAt('/annotate/alpha-run')

    await waitFor(() => expect(document.documentElement.dataset.theme).toBe('light'))
    expect(container.querySelector('.app-shell')).toHaveAttribute('data-theme', 'light')
    expect(screen.getByRole('button', { name: 'toggle annotation theme' })).toHaveTextContent('light')
  })
})
