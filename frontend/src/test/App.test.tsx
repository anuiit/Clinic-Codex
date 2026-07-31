import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from '../App'
import type { AuthSession } from '../types'
import type { AdminTab } from '../pages/adminAnnotations/model'

const apiMock = vi.hoisted(() => ({
  createFirstAdmin: vi.fn(),
  getBootstrapStatus: vi.fn(),
  getAuthSession: vi.fn(),
  login: vi.fn(),
  logout: vi.fn(),
}))

vi.mock('../services/api', () => apiMock)

const AUTH_SESSION = {
  status: 'ok',
  auth_enabled: true,
  user: {
    id: 'user-1',
    email: 'admin@example.com',
    roles: ['org_admin'],
    permissions: [
      'analysis.submit',
      'annotation.queue.read',
      'annotation.review',
      'training.read',
      'training.run',
      'member.manage',
    ],
  },
  csrf_token: 'csrf-test-token',
} satisfies AuthSession

type RouteThemeProps = {
  themeMode?: 'dark' | 'light'
  onToggleTheme?: () => void
  authSlot?: ReactNode
}

type AdminRouteProps = RouteThemeProps & {
  initialTab?: AdminTab
}

vi.mock('../pages/WorkspacePage', () => ({
  default: ({ themeMode = 'dark', onToggleTheme = () => undefined, authSlot }: RouteThemeProps) => (
    <div>
      <div data-testid="workspace-page">Workspace route</div>
      <header>{authSlot}</header>
      <button type="button" onClick={onToggleTheme} aria-label="toggle workspace theme">
        {themeMode}
      </button>
    </div>
  ),
}))

vi.mock('../pages/AnnotationPage', () => ({
  default: ({ themeMode = 'dark', onToggleTheme = () => undefined, authSlot }: RouteThemeProps) => (
    <div>
      <div data-testid="annotation-page">Annotation route</div>
      <header>{authSlot}</header>
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
    authSlot,
  }: AdminRouteProps) => (
    <div>
      <div data-testid="admin-annotations-page">Admin annotation route</div>
      <div data-testid="admin-initial-tab">{initialTab}</div>
      <header>{authSlot}</header>
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
  vi.clearAllMocks()
})

beforeEach(() => {
  apiMock.getBootstrapStatus.mockResolvedValue({
    status: 'ok',
    auth_enabled: true,
    bootstrap_available: false,
  })
  apiMock.getAuthSession.mockResolvedValue(AUTH_SESSION)
})

describe('App route contracts', () => {
  it('renders the shared app shell and workspace route at /', async () => {
    const { container } = renderAt('/')

    expect(container.querySelector('main')).not.toBeNull()
    await waitFor(() => expect(screen.getByTestId('workspace-page')).toBeInTheDocument())
    expect(screen.getByRole('link', { name: 'Administration' })).toHaveAttribute(
      'href',
      '/admin/annotations/review',
    )
  })

  it('shows an admin access button for annotation reviewers', async () => {
    apiMock.getAuthSession.mockResolvedValueOnce({
      ...AUTH_SESSION,
      user: {
        id: 'reviewer-1',
        email: 'reviewer@example.com',
        roles: ['reviewer'],
        permissions: ['annotation.queue.read', 'annotation.review'],
      },
    } satisfies AuthSession)

    renderAt('/')

    const adminLink = await screen.findByRole('link', { name: 'Administration' })
    expect(adminLink).toHaveAttribute('href', '/admin/annotations/review')
  })

  it('sends ML operators directly to the training admin tab', async () => {
    apiMock.getAuthSession.mockResolvedValueOnce({
      ...AUTH_SESSION,
      user: {
        id: 'operator-1',
        email: 'operator@example.com',
        roles: ['ml_operator'],
        permissions: ['training.read', 'training.run'],
      },
    } satisfies AuthSession)

    renderAt('/')

    const adminLink = await screen.findByRole('link', { name: 'Administration' })
    expect(adminLink).toHaveAttribute('href', '/admin/annotations/training')
  })

  it('does not show admin access to contributors', async () => {
    apiMock.getAuthSession.mockResolvedValueOnce({
      ...AUTH_SESSION,
      user: {
        id: 'contributor-1',
        email: 'contributor@example.com',
        roles: ['contributor'],
        permissions: ['analysis.submit'],
      },
    } satisfies AuthSession)

    renderAt('/')

    await waitFor(() => expect(screen.getByTestId('workspace-page')).toBeInTheDocument())
    expect(screen.queryByRole('link', { name: 'Administration' })).not.toBeInTheDocument()
  })

  it('does not repeat the admin access button inside the admin area', async () => {
    renderAt('/admin/annotations/review')

    await waitFor(() => expect(screen.getByTestId('admin-annotations-page')).toBeInTheDocument())
    expect(screen.queryByRole('link', { name: 'Administration' })).not.toBeInTheDocument()
  })

  it('keeps the annotation page on the separate /annotate/:id route', async () => {
    renderAt('/annotate/alpha-run')

    await waitFor(() => expect(screen.getByTestId('annotation-page')).toBeInTheDocument())
    expect(screen.queryByTestId('workspace-page')).not.toBeInTheDocument()
  })

  it('wires the canonical admin annotation route with shared theme props', async () => {
    const user = userEvent.setup()
    const { container } = renderAt('/admin/annotations/review')

    await waitFor(() => expect(screen.getByTestId('admin-annotations-page')).toBeInTheDocument())
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
    await waitFor(() => expect(screen.getByTestId('admin-annotations-page')).toBeInTheDocument())
    expect(screen.getByTestId('admin-initial-tab')).toHaveTextContent('review')
  })

  it('passes valid admin tabs through to the admin page contract', async () => {
    const { unmount } = renderAt('/admin/annotations/dataset')

    await waitFor(() => expect(screen.getByTestId('admin-annotations-page')).toBeInTheDocument())
    expect(screen.getByTestId('admin-initial-tab')).toHaveTextContent('dataset')

    unmount()
    renderAt('/admin/annotations/training')

    await waitFor(() => expect(screen.getByTestId('admin-annotations-page')).toBeInTheDocument())
    expect(screen.getByTestId('admin-initial-tab')).toHaveTextContent('training')
  })

  it('redirects invalid admin tabs to the review tab', async () => {
    renderAt('/admin/annotations/not-a-tab')

    await waitFor(() => expect(window.location.pathname).toBe('/admin/annotations/review'))
    await waitFor(() => expect(screen.getByTestId('admin-annotations-page')).toBeInTheDocument())
    expect(screen.getByTestId('admin-initial-tab')).toHaveTextContent('review')
  })

  it('redirects the singular admin annotation URL to the working admin route', async () => {
    renderAt('/admin/annotation')

    await waitFor(() => expect(window.location.pathname).toBe('/admin/annotations/review'))
    await waitFor(() => expect(screen.getByTestId('admin-annotations-page')).toBeInTheDocument())
  })

  it('redirects the legacy dashboard route to the workspace without merging pages', async () => {
    renderAt('/dashboard')

    await waitFor(() => expect(window.location.pathname).toBe('/'))
    await waitFor(() => expect(screen.getByTestId('workspace-page')).toBeInTheDocument())
  })

  it('preserves legacy /analysis/:id handoff as a workspace query redirect', async () => {
    renderAt('/analysis/alpha run')

    await waitFor(() => {
      expect(window.location.pathname).toBe('/')
      expect(window.location.search).toBe('?analysis=alpha%20run')
    })
    await waitFor(() => expect(screen.getByTestId('workspace-page')).toBeInTheDocument())
  })
  it('shows a dedicated login form at /login', () => {
    apiMock.getAuthSession.mockImplementationOnce(() => new Promise<AuthSession>(() => undefined));
    renderAt('/login')

    expect(screen.getByRole('heading', { name: /sign in/i })).toBeInTheDocument()
    expect(screen.getByLabelText(/email/i)).toBeInTheDocument()
    expect(screen.getByLabelText(/password/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument()
    expect(screen.queryByTestId('workspace-page')).not.toBeInTheDocument()
    expect(screen.queryByTestId('admin-annotations-page')).not.toBeInTheDocument()
  })
  it('creates the first local administrator and opens the admin area', async () => {
    const user = userEvent.setup()
    apiMock.getAuthSession.mockRejectedValueOnce(new Response('', { status: 401 }))
    apiMock.getBootstrapStatus.mockResolvedValueOnce({
      status: 'ok',
      auth_enabled: true,
      bootstrap_available: true,
    })
    apiMock.createFirstAdmin.mockResolvedValueOnce(AUTH_SESSION)
    renderAt('/login')

    expect(await screen.findByRole('heading', { name: /premier compte administrateur/i })).toBeInTheDocument()
    await user.type(screen.getByLabelText(/^email$/i), 'admin@example.test')
    await user.type(screen.getByLabelText(/^password$/i), 'a-secure-local-password')
    await user.type(screen.getByLabelText(/confirm password/i), 'a-secure-local-password')
    await user.click(screen.getByRole('button', { name: /create administrator account/i }))

    expect(apiMock.createFirstAdmin).toHaveBeenCalledWith({
      email: 'admin@example.test',
      password: 'a-secure-local-password',
    })
    await waitFor(() => expect(window.location.pathname).toBe('/admin/annotations/review'))
  })
  it('validates matching passwords before creating the first administrator', async () => {
    const user = userEvent.setup()
    apiMock.getBootstrapStatus.mockResolvedValueOnce({
      status: 'ok',
      auth_enabled: true,
      bootstrap_available: true,
    })
    renderAt('/login')

    await screen.findByRole('heading', { name: /premier compte administrateur/i })
    await user.type(screen.getByLabelText(/^email$/i), 'admin@example.test')
    await user.type(screen.getByLabelText(/^password$/i), 'a-secure-local-password')
    await user.type(screen.getByLabelText(/confirm password/i), 'different-password')
    await user.click(screen.getByRole('button', { name: /create administrator account/i }))

    expect(await screen.findByRole('alert')).toHaveTextContent(/passwords do not match/i)
    expect(apiMock.createFirstAdmin).not.toHaveBeenCalled()
  })
  it('focuses and associates a failed sign-in message', async () => {
    const user = userEvent.setup()
    apiMock.login.mockRejectedValueOnce({ response: { data: { error: 'Invalid credentials' } } })
    renderAt('/login')

    await user.type(screen.getByLabelText(/email/i), 'admin@example.test')
    await user.type(screen.getByLabelText(/password/i), 'wrong-password')
    await user.click(screen.getByRole('button', { name: /^sign in$/i }))

    const alert = await screen.findByRole('alert')
    expect(alert).toHaveFocus()
    expect(screen.getByRole('form', { name: /sign in/i })).toHaveAttribute('aria-describedby', 'login-error')
  })

  it('redirects unauthenticated admin visitors to /login', async () => {
    apiMock.getAuthSession.mockRejectedValueOnce(new Response('', { status: 401 }))
    renderAt('/admin/annotations/review')

    await waitFor(() => expect(window.location.pathname).toBe('/login'))
    expect(screen.queryByTestId('admin-annotations-page')).not.toBeInTheDocument()
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
