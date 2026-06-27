import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import AdminAnnotationsPage from './AdminAnnotationsPage';
import type { AdminAnnotationQueue, AdminTrainingJob, AdminTrainingSummary } from '../types';

const apiMock = vi.hoisted(() => ({
  adminAnnotationMediaUrl: vi.fn((path: string) => `http://api.test${path}`),
  getAdminAnnotationQueue: vi.fn(),
  getAdminTrainingSummary: vi.fn(),
  getLatestAdminTrainingJob: vi.fn(),
  modifyAdminAnnotationElement: vi.fn(),
  setAdminAnnotationReviewStatus: vi.fn(),
  startAdminTrainingJob: vi.fn(),
}));

vi.mock('../services/api', () => apiMock);

function queueWithStatuses(status0: 'pending' | 'approved' | 'rejected', status1: 'pending' | 'approved' | 'rejected'): AdminAnnotationQueue {
  const trainable0 = status0 === 'approved';
  const trainable1 = status1 === 'approved';
  return {
    status: 'ok',
    schema_version: 1,
    local_only: true,
    warning: 'Local/dev-only annotation review endpoint. It is not production-secured.',
    counts: {
      total: 2,
      pending: [status0, status1].filter((status) => status === 'pending').length,
      approved: [status0, status1].filter((status) => status === 'approved').length,
      rejected: [status0, status1].filter((status) => status === 'rejected').length,
      trainable: [trainable0, trainable1].filter(Boolean).length,
    },
    analyses: [
      {
        analysis_id: 'analysis-1',
        uploaded_at: '2026-05-26T13:00:00+00:00',
        image_path: '/tmp/annotations/analysis-1/image.png',
        image_url: '/admin/annotations/analysis-1/image',
        image_exists: true,
        elements: [
          {
            key: 'analysis-1:0',
            analysis_id: 'analysis-1',
            index: 0,
            class_name: 'atl',
            bbox: [0, 1, 2, 3],
            crop_path: '/tmp/annotations/analysis-1/elements/0.png',
            crop_url: '/admin/annotations/analysis-1/0/crop',
            crop_exists: true,
            review_status: status0,
            trainable: trainable0,
            source_fingerprint: 'fingerprint-0',
            stale_decision: false,
          },
          {
            key: 'analysis-1:1',
            analysis_id: 'analysis-1',
            index: 1,
            class_name: 'calli',
            bbox: [4, 5, 6, 7],
            crop_path: '/tmp/annotations/analysis-1/elements/1.png',
            crop_url: '/admin/annotations/analysis-1/1/crop',
            crop_exists: true,
            review_status: status1,
            trainable: trainable1,
            source_fingerprint: 'fingerprint-1',
            stale_decision: false,
          },
        ],
      },
    ],
    diagnostics: [],
  };
}


function trainingJob(overrides: Partial<AdminTrainingJob> = {}): AdminTrainingJob {
  return {
    run_id: 'run-1',
    status: 'running',
    dry_run: true,
    device: 'cpu',
    batch_size: 8,
    started_at: '2026-05-26T14:00:00+00:00',
    exit_code: null,
    log_tail: ['stage 1'],
    ...overrides,
  };
}

function trainingSummary(overrides: Partial<AdminTrainingSummary> = {}): AdminTrainingSummary {
  return {
    status: 'ok',
    local_only: true,
    warning: 'local only',
    training_jobs_enabled: false,
    launch_allowed_for_request: false,
    launch_disabled_reasons: ['disabled_by_default: set ENABLE_ADMIN_TRAINING_JOBS=1 to allow local launches'],
    data: {
      total: 4,
      pending: 1,
      approved: 2,
      rejected: 1,
      trainable: 1,
      classes: ['atl'],
      per_class: { atl: 1 },
      diagnostics: [],
    },
    parameters: {
      editable: { dry_run: true, device: ['auto', 'cpu', 'mps', 'cuda'], batch_size: { default: 16, min: 1, max: 256 } },
      script_env_defaults: { BATCH_SIZE: '16', DEVICE: 'auto' },
      config: { training: { num_epochs: 100 }, model: { backbone: 'dinov2_vits14' } },
    },
    paths: { script: '/repo/scripts/retrain.sh', runs_dir: '/repo/backend/training_runs', model_dir_override_active: false },
    artifacts: {
      approved_export_manifest: { path: '/repo/backend/training_data/approved/Elements/_approved_export_manifest.json', exists: true, sha256: 'abc123' },
      model_registry: { status: 'ok', promoted_version: '20260527T010203Z-demo' },
    },
    latest_job: null,
    ...overrides,
  };
}

describe('AdminAnnotationsPage', () => {
  beforeEach(() => {
    apiMock.adminAnnotationMediaUrl.mockClear();
    apiMock.getAdminAnnotationQueue.mockReset();
    apiMock.getAdminTrainingSummary.mockReset();
    apiMock.getLatestAdminTrainingJob.mockReset();
    apiMock.modifyAdminAnnotationElement.mockReset();
    apiMock.setAdminAnnotationReviewStatus.mockReset();
    apiMock.startAdminTrainingJob.mockReset();
  });

  it('renders local-only warning, grouped queue, visual context, statuses, and counters', async () => {
    apiMock.getAdminAnnotationQueue.mockResolvedValueOnce(queueWithStatuses('pending', 'rejected'));

    render(<AdminAnnotationsPage themeMode="light" onToggleTheme={vi.fn()} />);

    expect(await screen.findByRole('heading', { name: /annotation admin console/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /activer le mode sombre/i })).toBeInTheDocument();
    expect(await screen.findByRole('tablist', { name: /admin annotation sections/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /review/i })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('tab', { name: /dataset/i })).toHaveAttribute('aria-selected', 'false');
    expect(screen.getByRole('tab', { name: /training/i })).toHaveAttribute('aria-selected', 'false');
    expect(screen.getByText(/not production-secured/i)).toBeInTheDocument();
    expect(screen.getByRole('img', { name: /original submission analysis-1/i })).toHaveAttribute(
      'src',
      'http://api.test/admin/annotations/analysis-1/image',
    );
    expect(screen.getByRole('listbox', { name: /compact review queue/i })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: /select review element 0 atl/i })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('heading', { name: /element #0 · atl/i })).toBeInTheDocument();
    expect(screen.getByRole('img', { name: 'Crop 0 for atl' })).toBeInTheDocument();
    expect(screen.getAllByLabelText(/Pending review status/i)).toHaveLength(2);
    expect(screen.getByLabelText(/Rejected review status/i)).toBeInTheDocument();
    expect(screen.getByText('Total')).toBeInTheDocument();
    expect(screen.getByText('Trainable')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /retrain/i })).not.toBeInTheDocument();
  });

  it('selects compact Review rows and navigates the inspector within active filters', async () => {
    const user = userEvent.setup();
    apiMock.getAdminAnnotationQueue.mockResolvedValueOnce(queueWithStatuses('pending', 'rejected'));

    render(<AdminAnnotationsPage />);

    const firstRow = await screen.findByRole('option', { name: /select review element 0 atl/i });
    const secondRow = screen.getByRole('option', { name: /select review element 1 calli/i });
    expect(firstRow).toHaveAttribute('aria-selected', 'true');

    await user.click(secondRow);

    expect(screen.getByRole('heading', { name: /element #1 · calli/i })).toBeInTheDocument();
    expect(secondRow).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('button', { name: /next element/i })).toBeDisabled();

    await user.click(screen.getByRole('button', { name: /previous element/i }));
    expect(screen.getByRole('heading', { name: /element #0 · atl/i })).toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText(/review status filter/i), 'rejected');
    expect(screen.getByRole('option', { name: /select review element 1 calli/i })).toBeInTheDocument();
    expect(screen.queryByRole('option', { name: /select review element 0 atl/i })).not.toBeInTheDocument();

    await user.click(screen.getByRole('option', { name: /select review element 1 calli/i }));
    expect(screen.getByRole('heading', { name: /element #1 · calli/i })).toBeInTheDocument();
  });

  it('switches between accessible Dataset and Training tabs without refetching the queue', async () => {
    const user = userEvent.setup();
    apiMock.getAdminAnnotationQueue.mockResolvedValueOnce(queueWithStatuses('approved', 'rejected'));
    apiMock.getAdminTrainingSummary.mockResolvedValueOnce(trainingSummary());

    render(<AdminAnnotationsPage />);
    await screen.findByRole('tab', { name: /dataset/i });

    await user.click(screen.getByRole('tab', { name: /dataset/i }));
    expect(screen.getByRole('tab', { name: /dataset/i })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('tabpanel', { name: /dataset tab panel/i })).toHaveTextContent(/approved-only dataset overview/i);

    await user.click(screen.getByRole('tab', { name: /training/i }));
    expect(screen.getByRole('tab', { name: /training/i })).toHaveAttribute('aria-selected', 'true');
    expect(await screen.findByRole('heading', { name: /guarded local training/i })).toBeInTheDocument();
    expect(apiMock.getAdminAnnotationQueue).toHaveBeenCalledTimes(1);
  });

  it('separates Dataset trainable, rejected, and diagnostic buckets with filters and jump-to-review', async () => {
    const user = userEvent.setup();
    const mixed = queueWithStatuses('approved', 'approved');
    mixed.analyses[0].elements[1] = {
      ...mixed.analyses[0].elements[1],
      trainable: false,
      stale_decision: true,
    };
    mixed.analyses[0].elements.push(
      {
        ...mixed.analyses[0].elements[0],
        key: 'analysis-1:2',
        index: 2,
        class_name: 'atl',
        bbox: [2, 2, 4, 4],
        crop_url: '/admin/annotations/analysis-1/2/crop',
        review_status: 'rejected',
        trainable: false,
        source_fingerprint: 'fingerprint-2',
      },
      {
        ...mixed.analyses[0].elements[0],
        key: 'analysis-1:3',
        index: 3,
        class_name: 'maya',
        bbox: [3, 3, 4, 4],
        crop_url: '/admin/annotations/analysis-1/3/crop',
        review_status: 'pending',
        trainable: false,
        source_fingerprint: 'fingerprint-3',
      },
    );
    mixed.counts = { total: 4, pending: 1, approved: 2, rejected: 1, trainable: 1 };
    mixed.diagnostics = [
      { code: 'stale_decision', message: 'fingerprint mismatch', key: 'analysis-1:1', analysis_id: 'analysis-1', index: 1 },
    ];
    apiMock.getAdminAnnotationQueue.mockResolvedValueOnce(mixed);

    render(<AdminAnnotationsPage />);
    await user.click(await screen.findByRole('tab', { name: /dataset/i }));

    expect(screen.getByText(/trainable class distribution/i)).toBeInTheDocument();
    expect(screen.getByRole('listbox', { name: /dataset review rows/i })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: /select dataset element 0 atl/i })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('heading', { name: /dataset element #0 · atl/i })).toBeInTheDocument();
    expect(screen.getByRole('img', { name: 'Dataset crop 0 for atl' })).toBeInTheDocument();
    expect(screen.getByText(/stale_decision: fingerprint mismatch/i)).toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText(/dataset status filter/i), 'rejected');
    expect(screen.getByRole('img', { name: 'Dataset crop 2 for atl' })).toBeInTheDocument();
    expect(screen.queryByRole('img', { name: 'Dataset crop 0 for atl' })).not.toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText(/dataset status filter/i), 'all');
    await user.selectOptions(screen.getByLabelText(/dataset class filter/i), 'calli');
    expect(screen.getByRole('img', { name: 'Dataset crop 1 for calli' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /dataset element #1 · calli/i })).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /jump to review element 1/i }));

    expect(screen.getByRole('tab', { name: /review/i })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('heading', { name: /element #1 · calli/i })).toBeInTheDocument();
    expect(screen.queryByLabelText(/class name/i)).not.toBeInTheDocument();
  });

  it('renders Training tab disabled summary, parameters, and CLI alternative', async () => {
    const user = userEvent.setup();
    apiMock.getAdminAnnotationQueue.mockResolvedValueOnce(queueWithStatuses('approved', 'rejected'));
    apiMock.getAdminTrainingSummary.mockResolvedValueOnce(trainingSummary());

    render(<AdminAnnotationsPage />);
    await user.click(await screen.findByRole('tab', { name: /training/i }));

    expect(await screen.findByRole('heading', { name: /guarded local training/i })).toBeInTheDocument();
    expect(screen.getByText(/launch disabled for this request/i)).toBeInTheDocument();
    expect(screen.getByText(/disabled_by_default/i)).toBeInTheDocument();
    expect(screen.getAllByText(/enable_admin_training_jobs=1/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/committed default remains disabled/i)).toBeInTheDocument();
    expect(screen.getByText(/bash scripts\/retrain\.sh --dry-run/i)).toBeInTheDocument();
    expect(screen.getByRole('region', { name: /training pre-action summary/i })).toHaveTextContent(/blocked by backend guard/i);
    expect(screen.getByText(/dry run validates approved-only export\/training wiring/i)).toBeInTheDocument();
    expect(screen.getByText(/atl: 1/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /start dry run/i })).toBeDisabled();
  });

  it('keeps Training launch disabled from backend permission even when visible data looks valid', async () => {
    const user = userEvent.setup();
    apiMock.getAdminAnnotationQueue.mockResolvedValueOnce(queueWithStatuses('approved', 'approved'));
    apiMock.getAdminTrainingSummary.mockResolvedValueOnce(trainingSummary({
      training_jobs_enabled: true,
      launch_allowed_for_request: false,
      launch_disabled_reasons: ['nonlocal_request: use loopback localhost'],
      data: {
        total: 2,
        pending: 0,
        approved: 2,
        rejected: 0,
        trainable: 2,
        classes: ['atl', 'calli'],
        per_class: { atl: 1, calli: 1 },
        diagnostics: [],
      },
    }));

    render(<AdminAnnotationsPage />);
    await user.click(await screen.findByRole('tab', { name: /training/i }));

    const summary = await screen.findByRole('region', { name: /training pre-action summary/i });
    expect(within(summary).getByText(/2 approved crops/i)).toBeInTheDocument();
    expect(within(summary).getByText(/atl, calli/i)).toBeInTheDocument();
    expect(within(summary).getByText(/blocked by backend guard/i)).toBeInTheDocument();
    expect(screen.getByText(/nonlocal_request/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /start dry run/i })).toBeDisabled();
  });

  it('updates Training pre-action summary for dry-run versus full-training choices', async () => {
    const user = userEvent.setup();
    apiMock.getAdminAnnotationQueue.mockResolvedValueOnce(queueWithStatuses('approved', 'rejected'));
    apiMock.getAdminTrainingSummary.mockResolvedValueOnce(trainingSummary({
      training_jobs_enabled: true,
      launch_allowed_for_request: true,
      launch_disabled_reasons: [],
    }));

    render(<AdminAnnotationsPage />);
    await user.click(await screen.findByRole('tab', { name: /training/i }));

    const summary = await screen.findByRole('region', { name: /training pre-action summary/i });
    expect(within(summary).getByText(/ready for local launch/i)).toBeInTheDocument();
    expect(within(summary).getByText(/dry run selected/i)).toBeInTheDocument();
    expect(within(summary).getByText(/dry run validates approved-only export\/training wiring/i)).toBeInTheDocument();
    expect(within(summary).getByText(/auto/i)).toBeInTheDocument();
    expect(within(summary).getByText('16')).toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText(/dry run/i), 'no');
    await user.selectOptions(screen.getByLabelText(/device/i), 'cpu');
    await user.clear(screen.getByLabelText(/batch size/i));
    await user.type(screen.getByLabelText(/batch size/i), '8');

    expect(within(summary).getByText(/full training selected/i)).toBeInTheDocument();
    expect(within(summary).getByText(/candidate package under backend\/model_registry\/versions/i)).toBeInTheDocument();
    expect(within(summary).getByText(/scripts\/promote_model\.py <version_id>/i)).toBeInTheDocument();
    expect(within(summary).getByText(/restart the backend/i)).toBeInTheDocument();
    expect(within(summary).queryByText(/can overwrite classifier artifacts/i)).not.toBeInTheDocument();
    expect(within(summary).getByText(/cpu/i)).toBeInTheDocument();
    expect(within(summary).getByText('8')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /artifact status/i })).toBeInTheDocument();
    expect(screen.getByText('approved_export_manifest')).toBeInTheDocument();
    expect(screen.getByText('model_registry')).toBeInTheDocument();
    expect(screen.getByText(/20260527T010203Z-demo/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /start training/i })).toBeEnabled();
  });

  it('warns when MODEL_DIR override can decouple promotion from the loaded runtime', async () => {
    const user = userEvent.setup();
    apiMock.getAdminAnnotationQueue.mockResolvedValueOnce(queueWithStatuses('approved', 'rejected'));
    apiMock.getAdminTrainingSummary.mockResolvedValueOnce(trainingSummary({
      training_jobs_enabled: true,
      launch_allowed_for_request: true,
      launch_disabled_reasons: [],
      paths: {
        script: '/repo/scripts/retrain.sh',
        runs_dir: '/repo/backend/training_runs',
        model_dir_override_active: true,
      },
    }));

    render(<AdminAnnotationsPage />);
    await user.click(await screen.findByRole('tab', { name: /training/i }));

    expect(await screen.findByText(/model_dir override active/i)).toBeInTheDocument();
    const warningCopy = screen.getByText(/promotion to/i);
    expect(warningCopy).toHaveTextContent(/backend\/codex_model/);
    expect(warningCopy).toHaveTextContent(/until\s*model_dir\s*is unset/i);
  });

  it('starts a guarded dry-run training job and displays log tail', async () => {
    const user = userEvent.setup();
    const started = trainingJob({ run_id: 'run-started', status: 'running', log_tail: ['mock dry run started'] });
    apiMock.getAdminAnnotationQueue.mockResolvedValueOnce(queueWithStatuses('approved', 'rejected'));
    apiMock.getAdminTrainingSummary.mockResolvedValueOnce(trainingSummary({
      training_jobs_enabled: true,
      launch_allowed_for_request: true,
      launch_disabled_reasons: [],
    }));
    apiMock.startAdminTrainingJob.mockResolvedValueOnce({ status: 'ok', local_only: true, job: started });

    render(<AdminAnnotationsPage />);
    await user.click(await screen.findByRole('tab', { name: /training/i }));
    await screen.findByRole('button', { name: /start dry run/i });
    await user.selectOptions(screen.getByLabelText(/device/i), 'cpu');
    await user.clear(screen.getByLabelText(/batch size/i));
    await user.type(screen.getByLabelText(/batch size/i), '8');
    await user.type(screen.getByLabelText(/notes/i), 'smoke');

    await user.click(screen.getByRole('button', { name: /start dry run/i }));

    await waitFor(() => {
      expect(apiMock.startAdminTrainingJob).toHaveBeenCalledWith({
        dry_run: true,
        device: 'cpu',
        batch_size: 8,
        notes: 'smoke',
      });
    });
    expect(await screen.findByText(/latest job run-started/i)).toBeInTheDocument();
    expect(screen.getByText(/mock dry run started/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /training job running/i })).toBeDisabled();
  });

  it('polls a running Training job until terminal status', async () => {
    const user = userEvent.setup();
    apiMock.getAdminAnnotationQueue.mockResolvedValueOnce(queueWithStatuses('approved', 'rejected'));
    apiMock.getAdminTrainingSummary.mockResolvedValueOnce(trainingSummary({ latest_job: trainingJob({ run_id: 'run-poll', status: 'running' }) }));
    apiMock.getLatestAdminTrainingJob.mockResolvedValueOnce({
      status: 'ok',
      local_only: true,
      job: trainingJob({ run_id: 'run-poll', status: 'succeeded', exit_code: 0, log_tail: ['done'] }),
    });

    render(<AdminAnnotationsPage />);
    await user.click(await screen.findByRole('tab', { name: /training/i }));
    expect(await screen.findByText(/latest job run-poll/i)).toBeInTheDocument();

    await waitFor(() => expect(screen.getByText('succeeded')).toBeInTheDocument(), { timeout: 2500 });
    expect(screen.getByText(/done/i)).toBeInTheDocument();
  });

  it('approves one element and reloads the queue without changing other statuses locally', async () => {
    const user = userEvent.setup();
    apiMock.getAdminAnnotationQueue
      .mockResolvedValueOnce(queueWithStatuses('pending', 'rejected'))
      .mockResolvedValueOnce(queueWithStatuses('approved', 'rejected'));
    apiMock.setAdminAnnotationReviewStatus.mockResolvedValueOnce({
      status: 'ok',
      local_only: true,
      warning: 'local only',
      element: queueWithStatuses('approved', 'rejected').analyses[0].elements[0],
    });

    render(<AdminAnnotationsPage />);
    await screen.findByRole('button', { name: /approve element 0/i });

    await user.click(screen.getByRole('button', { name: /approve element 0/i }));

    await waitFor(() => {
      expect(apiMock.setAdminAnnotationReviewStatus).toHaveBeenCalledWith('analysis-1', 0, 'approved');
    });
    expect(await screen.findAllByLabelText(/Approved review status/i)).toHaveLength(2);
    expect(screen.getByLabelText(/Rejected review status/i)).toBeInTheDocument();
  });

  it('edits class and bbox, saving through the modify endpoint as pending', async () => {
    const user = userEvent.setup();
    const initial = queueWithStatuses('approved', 'rejected');
    const updated = queueWithStatuses('pending', 'rejected');
    updated.analyses[0].elements[0] = {
      ...updated.analyses[0].elements[0],
      class_name: 'new-atl',
      bbox: [1, 2, 5, 6],
      trainable: false,
    };
    apiMock.getAdminAnnotationQueue.mockResolvedValueOnce(initial).mockResolvedValueOnce(updated);
    apiMock.modifyAdminAnnotationElement.mockResolvedValueOnce({
      status: 'ok',
      local_only: true,
      warning: 'local only',
      element: updated.analyses[0].elements[0],
    });

    render(<AdminAnnotationsPage />);
    await user.click(await screen.findByRole('button', { name: /edit element 0/i }));
    await user.clear(screen.getByLabelText(/class name/i));
    await user.type(screen.getByLabelText(/class name/i), 'new-atl');
    await user.clear(screen.getByLabelText(/bbox x for element 0/i));
    await user.type(screen.getByLabelText(/bbox x for element 0/i), '1');
    await user.clear(screen.getByLabelText(/bbox y for element 0/i));
    await user.type(screen.getByLabelText(/bbox y for element 0/i), '2');
    await user.clear(screen.getByLabelText(/bbox w for element 0/i));
    await user.type(screen.getByLabelText(/bbox w for element 0/i), '5');
    await user.clear(screen.getByLabelText(/bbox h for element 0/i));
    await user.type(screen.getByLabelText(/bbox h for element 0/i), '6');

    await user.click(screen.getByRole('button', { name: /save changes/i }));

    await waitFor(() => {
      expect(apiMock.modifyAdminAnnotationElement).toHaveBeenCalledWith('analysis-1', 0, {
        class_name: 'new-atl',
        bbox: [1, 2, 5, 6],
        approve_after_save: undefined,
      });
    });
    expect(await screen.findByRole('heading', { name: /element #0 · new-atl/i })).toBeInTheDocument();
    expect(screen.getAllByLabelText(/Pending review status/i)).toHaveLength(2);
  });

  it('explains inspector action consequences and preserves Save & approve as one modify intent', async () => {
    const user = userEvent.setup();
    const initial = queueWithStatuses('pending', 'rejected');
    const updated = queueWithStatuses('approved', 'rejected');
    apiMock.getAdminAnnotationQueue.mockResolvedValueOnce(initial).mockResolvedValueOnce(updated);
    apiMock.modifyAdminAnnotationElement.mockResolvedValueOnce({
      status: 'ok',
      local_only: true,
      warning: 'local only',
      element: updated.analyses[0].elements[0],
    });

    render(<AdminAnnotationsPage />);

    expect(await screen.findByText(/marks this element approved and eligible only when crop and fingerprint checks are valid/i)).toBeInTheDocument();
    expect(screen.getByText(/excludes this element from approved-only training/i)).toBeInTheDocument();
    expect(screen.getByText(/changes class or bbox, regenerates the crop/i)).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /edit element 0/i }));
    expect(screen.getByText(/saving rewrites canonical metadata and regenerates the crop/i)).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /save & approve/i }));

    await waitFor(() => {
      expect(apiMock.modifyAdminAnnotationElement).toHaveBeenCalledWith('analysis-1', 0, {
        class_name: 'atl',
        bbox: [0, 1, 2, 3],
        approve_after_save: true,
      });
    });
  });

  it('preserves visible status and shows an error when a review mutation fails', async () => {
    const user = userEvent.setup();
    apiMock.getAdminAnnotationQueue.mockResolvedValueOnce(queueWithStatuses('pending', 'pending'));
    apiMock.setAdminAnnotationReviewStatus.mockRejectedValueOnce(new Error('offline'));

    render(<AdminAnnotationsPage />);
    await screen.findByRole('button', { name: /reject element 0/i });

    await user.click(screen.getByRole('button', { name: /reject element 0/i }));

    expect(await screen.findByText(/could not mark element 0 as rejected/i)).toBeInTheDocument();
    expect(screen.getAllByLabelText(/Pending review status/i)).toHaveLength(3);
    expect(apiMock.getAdminAnnotationQueue).toHaveBeenCalledTimes(1);
  });
});
