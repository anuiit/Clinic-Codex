import axios, { type AxiosRequestConfig } from 'axios';
import type {
  SegmentResult,
  ClassifyResult,
  ClassesResult,
  SimilarResult,
  TrustResult,
  SaveAnnotationPayload,
  SaveAnnotationResponse,
  SaveAnnotationResult,
  SaveAnnotationErrorCode,
} from '../types';

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:7117';

export interface ApiRequestOptions {
  signal?: AbortSignal;
}

function apiUrl(path: string): string {
  return `${BASE_URL}${path}`;
}

function requestConfig(options?: ApiRequestOptions): AxiosRequestConfig | undefined {
  return options?.signal ? { signal: options.signal } : undefined;
}

async function postData<T>(path: string, payload: unknown, options?: ApiRequestOptions): Promise<T> {
  const config = requestConfig(options);
  const { data } = config
    ? await axios.post<T>(apiUrl(path), payload, config)
    : await axios.post<T>(apiUrl(path), payload);
  return data;
}

async function getData<T>(path: string, options?: ApiRequestOptions): Promise<T> {
  const config = requestConfig(options);
  const { data } = config
    ? await axios.get<T>(apiUrl(path), config)
    : await axios.get<T>(apiUrl(path));
  return data;
}

function imageForm(file: File): FormData {
  const form = new FormData();
  form.append('image', file);
  return form;
}

function dataUrlPayload(imageDataUrl: string): string | undefined {
  return imageDataUrl.split(',')[1];
}

export async function segmentGlyph(file: File, options?: ApiRequestOptions): Promise<SegmentResult> {
  return postData<SegmentResult>('/segment', imageForm(file), options);
}

export async function classifyElement(file: File, options?: ApiRequestOptions): Promise<ClassifyResult> {
  return postData<ClassifyResult>('/classify', imageForm(file), options);
}

export async function getClasses(options?: ApiRequestOptions): Promise<ClassesResult> {
  return getData<ClassesResult>('/classes', options);
}

export async function getSimilar(
  imageDataUrl: string,
  bbox: [number, number, number, number],
  limit = 5,
  options?: ApiRequestOptions,
): Promise<SimilarResult> {
  return postData<SimilarResult>(
    '/similar',
    {
      image_base64: dataUrlPayload(imageDataUrl),
      bbox,
      limit,
      mode: 'prototype',
    },
    options,
  );
}

export async function getTrust(
  imageDataUrl: string,
  bbox: [number, number, number, number],
  predictedClass: string,
  topK = 10,
  options?: ApiRequestOptions,
): Promise<TrustResult> {
  return postData<TrustResult>(
    '/trust',
    {
      image_base64: dataUrlPayload(imageDataUrl),
      bbox,
      predicted_class: predictedClass,
      top_k: topK,
    },
    options,
  );
}

export async function saveAnnotation(
  payload: SaveAnnotationPayload,
  options?: ApiRequestOptions,
): Promise<SaveAnnotationResult> {
  const requestInit: RequestInit = {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  };
  if (options?.signal) {
    requestInit.signal = options.signal;
  }

  try {
    const res = await fetch(apiUrl('/save-annotation'), requestInit);

    if (!res.ok) {
      const errorData = await res.json().catch(() => null) as Partial<{
        error_code: SaveAnnotationErrorCode;
        message: string;
        hint?: string | null;
        trace_id?: string;
      }> | null;

      return {
        ok: false,
        error_code: errorData?.error_code ?? 'NETWORK_ERROR',
        message: errorData?.message ?? `save-annotation failed: ${res.status}`,
        hint: errorData?.hint ?? undefined,
        trace_id: errorData?.trace_id,
      };
    }

    const data = await res.json() as SaveAnnotationResponse;
    return { ok: true, ...data };
  } catch {
    return {
      ok: false,
      error_code: 'NETWORK_ERROR',
      message: 'Network error while saving annotation',
    };
  }
}
