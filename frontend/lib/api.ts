export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

type JsonBody =
  | Record<string, unknown>
  | Array<unknown>
  | string
  | number
  | boolean
  | null;

function isFormData(body: BodyInit | null | undefined): body is FormData {
  return typeof FormData !== "undefined" && body instanceof FormData;
}

async function parseApiError(response: Response): Promise<string> {
  const fallback = `${response.status} ${response.statusText}`;

  try {
    const payload = (await response.json()) as { detail?: unknown; message?: unknown };
    if (typeof payload?.detail === "string") {
      return payload.detail;
    }
    if (payload?.detail !== undefined) {
      return JSON.stringify(payload.detail);
    }
    if (typeof payload?.message === "string") {
      return payload.message;
    }
    return fallback;
  } catch {
    return fallback;
  }
}

function createHeaders(options: RequestInit): HeadersInit {
  const headers = new Headers(options.headers ?? {});
  const body = options.body;

  if (!isFormData(body) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  return headers;
}

export async function fetchAPI<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: createHeaders(options),
  });

  if (!response.ok) {
    const detail = await parseApiError(response);
    throw new Error(`API request failed: ${detail}`);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

export async function postJSON<TResponse, TBody extends JsonBody>(
  path: string,
  body: TBody,
): Promise<TResponse> {
  return fetchAPI<TResponse>(path, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function uploadFile<TResponse>(
  path: string,
  file: File,
  fieldName = "file",
  additionalFields: Record<string, string> = {},
): Promise<TResponse> {
  const formData = new FormData();
  formData.append(fieldName, file);

  for (const [key, value] of Object.entries(additionalFields)) {
    formData.append(key, value);
  }

  return fetchAPI<TResponse>(path, {
    method: "POST",
    body: formData,
  });
}
