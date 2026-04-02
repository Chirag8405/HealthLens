const API_BASE_URL = "http://localhost:8000";

export async function fetchAPI<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const response = await fetch(API_BASE_URL + path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers ?? {}),
    },
  });

  if (!response.ok) {
    let detail = String(response.status) + " " + response.statusText;

    try {
      const data = await response.json();
      if (data?.detail) {
        detail =
          typeof data.detail === "string"
            ? data.detail
            : JSON.stringify(data.detail);
      }
    } catch {
      // Keep default status detail when response body is not JSON.
    }

    throw new Error("API request failed: " + detail);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}
