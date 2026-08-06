// F1-v0 archetype thumbnails behind a dev flag.
// Enable with `localStorage.setItem('clinic.showArchetypes', '1')` or
// `VITE_SHOW_ARCHETYPES=1`. Deliberately off by default until validated.
const STORAGE_KEY = "clinic.showArchetypes";

export function isArchetypeGalleryEnabled(): boolean {
  if (import.meta.env?.VITE_SHOW_ARCHETYPES === "1") {
    return true;
  }
  try {
    return window.localStorage.getItem(STORAGE_KEY) === "1";
  } catch {
    return false;
  }
}
