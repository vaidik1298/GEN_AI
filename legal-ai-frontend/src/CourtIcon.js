import React from "react";

export function CourtIcon({ className = "" }) {
  return (
    <svg
      className={className}
      viewBox="0 0 64 64"
      fill="none"
      aria-hidden="true"
    >
      <ellipse cx="32" cy="56" rx="24" ry="4" fill="#9a8c98" />
      <rect x="28" y="24" width="8" height="24" rx="2" fill="#4a4e69" />
      <rect x="12" y="48" width="40" height="6" rx="2" fill="#c9ada7" />
      <rect x="8" y="54" width="48" height="4" rx="2" fill="#c9ada7" />
      <polygon points="32,8 8,24 56,24" fill="#c9ada7" />
      <rect x="30" y="12" width="4" height="12" rx="2" fill="#4a4e69" />
    </svg>
  );
}