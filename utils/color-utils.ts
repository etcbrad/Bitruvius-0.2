

// utils/color-utils.ts

/**
 * Adjusts the brightness of a hex color.
 * @param hex The hex color string (e.g., "#RRGGBB").
 * @param factor Brightness factor (e.g., 0.8 for 20% darker, 1.2 for 20% lighter).
 * @returns The adjusted hex color string.
 */
export const adjustBrightness = (hex: string, factor: number): string => {
    if (!hex || hex === 'none' || !hex.startsWith('#')) return hex;

    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);

    const newR = Math.min(255, Math.max(0, Math.round(r * factor)));
    const newG = Math.min(255, Math.max(0, Math.round(g * factor)));
    const newB = Math.min(255, Math.max(0, Math.round(b * factor)));

    return `#${[newR, newG, newB].map(c => c.toString(16).padStart(2, '0')).join('')}`;
};