/**
 * DWIGHT Utility Functions
 * @module utils
 */

import { CONFIG } from './config.js';
import { gameState } from './state.js';

/**
 * Check if position is within bounds
 * @param {number} row - Row index
 * @param {number} col - Column index
 * @returns {boolean}
 */
export function inBounds(row, col) {
    return row > 0 && row < CONFIG.ROWS - 1 && col > 0 && col < CONFIG.COLS - 1;
}

/**
 * Check if position has debris
 * @param {number} row - Row index
 * @param {number} col - Column index
 * @returns {boolean}
 */
export function isDebris(row, col) {
    return gameState.debris.some(d => d.row === row && d.col === col);
}

/**
 * Check if position is flooded
 * @param {number} row - Row index
 * @param {number} col - Column index
 * @returns {boolean}
 */
export function isFlooded(row, col) {
    return gameState.floods.some(f => f.row === row && f.col === col);
}

/**
 * Check if position has a bomb
 * @param {number} row - Row index
 * @param {number} col - Column index
 * @returns {boolean}
 */
export function hasBomb(row, col) {
    return gameState.bombs.some(b => b.row === row && b.col === col);
}

/**
 * Check if position has fire
 * @param {number} row - Row index
 * @param {number} col - Column index
 * @returns {boolean}
 */
export function hasFire(row, col) {
    return gameState.fires.some(f => f.row === row && f.col === col);
}

/**
 * Calculate Manhattan distance between two points
 * @param {number} r1 - Row 1
 * @param {number} c1 - Col 1
 * @param {number} r2 - Row 2
 * @param {number} c2 - Col 2
 * @returns {number}
 */
export function manhattanDistance(r1, c1, r2, c2) {
    return Math.abs(r1 - r2) + Math.abs(c1 - c2);
}

/**
 * Clamp a value between min and max
 * @param {number} value - Value to clamp
 * @param {number} min - Minimum value
 * @param {number} max - Maximum value
 * @returns {number}
 */
export function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

/**
 * Linear interpolation
 * @param {number} a - Start value
 * @param {number} b - End value
 * @param {number} t - Interpolation factor (0-1)
 * @returns {number}
 */
export function lerp(a, b, t) {
    return a + (b - a) * t;
}

/**
 * Format time in MM:SS format
 * @param {number} seconds - Time in seconds
 * @returns {string}
 */
export function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

/**
 * Delay execution
 * @param {number} ms - Milliseconds to wait
 * @returns {Promise<void>}
 */
export function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
