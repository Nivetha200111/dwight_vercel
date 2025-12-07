/**
 * DWIGHT Canvas Setup Module
 * @module renderer/canvas
 */

import { CONFIG } from '../config.js';
import { gameState } from '../state.js';

/** @type {HTMLCanvasElement|null} */
let canvas = null;

/** @type {CanvasRenderingContext2D|null} */
let ctx = null;

/** @type {number} */
let canvasWidth = 0;

/** @type {number} */
let canvasHeight = 0;

/** @type {number} */
export let tileSize = 10;

/**
 * Initialize the main canvas
 */
export function initCanvas() {
    canvas = document.getElementById('game-canvas');
    if (!canvas) {
        console.error('Canvas element not found');
        return;
    }

    ctx = canvas.getContext('2d');
    if (!ctx) {
        console.error('Could not get 2D context');
        return;
    }

    // Disable image smoothing for crisp pixels
    ctx.imageSmoothingEnabled = false;

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
}

/**
 * Resize canvas to fit container
 */
export function resizeCanvas() {
    const container = document.getElementById('canvas-container');
    if (!container || !canvas || !ctx) return;

    const rect = container.getBoundingClientRect();
    const containerWidth = rect.width || 800;
    const containerHeight = rect.height || 600;

    // Calculate tile size to fit the maze
    const maxTileWidth = Math.floor(containerWidth / CONFIG.COLS);
    const maxTileHeight = Math.floor(containerHeight / CONFIG.ROWS);
    const baseTile = Math.max(4, Math.min(maxTileWidth, maxTileHeight, CONFIG.MAX_TILE));
    tileSize = Math.min(baseTile * (gameState.zoom || 1), CONFIG.MAX_TILE * 1.5);

    canvasWidth = CONFIG.COLS * tileSize;
    canvasHeight = CONFIG.ROWS * tileSize;

    canvas.width = canvasWidth;
    canvas.height = canvasHeight;

    // Center canvas in container
    canvas.style.width = canvasWidth + 'px';
    canvas.style.height = canvasHeight + 'px';
    canvas.style.position = 'absolute';
    canvas.style.left = '50%';
    canvas.style.top = '50%';
    canvas.style.transform = 'translate(-50%, -50%)';

    // Re-disable smoothing after resize
    ctx.imageSmoothingEnabled = false;

    console.log('Canvas resized:', canvasWidth, 'x', canvasHeight, 'tileSize:', tileSize);
}

/**
 * Get the canvas element
 * @returns {HTMLCanvasElement|null}
 */
export function getCanvas() {
    return canvas;
}

/**
 * Get the rendering context
 * @returns {CanvasRenderingContext2D|null}
 */
export function getContext() {
    return ctx;
}

/**
 * Get canvas dimensions
 * @returns {{width: number, height: number}}
 */
export function getCanvasDimensions() {
    return { width: canvasWidth, height: canvasHeight };
}

/**
 * Get POV canvas context
 * @returns {CanvasRenderingContext2D|null}
 */
export function getPOVContext() {
    const povCanvas = document.getElementById('pov-canvas');
    if (!povCanvas) return null;

    const povCtx = povCanvas.getContext('2d');
    if (povCtx) {
        povCtx.imageSmoothingEnabled = false;
    }
    return povCtx;
}

/**
 * Clear the canvas
 */
export function clearCanvas() {
    if (!ctx) return;
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);
}

/**
 * Fill canvas with background color
 * @param {string} [color='#1a1a1a'] - Background color
 */
export function fillBackground(color = '#1a1a1a') {
    if (!ctx) return;
    ctx.fillStyle = color;
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);
}
