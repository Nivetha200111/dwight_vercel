/**
 * DWIGHT Overlay Rendering Module
 * @module renderer/overlays
 */

import { CONFIG, TILE } from '../config.js';
import { gameState } from '../state.js';
import { getContext, getCanvasDimensions, tileSize } from './canvas.js';
import { inBounds } from '../utils.js';

/**
 * Draw grid lines
 */
export function drawGrid() {
    const ctx = getContext();
    if (!ctx) return;

    const { width, height } = getCanvasDimensions();

    ctx.save();
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.15)';
    ctx.lineWidth = 1;

    const shakeX = gameState.shake?.x || 0;
    const shakeY = gameState.shake?.y || 0;

    for (let r = 0; r <= CONFIG.ROWS; r++) {
        ctx.beginPath();
        ctx.moveTo(shakeX, r * tileSize + shakeY);
        ctx.lineTo(width + shakeX, r * tileSize + shakeY);
        ctx.stroke();
    }

    for (let c = 0; c <= CONFIG.COLS; c++) {
        ctx.beginPath();
        ctx.moveTo(c * tileSize + shakeX, shakeY);
        ctx.lineTo(c * tileSize + shakeX, height + shakeY);
        ctx.stroke();
    }

    ctx.restore();
}

/**
 * Draw pheromone visualization
 */
export function drawPheromones() {
    if (!gameState.showPheromones) return;

    const ctx = getContext();
    if (!ctx) return;

    ctx.save();
    const shakeX = gameState.shake?.x || 0;
    const shakeY = gameState.shake?.y || 0;

    for (let r = 0; r < CONFIG.ROWS; r++) {
        for (let c = 0; c < CONFIG.COLS; c++) {
            const tile = gameState.maze[r]?.[c];
            if (tile === TILE.WALL) continue;

            const distExit = Math.min(
                ...gameState.exits.map(([er, ec]) => Math.abs(er - r) + Math.abs(ec - c)),
                999
            );
            const distFire = gameState.fires.length
                ? Math.min(...gameState.fires.map(f => Math.abs(f.row - r) + Math.abs(f.col - c)))
                : 999;

            const safeVal = 1 / (distExit + 1);
            const dangerVal = distFire < 999 ? 1 / (distFire + 1) : 0;

            if (safeVal > 0.05) {
                ctx.fillStyle = `rgba(0, 255, 136, ${Math.min(0.25, safeVal * 0.6)})`;
                ctx.fillRect(c * tileSize + shakeX, r * tileSize + shakeY, tileSize, tileSize);
            }
            if (dangerVal > 0.05) {
                ctx.fillStyle = `rgba(255, 68, 68, ${Math.min(0.35, dangerVal * 0.9)})`;
                ctx.fillRect(c * tileSize + shakeX, r * tileSize + shakeY, tileSize, tileSize);
            }
        }
    }

    ctx.restore();
}

/**
 * Draw neural network predictions
 */
export function drawPredictions() {
    if (!gameState.showPredictions) return;

    const ctx = getContext();
    if (!ctx) return;

    ctx.save();

    gameState.predictions.forEach(pred => {
        const x = pred.col * tileSize + (gameState.shake?.x || 0);
        const y = pred.row * tileSize + (gameState.shake?.y || 0);
        const pulse = Math.sin(gameState.time * 4) * 0.3 + 0.7;

        ctx.fillStyle = `rgba(192, 101, 255, ${0.3 * pred.prob * pulse})`;
        ctx.fillRect(x, y, tileSize, tileSize);
    });

    ctx.restore();
}

/**
 * Draw prediction cones from fire sources
 */
export function drawPredictionCones() {
    if (!gameState.showPredictionBeams) return;

    const ctx = getContext();
    if (!ctx) return;

    ctx.save();
    const shakeX = gameState.shake?.x || 0;
    const shakeY = gameState.shake?.y || 0;
    const pulse = Math.sin(gameState.time * 4) * 0.2 + 0.3;

    const directions = [[-1, 0], [1, 0], [0, -1], [0, 1]];

    gameState.fires.forEach(fire => {
        directions.forEach(([dr, dc]) => {
            for (let step = 1; step <= 6; step++) {
                const nr = fire.row + dr * step;
                const nc = fire.col + dc * step;
                if (!inBounds(nr, nc)) break;
                if (gameState.maze[nr]?.[nc] === TILE.WALL) break;

                const alpha = Math.max(0.08, 0.2 - step * 0.02) + pulse * 0.05;
                ctx.fillStyle = `rgba(192,101,255, ${alpha})`;
                ctx.fillRect(nc * tileSize + shakeX, nr * tileSize + shakeY, tileSize, tileSize);
            }
        });
    });

    ctx.restore();
}

/**
 * Draw mesh network overlay
 */
export function drawMeshOverlay() {
    if (!gameState.showMesh) return;

    const ctx = getContext();
    if (!ctx) return;

    ctx.save();
    const nodes = gameState.people.filter(p => p.alive && !p.escaped);
    ctx.strokeStyle = 'rgba(0, 200, 255, 0.25)';
    ctx.lineWidth = 1;

    nodes.forEach(a => {
        nodes.forEach(b => {
            if (a.id >= b.id) return;
            const dist = Math.abs(a.row - b.row) + Math.abs(a.col - b.col);
            if (dist <= 10) {
                ctx.beginPath();
                ctx.moveTo(
                    a.col * tileSize + tileSize / 2 + (gameState.shake?.x || 0),
                    a.row * tileSize + tileSize / 2 + (gameState.shake?.y || 0)
                );
                ctx.lineTo(
                    b.col * tileSize + tileSize / 2 + (gameState.shake?.x || 0),
                    b.row * tileSize + tileSize / 2 + (gameState.shake?.y || 0)
                );
                ctx.stroke();
            }
        });
    });

    ctx.restore();
}

/**
 * Draw RL coordinator arrows
 */
export function drawRLArrows() {
    if (!gameState.showRLOverlay) return;

    const ctx = getContext();
    if (!ctx) return;

    ctx.save();
    ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';

    gameState.rlOverlay.arrows.forEach(arrow => {
        const baseX = arrow.col * tileSize + tileSize / 2 + (gameState.shake?.x || 0);
        const baseY = arrow.row * tileSize + tileSize / 2 + (gameState.shake?.y || 0);
        const scale = tileSize * 0.4;
        const dx = arrow.dx * scale;
        const dy = arrow.dy * scale;

        ctx.beginPath();
        ctx.moveTo(baseX + dx, baseY + dy);
        ctx.lineTo(baseX - dy * 0.5, baseY + dx * 0.5);
        ctx.lineTo(baseX + dy * 0.5, baseY - dx * 0.5);
        ctx.closePath();
        ctx.fill();
    });

    ctx.restore();
}
