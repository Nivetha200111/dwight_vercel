/**
 * DWIGHT Hazards Rendering Module
 * @module renderer/hazards
 */

import { COLORS } from '../config.js';
import { gameState } from '../state.js';
import { getContext, tileSize } from './canvas.js';

/**
 * Draw fire at a position
 * @param {number} row - Grid row
 * @param {number} col - Grid column
 * @param {number} [intensity=1] - Fire intensity
 */
export function drawFire(row, col, intensity = 1) {
    const ctx = getContext();
    if (!ctx) return;

    const x = col * tileSize + (gameState.shake?.x || 0);
    const y = row * tileSize + (gameState.shake?.y || 0);
    const time = gameState.time;

    ctx.save();

    // Base
    ctx.fillStyle = '#3d1a0a';
    ctx.fillRect(x, y, tileSize, tileSize);

    // Light halo
    const radial = ctx.createRadialGradient(
        x + tileSize / 2, y + tileSize / 2, 2,
        x + tileSize / 2, y + tileSize / 2, tileSize * 2
    );
    radial.addColorStop(0, 'rgba(255,100,20,0.25)');
    radial.addColorStop(1, 'rgba(255,100,20,0)');
    ctx.fillStyle = radial;
    ctx.fillRect(x - tileSize, y - tileSize, tileSize * 3, tileSize * 3);

    // Animated flames
    for (let i = 0; i < 3; i++) {
        const fx = x + 2 + i * (tileSize / 3);
        const fh = tileSize * 0.6 + Math.sin(time * 12 + i + col * 0.5) * (tileSize * 0.3);

        const gradient = ctx.createLinearGradient(fx, y + tileSize, fx, y + tileSize - fh);
        gradient.addColorStop(0, COLORS.FIRE_OUTER);
        gradient.addColorStop(0.4, COLORS.FIRE_MID);
        gradient.addColorStop(1, COLORS.FIRE_CORE);

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.moveTo(fx, y + tileSize);
        ctx.lineTo(fx + tileSize / 4, y + tileSize);
        ctx.lineTo(fx + tileSize / 8, y + tileSize - fh);
        ctx.closePath();
        ctx.fill();
    }

    // Glow
    ctx.shadowColor = COLORS.FIRE_OUTER;
    ctx.shadowBlur = 15 * intensity;
    ctx.fillStyle = 'transparent';
    ctx.fillRect(x, y, tileSize, tileSize);

    ctx.restore();
}

/**
 * Draw a bomb
 * @param {Object} bomb - Bomb object
 */
export function drawBomb(bomb) {
    const ctx = getContext();
    if (!ctx) return;

    const x = bomb.col * tileSize + (gameState.shake?.x || 0);
    const y = bomb.row * tileSize + (gameState.shake?.y || 0);
    const pulse = Math.sin(gameState.time * 8 + (bomb.timer || 0)) * 0.5 + 0.5;

    ctx.save();

    ctx.fillStyle = '#1c1c1c';
    ctx.fillRect(x + 3, y + 3, tileSize - 6, tileSize - 6);
    ctx.fillStyle = `rgba(255, 207, 93, ${0.3 + 0.5 * pulse})`;
    ctx.fillRect(x + 4, y + 4, tileSize - 8, tileSize - 8);
    ctx.strokeStyle = COLORS.BOMB_GLOW;
    ctx.lineWidth = 2;
    ctx.strokeRect(x + 2, y + 2, tileSize - 4, tileSize - 4);

    // Fuse
    const fuseHeight = Math.max(4, tileSize * 0.6 * Math.max(0.1, (bomb.timer || 0) / 3));
    ctx.fillStyle = COLORS.BOMB_GLOW;
    ctx.fillRect(x + tileSize / 2 - 1, y - fuseHeight * 0.4, 2, fuseHeight * 0.6);
    ctx.beginPath();
    ctx.arc(x + tileSize / 2, y - fuseHeight * 0.4, 3, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();
}

/**
 * Draw flood water
 * @param {Object} flood - Flood object
 */
export function drawFlood(flood) {
    const ctx = getContext();
    if (!ctx) return;

    const x = flood.col * tileSize + (gameState.shake?.x || 0);
    const y = flood.row * tileSize + (gameState.shake?.y || 0);
    const depth = Math.min(1.2, flood.depth || 0.6);
    const wave = Math.sin(gameState.time * 5 + flood.col + flood.row) * 0.1;

    ctx.save();
    ctx.fillStyle = `rgba(43, 132, 255, ${0.35 + depth * 0.25})`;
    ctx.fillRect(x, y + tileSize * (0.15 - wave), tileSize, tileSize);
    ctx.fillStyle = `rgba(255, 255, 255, ${0.08 + depth * 0.05})`;
    ctx.fillRect(x, y + tileSize * 0.6, tileSize, tileSize * 0.4);
    ctx.strokeStyle = COLORS.WATER_DARK;
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, tileSize, tileSize);
    ctx.restore();
}

/**
 * Draw debris
 * @param {Object} debris - Debris object
 */
export function drawDebris(debris) {
    const ctx = getContext();
    if (!ctx) return;

    const x = debris.col * tileSize + (gameState.shake?.x || 0);
    const y = debris.row * tileSize + (gameState.shake?.y || 0);

    ctx.save();
    ctx.fillStyle = '#4a4f59';
    ctx.fillRect(x, y, tileSize, tileSize);
    ctx.fillStyle = '#6b707c';
    for (let i = 0; i < 3; i++) {
        ctx.fillRect(
            x + Math.random() * tileSize * 0.6,
            y + Math.random() * tileSize * 0.6,
            3, 3
        );
    }
    ctx.strokeStyle = '#1d1f24';
    ctx.strokeRect(x, y, tileSize, tileSize);
    ctx.restore();
}

/**
 * Draw furniture item
 * @param {Object} item - Furniture object
 */
export function drawFurniture(item) {
    const ctx = getContext();
    if (!ctx) return;

    const x = item.col * tileSize + (gameState.shake?.x || 0);
    const y = item.row * tileSize + (gameState.shake?.y || 0);
    const top = tileSize * 0.6;

    ctx.save();
    ctx.fillStyle = '#5f4025';
    ctx.fillRect(x + 1, y + tileSize - top, tileSize - 2, top);
    ctx.fillStyle = '#7a5530';
    ctx.fillRect(x + 1, y + tileSize - top - 3, tileSize - 2, 3);
    ctx.strokeStyle = '#1f140d';
    ctx.strokeRect(x + 1, y + tileSize - top, tileSize - 2, top);
    ctx.fillStyle = '#2a1a0f';
    ctx.fillRect(x + 2, y + tileSize - 4, 3, 4);
    ctx.fillRect(x + tileSize - 5, y + tileSize - 4, 3, 4);
    ctx.restore();
}

/**
 * Draw fire particles
 */
export function drawParticles() {
    const ctx = getContext();
    if (!ctx) return;

    ctx.save();

    gameState.fires.forEach(fire => {
        if (Math.random() < 0.3) {
            const x = fire.col * tileSize + tileSize / 2 + (Math.random() - 0.5) * tileSize;
            const y = fire.row * tileSize + (Math.random() * tileSize * 0.5);
            const size = 2 + Math.random() * 3;

            ctx.fillStyle = Math.random() > 0.5 ? COLORS.FIRE_MID : COLORS.FIRE_OUTER;
            ctx.beginPath();
            ctx.arc(
                x + (gameState.shake?.x || 0),
                y + (gameState.shake?.y || 0),
                size, 0, Math.PI * 2
            );
            ctx.fill();
        }
    });

    ctx.restore();
}

/**
 * Draw earthquake effect
 */
export function drawEarthquakeEffect() {
    if (!gameState.earthquakeActive || !gameState.earthquakeEpicenter) return;

    const ctx = getContext();
    if (!ctx) return;

    const shakeX = gameState.shake?.x || 0;
    const shakeY = gameState.shake?.y || 0;
    const centerX = gameState.earthquakeEpicenter.col * tileSize + tileSize / 2 + shakeX;
    const centerY = gameState.earthquakeEpicenter.row * tileSize + tileSize / 2 + shakeY;
    const pulse = Math.sin(gameState.time * 6) * 0.5 + 0.5;
    const reach = Math.max(1, gameState.earthquakeTimer || 0);

    ctx.save();
    ctx.strokeStyle = `rgba(207, 213, 221, ${0.3 + 0.3 * pulse})`;
    ctx.lineWidth = 2;

    for (let i = 1; i <= 3; i++) {
        const radius = tileSize * (i * 2 + reach * 0.4 + pulse);
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
        ctx.stroke();
    }

    ctx.restore();
}
