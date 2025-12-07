/**
 * DWIGHT Tile Rendering Module
 * @module renderer/tiles
 */

import { TILE, COLORS } from '../config.js';
import { gameState } from '../state.js';
import { getContext, tileSize } from './canvas.js';

/**
 * Draw a tile (simple version without gameState dependency)
 * @param {number} row - Grid row
 * @param {number} col - Grid column
 * @param {number} tile - Tile type
 * @param {number} [shakeX=0] - X shake offset
 * @param {number} [shakeY=0] - Y shake offset
 */
export function drawTileSimple(row, col, tile, shakeX = 0, shakeY = 0) {
    const ctx = getContext();
    if (!ctx) return;

    const x = col * tileSize + shakeX;
    const y = row * tileSize + shakeY;

    switch (tile) {
        case TILE.FLOOR:
            ctx.fillStyle = (row + col) % 2 === 0 ? COLORS.FLOOR : COLORS.FLOOR_ALT;
            ctx.fillRect(x, y, tileSize, tileSize);
            ctx.fillStyle = 'rgba(0,0,0,0.08)';
            ctx.fillRect(x + 1, y + 1, tileSize - 2, tileSize - 2);
            break;

        case TILE.WALL:
            ctx.fillStyle = COLORS.WALL;
            ctx.fillRect(x, y, tileSize, tileSize);
            ctx.fillStyle = COLORS.WALL_HIGHLIGHT;
            ctx.fillRect(x, y, tileSize, 2);
            ctx.fillRect(x, y, 2, tileSize);
            ctx.fillStyle = 'rgba(255,255,255,0.05)';
            ctx.fillRect(x, y, tileSize, 3);
            break;

        case TILE.CORRIDOR:
            ctx.fillStyle = COLORS.CORRIDOR;
            ctx.fillRect(x, y, tileSize, tileSize);
            ctx.fillStyle = 'rgba(255,255,255,0.05)';
            ctx.fillRect(x, y, tileSize, 2);
            break;

        case TILE.CARPET:
            ctx.fillStyle = COLORS.CARPET;
            ctx.fillRect(x, y, tileSize, tileSize);
            ctx.fillStyle = 'rgba(0,0,0,0.12)';
            ctx.fillRect(x + 1, y + 1, tileSize - 2, tileSize - 2);
            break;

        case TILE.EXIT:
            ctx.fillStyle = COLORS.EXIT;
            ctx.fillRect(x, y, tileSize, tileSize);
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1;
            ctx.strokeRect(x + 1, y + 1, tileSize - 2, tileSize - 2);
            ctx.fillStyle = 'rgba(23, 221, 98, 0.3)';
            ctx.fillRect(x - 1, y - 1, tileSize + 2, tileSize + 2);
            break;

        case TILE.DOOR:
            ctx.fillStyle = COLORS.DOOR;
            ctx.fillRect(x, y, tileSize, tileSize);
            break;

        default:
            ctx.fillStyle = COLORS.FLOOR;
            ctx.fillRect(x, y, tileSize, tileSize);
    }
}

/**
 * Draw a tile with full effects
 * @param {number} row - Grid row
 * @param {number} col - Grid column
 * @param {number} tile - Tile type
 */
export function drawTile(row, col, tile) {
    const ctx = getContext();
    if (!ctx) return;

    const x = col * tileSize + (gameState.shake?.x || 0);
    const y = row * tileSize + (gameState.shake?.y || 0);

    ctx.save();

    switch (tile) {
        case TILE.FLOOR:
            ctx.fillStyle = (row + col) % 2 === 0 ? COLORS.FLOOR : COLORS.FLOOR_ALT;
            ctx.fillRect(x, y, tileSize, tileSize);
            break;

        case TILE.WALL:
            ctx.fillStyle = COLORS.WALL;
            ctx.fillRect(x, y, tileSize, tileSize);
            ctx.fillStyle = COLORS.WALL_HIGHLIGHT;
            ctx.fillRect(x, y, tileSize, 2);
            ctx.fillRect(x, y, 2, tileSize);
            ctx.fillStyle = 'rgba(0,0,0,0.1)';
            for (let i = 0; i < 3; i++) {
                const tx = x + Math.random() * tileSize;
                const ty = y + Math.random() * tileSize;
                ctx.fillRect(tx, ty, 2, 2);
            }
            break;

        case TILE.CORRIDOR:
            ctx.fillStyle = COLORS.CORRIDOR;
            ctx.fillRect(x, y, tileSize, tileSize);
            ctx.fillStyle = 'rgba(0,0,0,0.05)';
            if ((row + col) % 3 === 0) {
                ctx.fillRect(x + 2, y + 2, tileSize - 4, tileSize - 4);
            }
            break;

        case TILE.CARPET:
            ctx.fillStyle = COLORS.CARPET;
            ctx.fillRect(x, y, tileSize, tileSize);
            ctx.fillStyle = 'rgba(0,0,0,0.1)';
            for (let i = 0; i < tileSize; i += 3) {
                ctx.fillRect(x + i, y, 1, tileSize);
            }
            break;

        case TILE.EXIT:
            const glow = Math.sin(gameState.time * 4) * 0.3 + 0.7;
            ctx.fillStyle = COLORS.EXIT;
            ctx.fillRect(x, y, tileSize, tileSize);
            ctx.shadowColor = COLORS.EXIT;
            ctx.shadowBlur = 10 * glow;
            ctx.fillRect(x + 2, y + 2, tileSize - 4, tileSize - 4);
            ctx.shadowBlur = 0;
            ctx.fillStyle = 'white';
            ctx.beginPath();
            ctx.moveTo(x + tileSize / 2, y + 2);
            ctx.lineTo(x + tileSize - 2, y + tileSize / 2);
            ctx.lineTo(x + tileSize / 2, y + tileSize - 2);
            ctx.closePath();
            ctx.fill();
            break;

        case TILE.DOOR:
            ctx.fillStyle = COLORS.DOOR;
            ctx.fillRect(x, y, tileSize, tileSize);
            ctx.fillStyle = 'rgba(0,0,0,0.3)';
            ctx.fillRect(x, y, 2, tileSize);
            ctx.fillRect(x + tileSize - 2, y, 2, tileSize);
            break;
    }

    ctx.restore();
}

/**
 * Draw smoke overlay on a tile
 * @param {number} row - Grid row
 * @param {number} col - Grid column
 * @param {number} level - Smoke density (0-1+)
 */
export function drawSmoke(row, col, level) {
    if (level < 0.1) return;

    const ctx = getContext();
    if (!ctx) return;

    const x = col * tileSize + (gameState.shake?.x || 0);
    const y = row * tileSize + (gameState.shake?.y || 0);

    ctx.save();
    ctx.fillStyle = `rgba(70, 70, 80, ${Math.min(level * 0.6, 0.8)})`;
    ctx.fillRect(x, y, tileSize, tileSize);
    ctx.restore();
}
