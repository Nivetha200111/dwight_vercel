/**
 * DWIGHT Entity Rendering Module
 * @module renderer/entities
 */

import { COLORS, STATE, TILE } from '../config.js';
import { gameState } from '../state.js';
import { getContext, getPOVContext, tileSize } from './canvas.js';
import { isFlooded, isDebris, hasBomb } from '../utils.js';

/**
 * Draw a person on the canvas
 * @param {Object} person - Person object
 */
export function drawPerson(person) {
    if (!person.alive || person.escaped) return;

    const ctx = getContext();
    if (!ctx) return;

    const x = person.col * tileSize + tileSize / 2 + (gameState.shake?.x || 0);
    const y = person.row * tileSize + tileSize / 2 + (gameState.shake?.y || 0);
    const size = tileSize * 0.6;

    ctx.save();

    // Shadow
    ctx.fillStyle = 'rgba(0,0,0,0.3)';
    ctx.beginPath();
    ctx.ellipse(x, y + size / 2 + 2, size / 2, size / 4, 0, 0, Math.PI * 2);
    ctx.fill();

    // Body color based on state
    let bodyColor = COLORS.NORMAL;
    let outlineColor = '#000';

    switch (person.state) {
        case STATE.WARDEN:
            bodyColor = COLORS.WARDEN;
            outlineColor = '#8b6914';
            break;
        case STATE.PANICKING:
            bodyColor = COLORS.PANICKING;
            break;
        case STATE.EVACUATING:
            bodyColor = COLORS.EVACUATING;
            break;
        case STATE.AWARE:
            bodyColor = COLORS.AWARE;
            break;
        case STATE.HEADPHONES:
            bodyColor = COLORS.HEADPHONES;
            break;
    }

    // Body
    ctx.fillStyle = outlineColor;
    ctx.fillRect(x - size / 2 - 1, y - size / 2 - 1, size + 2, size * 0.6 + 2);
    ctx.fillStyle = bodyColor;
    ctx.fillRect(x - size / 2, y - size / 2, size, size * 0.6);

    // Head
    ctx.fillStyle = '#e6c8a0';
    ctx.fillRect(x - size / 3, y - size / 2 - size * 0.4, size * 0.66, size * 0.4);

    // Warden hat
    if (person.isWarden) {
        ctx.fillStyle = COLORS.WARDEN;
        ctx.fillRect(x - size / 2, y - size / 2 - size * 0.5, size, size * 0.15);
    }

    // Health bar if damaged
    if (person.health < 90) {
        const barWidth = size;
        const barHeight = 3;
        const healthPercent = person.health / 100;

        ctx.fillStyle = '#600';
        ctx.fillRect(x - barWidth / 2, y - size / 2 - size * 0.7, barWidth, barHeight);
        ctx.fillStyle = '#0c0';
        ctx.fillRect(x - barWidth / 2, y - size / 2 - size * 0.7, barWidth * healthPercent, barHeight);
    }

    ctx.restore();
}

/**
 * Draw a sensor on the canvas
 * @param {Object} sensor - Sensor object
 */
export function drawSensor(sensor) {
    if (sensor.health <= 0) return;

    const ctx = getContext();
    if (!ctx) return;

    const x = sensor.col * tileSize + tileSize / 2 + (gameState.shake?.x || 0);
    const y = sensor.row * tileSize + tileSize / 2 + (gameState.shake?.y || 0);
    const radius = tileSize * 0.25;

    ctx.save();

    let color;
    switch (sensor.type) {
        case 'temperature': color = COLORS.TEMP_SENSOR; break;
        case 'smoke': color = COLORS.SMOKE_SENSOR; break;
        case 'co': color = COLORS.CO_SENSOR; break;
        case 'motion': color = COLORS.MOTION_SENSOR; break;
        default: color = '#888';
    }

    // Sensor body
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();

    // White outline
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Triggered animation
    if (sensor.triggered) {
        const wave = (gameState.time * 20) % 15 + 5;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.globalAlpha = 1 - wave / 20;
        ctx.beginPath();
        ctx.arc(x, y, wave, 0, Math.PI * 2);
        ctx.stroke();
    }

    ctx.restore();
}

/**
 * Draw sensor coverage zones
 */
export function drawSensorZones() {
    if (!gameState.showSensorZones) return;

    const ctx = getContext();
    if (!ctx) return;

    ctx.save();
    const shakeX = gameState.shake?.x || 0;
    const shakeY = gameState.shake?.y || 0;

    gameState.sensors.forEach(sensor => {
        if (sensor.health <= 0) return;

        const radiusTiles = sensor.type === 'motion' ? 5 : 4;
        const alpha = sensor.triggered ? 0.25 : 0.08;

        ctx.strokeStyle = sensor.triggered ? '#ff8c00' : '#00bcd4';
        ctx.globalAlpha = alpha;

        const rpx = radiusTiles * tileSize;
        ctx.beginPath();
        ctx.arc(
            sensor.col * tileSize + tileSize / 2 + shakeX,
            sensor.row * tileSize + tileSize / 2 + shakeY,
            rpx,
            0,
            Math.PI * 2
        );
        ctx.stroke();
    });

    ctx.restore();
}

/**
 * Render POV (first-person view) for selected person
 * @param {Object} person - Selected person
 */
export function renderPOV(person) {
    const povCtx = getPOVContext();
    if (!povCtx) return;

    const povCanvas = document.getElementById('pov-canvas');
    if (!povCanvas) return;

    povCtx.clearRect(0, 0, povCanvas.width, povCanvas.height);

    const radius = 4;
    const povTile = Math.floor(povCanvas.width / (radius * 2 + 1));

    for (let dr = -radius; dr <= radius; dr++) {
        for (let dc = -radius; dc <= radius; dc++) {
            const nr = person.row + dr;
            const nc = person.col + dc;
            const px = (dc + radius) * povTile;
            const py = (dr + radius) * povTile;

            let color = '#444';
            const tile = gameState.maze[nr]?.[nc];

            if (tile === TILE.WALL) color = COLORS.WALL;
            else if (tile === TILE.EXIT) color = COLORS.EXIT;
            else if (tile === TILE.CORRIDOR) color = COLORS.CORRIDOR;
            else if (tile === TILE.CARPET) color = COLORS.CARPET;
            else color = COLORS.FLOOR;

            povCtx.fillStyle = color;
            povCtx.fillRect(px, py, povTile, povTile);

            // Hazards overlay
            if (gameState.fires.some(f => f.row === nr && f.col === nc)) {
                povCtx.fillStyle = COLORS.FIRE_OUTER;
                povCtx.fillRect(px + 3, py + 3, povTile - 6, povTile - 6);
            }
            if (isFlooded(nr, nc)) {
                povCtx.fillStyle = COLORS.WATER;
                povCtx.globalAlpha = 0.6;
                povCtx.fillRect(px, py, povTile, povTile);
                povCtx.globalAlpha = 1;
            }
            if (isDebris(nr, nc)) {
                povCtx.fillStyle = '#4a4f59';
                povCtx.fillRect(px + 2, py + 2, povTile - 4, povTile - 4);
            }
            if (hasBomb(nr, nc)) {
                povCtx.fillStyle = COLORS.BOMB_GLOW;
                povCtx.fillRect(px + 4, py + 4, povTile - 8, povTile - 8);
            }

            const smoke = gameState.smoke[`${nr},${nc}`] || 0;
            if (smoke > 0.4) {
                povCtx.fillStyle = `rgba(70, 70, 80, ${Math.min(smoke, 0.8)})`;
                povCtx.fillRect(px, py, povTile, povTile);
            }

            // Other people
            const other = gameState.people.find(p =>
                p.row === nr && p.col === nc && p.alive && !p.escaped
            );
            if (other) {
                povCtx.fillStyle = other.isWarden ? COLORS.WARDEN : COLORS.NORMAL;
                povCtx.fillRect(px + 4, py + 4, povTile - 8, povTile - 8);
            }
        }
    }

    // Center marker for POV person
    povCtx.strokeStyle = '#fff';
    povCtx.lineWidth = 2;
    povCtx.strokeRect(radius * povTile + 2, radius * povTile + 2, povTile - 4, povTile - 4);
}
