/**
 * DWIGHT Renderer Module
 * @module renderer
 */

import { CONFIG, TILE } from '../config.js';
import { gameState } from '../state.js';
import {
    initCanvas,
    resizeCanvas,
    getContext,
    clearCanvas,
    fillBackground,
    tileSize,
} from './canvas.js';
import { drawTileSimple, drawSmoke } from './tiles.js';
import { drawPerson, drawSensor, drawSensorZones, renderPOV } from './entities.js';
import { drawFire, drawBomb, drawFlood, drawDebris, drawFurniture, drawParticles, drawEarthquakeEffect } from './hazards.js';
import { drawGrid, drawPheromones, drawPredictions, drawPredictionCones, drawMeshOverlay, drawRLArrows } from './overlays.js';

export {
    initCanvas,
    resizeCanvas,
    tileSize,
};

/**
 * Main render function - draws entire game state
 */
export function render() {
    const ctx = getContext();
    if (!ctx) return;

    clearCanvas();
    fillBackground('#1a1a1a');

    // Grid overlay
    drawGrid();

    const shakeX = gameState.shake?.x || 0;
    const shakeY = gameState.shake?.y || 0;

    // Draw tiles
    for (let r = 0; r < CONFIG.ROWS; r++) {
        for (let c = 0; c < CONFIG.COLS; c++) {
            const tile = (gameState.maze[r] && gameState.maze[r][c] !== undefined)
                ? gameState.maze[r][c]
                : TILE.FLOOR;
            drawTileSimple(r, c, tile, shakeX, shakeY);
        }
    }

    // Draw pheromones
    drawPheromones();

    // Draw predictions
    drawPredictions();
    drawPredictionCones();

    // Draw hazards
    gameState.floods.forEach(drawFlood);
    gameState.bombs.forEach(drawBomb);
    gameState.debris.forEach(drawDebris);
    gameState.furniture.forEach(drawFurniture);

    // Draw smoke
    Object.entries(gameState.smoke).forEach(([key, level]) => {
        const [r, c] = key.split(',').map(Number);
        drawSmoke(r, c, level);
    });

    // Draw fires
    gameState.fires.forEach(fire => {
        drawFire(fire.row, fire.col, fire.intensity || 1);
    });

    // Draw particles
    drawParticles();

    // Draw earthquake effect
    drawEarthquakeEffect();

    // Draw sensor zones
    drawSensorZones();

    // Draw mesh overlay
    drawMeshOverlay();

    // Draw RL arrows
    drawRLArrows();

    // Draw sensors
    if (gameState.showSensors) {
        gameState.sensors.forEach(drawSensor);
    }

    // Draw people (sorted by Y for depth)
    const sortedPeople = [...gameState.people].sort((a, b) => a.row - b.row);
    sortedPeople.forEach(drawPerson);

    // Render POV for selected person
    if (gameState.selectedPersonId !== null) {
        const selectedPerson = gameState.people.find(
            p => p.id === gameState.selectedPersonId && p.alive && !p.escaped
        );
        if (selectedPerson) {
            renderPOV(selectedPerson);
        }
    }
}
