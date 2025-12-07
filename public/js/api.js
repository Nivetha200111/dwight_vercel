/**
 * DWIGHT API Module
 * @module api
 */

import { CONFIG } from './config.js';
import { gameState } from './state.js';
import { delay } from './utils.js';

/**
 * @typedef {Object} SimulationResponse
 * @property {boolean} success - Whether request succeeded
 * @property {Object} config - Configuration object
 * @property {number[][]} maze - Building layout
 * @property {Array<[number, number]>} exits - Exit positions
 * @property {Object[]} people - People array
 * @property {Object[]} sensors - Sensors array
 * @property {Object} stats - Initial stats
 * @property {Object} neural - Neural network state
 * @property {Object} rl - RL coordinator state
 * @property {string} [error] - Error message if failed
 */

/**
 * @typedef {Object} HeadlessResponse
 * @property {boolean} success - Whether request succeeded
 * @property {Object} stats - Simulation statistics
 * @property {string} [error] - Error message if failed
 */

/** Maximum retry attempts for API calls */
const MAX_RETRIES = 3;

/** Retry delay in milliseconds */
const RETRY_DELAY = 2000;

/**
 * Fetch initial simulation state from backend
 * @param {number} [retries=0] - Current retry count
 * @returns {Promise<SimulationResponse>}
 * @throws {Error} If all retries fail
 */
export async function fetchInitialState(retries = 0) {
    try {
        const response = await fetch(CONFIG.API_URL);

        if (!response.ok) {
            throw new Error(`HTTP error: ${response.status}`);
        }

        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || 'Unknown error from server');
        }

        return data;
    } catch (error) {
        console.error(`API fetch failed (attempt ${retries + 1}):`, error);

        if (retries < MAX_RETRIES) {
            console.log(`Retrying in ${RETRY_DELAY}ms...`);
            await delay(RETRY_DELAY);
            return fetchInitialState(retries + 1);
        }

        throw new Error(`Failed to load simulation after ${MAX_RETRIES} attempts: ${error.message}`);
    }
}

/**
 * Run headless Python simulation
 * @param {number} [steps=240] - Number of simulation steps
 * @param {number} [dt=0.0333] - Time delta per step
 * @returns {Promise<HeadlessResponse>}
 */
export async function runPythonSimulation(steps = 240, dt = 0.0333) {
    // Update telemetry loading state
    gameState.pythonTelemetry.loading = true;
    gameState.pythonTelemetry.error = '';

    try {
        const url = `${CONFIG.PY_SIM_API_URL}?steps=${steps}&dt=${dt}`;
        const response = await fetch(url);

        if (!response.ok) {
            throw new Error(`HTTP error: ${response.status}`);
        }

        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || 'Simulation failed');
        }

        // Update telemetry state
        gameState.pythonTelemetry.stats = data.stats;
        gameState.pythonTelemetry.lastRun = new Date();
        gameState.pythonTelemetry.loading = false;

        return data;
    } catch (error) {
        console.error('Python simulation failed:', error);
        gameState.pythonTelemetry.error = error.message;
        gameState.pythonTelemetry.loading = false;
        throw error;
    }
}

/**
 * Initialize game state from API response
 * @param {SimulationResponse} data - API response data
 */
export function initializeFromResponse(data) {
    gameState.maze = data.maze;
    gameState.exits = data.exits;
    gameState.people = data.people;
    gameState.sensors = data.sensors;

    // Initialize stats
    if (data.stats) {
        gameState.stats = { ...gameState.stats, ...data.stats };
    }

    // Initialize neural state
    if (data.neural) {
        gameState.neural = { ...gameState.neural, ...data.neural };
    }

    // Initialize RL state
    if (data.rl) {
        gameState.rl = { ...gameState.rl, ...data.rl };
    }

    console.log('Game state initialized:', {
        mazeSize: `${data.config?.rows}x${data.config?.cols}`,
        people: gameState.people.length,
        sensors: gameState.sensors.length,
        exits: gameState.exits.length
    });
}
