/**
 * DWIGHT Game State Module
 * @module state
 */

import { CONFIG } from './config.js';

/**
 * @typedef {Object} Stats
 * @property {number} escaped - Number of people who escaped
 * @property {number} deaths - Number of deaths
 * @property {number} total - Total number of people
 * @property {boolean} alarmActive - Whether alarm is active
 */

/**
 * @typedef {Object} ShakeState
 * @property {number} x - X offset
 * @property {number} y - Y offset
 * @property {number} intensity - Shake intensity
 */

/**
 * @typedef {Object} Person
 * @property {number} id - Unique identifier
 * @property {number} row - Grid row
 * @property {number} col - Grid column
 * @property {string} state - Current state
 * @property {boolean} isWarden - Whether this is a warden
 * @property {number} health - Health (0-100)
 * @property {boolean} alive - Whether alive
 * @property {boolean} escaped - Whether escaped
 */

/**
 * @typedef {Object} Fire
 * @property {number} row - Grid row
 * @property {number} col - Grid column
 * @property {number} intensity - Fire intensity
 * @property {number} age - Fire age in seconds
 */

/**
 * @typedef {Object} Sensor
 * @property {number} id - Unique identifier
 * @property {number} row - Grid row
 * @property {number} col - Grid column
 * @property {string} type - Sensor type
 * @property {number} value - Current reading
 * @property {boolean} triggered - Whether triggered
 * @property {number} health - Health (0-100)
 */

/**
 * Global game state object
 * @type {Object}
 */
export const gameState = {
    /** @type {number[][]} */
    maze: [],

    /** @type {Array<[number, number]>} */
    exits: [],

    /** @type {Person[]} */
    people: [],

    /** @type {Sensor[]} */
    sensors: [],

    /** @type {Fire[]} */
    fires: [],

    /** @type {Object[]} */
    bombs: [],

    /** @type {Object[]} */
    floods: [],

    /** @type {Object[]} */
    debris: [],

    /** @type {Object[]} */
    furniture: [],

    /** @type {boolean} */
    earthquakeActive: false,

    /** @type {number} */
    earthquakeTimer: 0,

    /** @type {{row: number, col: number}|null} */
    earthquakeEpicenter: null,

    /** @type {Object<string, number>} */
    smoke: {},

    /** @type {Array<{row: number, col: number, prob: number}>} */
    predictions: [],

    /** @type {{arrows: Object[], log: string[]}} */
    rlOverlay: {
        arrows: [],
        log: []
    },

    // Display toggles
    /** @type {boolean} */
    showMesh: true,

    /** @type {boolean} */
    showRLOverlay: true,

    /** @type {boolean} */
    showPredictionBeams: true,

    /** @type {boolean} */
    showSensorZones: true,

    /** @type {boolean} */
    showPredictions: true,

    /** @type {boolean} */
    showPheromones: true,

    /** @type {boolean} */
    showSensors: true,

    /** @type {boolean} */
    showDetailInset: true,

    /** @type {Stats} */
    stats: {
        escaped: 0,
        deaths: 0,
        total: CONFIG.TOTAL_PEOPLE,
        alarmActive: false
    },

    /** @type {{confidence: number, predictions: Object[], safePheromone: number, dangerPheromone: number}} */
    neural: {
        confidence: 0,
        predictions: [],
        safePheromone: 0.1,
        dangerPheromone: 0
    },

    /** @type {{decisions: number, avgReward: number, epsilon: number}} */
    rl: {
        decisions: 0,
        avgReward: 0,
        epsilon: 0.2
    },

    // Simulation state
    /** @type {boolean} */
    paused: false,

    /** @type {number} */
    speed: 1.0,

    /** @type {number} */
    time: 0,

    /** @type {number|null} */
    startTime: null,

    // Selected tool
    /** @type {string} */
    selectedTool: 'fire',

    /** @type {number|null} */
    selectedPersonId: null,

    /** @type {boolean} */
    povDimEnabled: false,

    /** @type {ShakeState} */
    shake: { x: 0, y: 0, intensity: 0 },

    /** @type {number} */
    zoom: 1.0,

    /** @type {{loading: boolean, error: string, stats: Object|null, lastRun: Date|null}} */
    pythonTelemetry: {
        loading: false,
        error: '',
        stats: null,
        lastRun: null
    }
};

/**
 * Reset game state to initial values
 */
export function resetGameState() {
    gameState.maze = [];
    gameState.exits = [];
    gameState.people = [];
    gameState.sensors = [];
    gameState.fires = [];
    gameState.bombs = [];
    gameState.floods = [];
    gameState.debris = [];
    gameState.furniture = [];
    gameState.earthquakeActive = false;
    gameState.earthquakeTimer = 0;
    gameState.earthquakeEpicenter = null;
    gameState.smoke = {};
    gameState.predictions = [];
    gameState.rlOverlay = { arrows: [], log: [] };
    gameState.stats = {
        escaped: 0,
        deaths: 0,
        total: CONFIG.TOTAL_PEOPLE,
        alarmActive: false
    };
    gameState.neural = {
        confidence: 0,
        predictions: [],
        safePheromone: 0.1,
        dangerPheromone: 0
    };
    gameState.rl = {
        decisions: 0,
        avgReward: 0,
        epsilon: 0.2
    };
    gameState.paused = false;
    gameState.time = 0;
    gameState.startTime = null;
    gameState.selectedPersonId = null;
    gameState.shake = { x: 0, y: 0, intensity: 0 };
}
