/**
 * DWIGHT Configuration Module
 * @module config
 */

/**
 * Global configuration constants
 * @constant {Object}
 */
export const CONFIG = {
    ROWS: 45,
    COLS: 70,
    TILE_SIZE: 12,
    MAX_TILE: 18,
    TOTAL_PEOPLE: 60,
    NUM_WARDENS: 4,
    NUM_SENSORS: 25,
    API_URL: '/api/simulate',
    PY_SIM_API_URL: '/api/headless'
};

/**
 * Tile type constants
 * @enum {number}
 */
export const TILE = {
    FLOOR: 0,
    WALL: 1,
    EXIT: 2,
    DOOR: 3,
    CORRIDOR: 4,
    CARPET: 5
};

/**
 * Person state constants
 * @enum {string}
 */
export const STATE = {
    WORKING: 'working',
    HEADPHONES: 'headphones',
    AWARE: 'aware',
    EVACUATING: 'evacuating',
    PANICKING: 'panicking',
    WARDEN: 'warden'
};

/**
 * Minecraft-inspired color palette
 * @constant {Object}
 */
export const COLORS = {
    // Tiles
    FLOOR: '#b4afa5',
    FLOOR_ALT: '#aaa69c',
    WALL: '#2d3037',
    WALL_HIGHLIGHT: '#414550',
    CORRIDOR: '#9b9691',
    CARPET: '#614141',
    EXIT: '#17dd62',
    DOOR: '#785028',

    // Hazards
    FIRE_CORE: '#ffffc8',
    FIRE_MID: '#ffdc50',
    FIRE_OUTER: '#ff6414',
    SMOKE: '#464650',
    BOMB: '#ffcf5d',
    BOMB_GLOW: '#ff8c00',
    WATER: '#2b84ff',
    WATER_DARK: '#1a4fa3',
    EARTHQUAKE: '#cfd5dd',

    // Pheromones
    SAFE_PHEROMONE: 'rgba(0, 255, 136, 0.6)',
    DANGER_PHEROMONE: 'rgba(255, 68, 68, 0.7)',
    PREDICTION: 'rgba(192, 101, 255, 0.55)',

    // People
    NORMAL: '#6496ff',
    AWARE: '#ffff64',
    EVACUATING: '#64ff64',
    PANICKING: '#ff5050',
    WARDEN: '#fcee4b',
    HEADPHONES: '#ff64ff',

    // Sensors
    TEMP_SENSOR: '#ff6600',
    SMOKE_SENSOR: '#999999',
    CO_SENSOR: '#ff00ff',
    MOTION_SENSOR: '#00ccff'
};
