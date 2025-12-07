/**
 * ═══════════════════════════════════════════════════════════════════════════
 * DWIGHT UX - Minecraft-Themed Neural ACO Emergency Evacuation System
 * Frontend JavaScript - Game Logic, Rendering, and Simulation
 * ═══════════════════════════════════════════════════════════════════════════
 */

// ═══════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

const CONFIG = {
    ROWS: 45,
    COLS: 70,
    TILE_SIZE: 12,
    MAX_TILE: 18,
    TOTAL_PEOPLE: 60,
    NUM_WARDENS: 4,
    NUM_SENSORS: 25,
    SPRINKLER_SPACING: 6,
    API_URL: '/api/simulate',
    HEADLESS_URL: '/api/step',
    COMMAND_URL: '/api/command',
    USE_BACKEND: true,
    PY_SIM_API_URL: '/api/headless'
};

async function sendCommand(payload) {
    try {
        await fetch(CONFIG.COMMAND_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
    } catch (err) {
        console.warn('Command failed', err);
    }
}

// Tile types
const TILE = {
    FLOOR: 0,
    WALL: 1,
    EXIT: 2,
    DOOR: 3,
    CORRIDOR: 4,
    CARPET: 5,
    LADDER: 6, // extendable ladders/steps to escape corners
    STROBE: 7  // virtual tile overlay for floor strobes (render-only)
};

// Person states
const STATE = {
    WORKING: 'working',
    HEADPHONES: 'headphones',
    AWARE: 'aware',
    EVACUATING: 'evacuating',
    PANICKING: 'panicking',
    WARDEN: 'warden'
};

// Minecraft color palette
const COLORS = {
    // Tiles
    FLOOR: '#b4afa5',
    FLOOR_ALT: '#aaa69c',
    WALL: '#2d3037',
    WALL_HIGHLIGHT: '#414550',
    CORRIDOR: '#9b9691',
    CARPET: '#614141',
    EXIT: '#17dd62',
    DOOR: '#785028',
    LADDER: '#55ddff',

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
    SAFE_PHEROMONE: 'rgba(0, 255, 136, 0.6)',     // more opaque for emphasis
    DANGER_PHEROMONE: 'rgba(255, 68, 68, 0.7)',   // more opaque for emphasis
    PREDICTION: 'rgba(192, 101, 255, 0.55)',
    STROBE: 'rgba(80, 255, 180, 0.25)',

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

// ═══════════════════════════════════════════════════════════════════════════
// GAME STATE
// ═══════════════════════════════════════════════════════════════════════════

const gameState = {
    maze: [],
    exits: [],
    people: [],
    sensors: [],
    fires: [],
    bombs: [],
    floods: [],
    debris: [],
    sprinklers: [],
    furniture: [],
    earthquakeActive: false,
    earthquakeTimer: 0,
    earthquakeEpicenter: null,
    smoke: {},
    predictions: [],
    rlOverlay: {
        arrows: [],
        log: []
    },
    showMesh: true,
    showRLOverlay: true,
    showPredictionBeams: true,
    showSensorZones: true,
    showStrobes: true,
    showSprinklers: true,

    // Stats
    stats: {
        escaped: 0,
        deaths: 0,
        total: CONFIG.TOTAL_PEOPLE,
        alarmActive: false
    },

    // Neural/RL data
    neural: {
        confidence: 0,
        predictions: [],
        safePheromone: 0.1,
        dangerPheromone: 0
    },
    rl: {
        decisions: 0,
        avgReward: 0,
        epsilon: 0.2
    },

    // Simulation state
    paused: false,
    speed: 1.0,
    time: 0,
    startTime: null,

    // Display toggles
    showPredictions: true,
    showPheromones: true,
    showSensors: true,
    showDetailInset: true,
    showStrobes: true,
    showSprinklers: true,

    // Selected tool
    selectedTool: 'fire',
    selectedPersonId: null,
    povDimEnabled: false,

    // Screen shake
    shake: { x: 0, y: 0, intensity: 0 },

    // Zoom
    zoom: 1.0,

    // Python headless telemetry
    pythonTelemetry: {
        loading: false,
        error: '',
        stats: null,
        lastRun: null
    }
};

// Utility helpers
function inBounds(row, col) {
    return row > 0 && row < CONFIG.ROWS - 1 && col > 0 && col < CONFIG.COLS - 1;
}

function isDebris(row, col) {
    return gameState.debris.some(d => d.row === row && d.col === col);
}

function isFlooded(row, col) {
    return gameState.floods.some(f => f.row === row && f.col === col);
}

function hasBomb(row, col) {
    return gameState.bombs.some(b => b.row === row && b.col === col);
}

// ═══════════════════════════════════════════════════════════════════════════
// CANVAS & RENDERING
// ═══════════════════════════════════════════════════════════════════════════

let canvas, ctx;
let canvasWidth, canvasHeight;
let tileSize = 10; // Default tile size

function initCanvas() {
    canvas = document.getElementById('game-canvas');
    ctx = canvas.getContext('2d');

    // Disable image smoothing for crisp pixels
    ctx.imageSmoothingEnabled = false;

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
}

function resizeCanvas() {
    const container = document.getElementById('canvas-container');
    if (!container) return;

    const rect = container.getBoundingClientRect();

    // Ensure we have valid dimensions
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
    if (ctx) {
        ctx.imageSmoothingEnabled = false;
    }

    console.log('Canvas resized:', canvasWidth, 'x', canvasHeight, 'tileSize:', tileSize);
}

function getPOVContext() {
    const povCanvas = document.getElementById('pov-canvas');
    if (!povCanvas) return null;
    const povCtx = povCanvas.getContext('2d');
    povCtx.imageSmoothingEnabled = false;
    return povCtx;
}

// ═══════════════════════════════════════════════════════════════════════════
// DRAWING FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

// Simple tile drawing that doesn't depend on gameState.shake
function drawTileSimple(row, col, tile, shakeX, shakeY) {
    const x = col * tileSize + (shakeX || 0);
    const y = row * tileSize + (shakeY || 0);

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
            const glow = Math.sin(gameState.time * 4) * 0.3 + 0.7;
            ctx.fillStyle = COLORS.EXIT;
            ctx.fillRect(x, y, tileSize, tileSize);
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1;
            ctx.strokeRect(x + 1, y + 1, tileSize - 2, tileSize - 2);
            ctx.fillStyle = `rgba(23, 221, 98, 0.3)`;
            ctx.fillRect(x - 1, y - 1, tileSize + 2, tileSize + 2);
            break;

        case TILE.DOOR:
            ctx.fillStyle = COLORS.DOOR;
            ctx.fillRect(x, y, tileSize, tileSize);
            break;

        case TILE.LADDER:
            ctx.fillStyle = COLORS.LADDER;
            ctx.fillRect(x, y, tileSize, tileSize);
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(x + tileSize * 0.3, y + tileSize * 0.2);
            ctx.lineTo(x + tileSize * 0.3, y + tileSize * 0.8);
            ctx.moveTo(x + tileSize * 0.7, y + tileSize * 0.2);
            ctx.lineTo(x + tileSize * 0.7, y + tileSize * 0.8);
            ctx.moveTo(x + tileSize * 0.3, y + tileSize * 0.4);
            ctx.lineTo(x + tileSize * 0.7, y + tileSize * 0.4);
            ctx.moveTo(x + tileSize * 0.3, y + tileSize * 0.6);
            ctx.lineTo(x + tileSize * 0.7, y + tileSize * 0.6);
            ctx.stroke();
            break;

        default:
            ctx.fillStyle = COLORS.FLOOR;
            ctx.fillRect(x, y, tileSize, tileSize);
    }
}

function drawTile(row, col, tile) {
    const x = col * tileSize + (gameState.shake?.x || 0);
    const y = row * tileSize + (gameState.shake?.y || 0);

    ctx.save();

    switch (tile) {
        case TILE.FLOOR:
            ctx.fillStyle = (row + col) % 2 === 0 ? COLORS.FLOOR : COLORS.FLOOR_ALT;
            ctx.fillRect(x, y, tileSize, tileSize);
            break;

        case TILE.WALL:
            // 3D wall effect
            ctx.fillStyle = COLORS.WALL;
            ctx.fillRect(x, y, tileSize, tileSize);

            // Highlight
            ctx.fillStyle = COLORS.WALL_HIGHLIGHT;
            ctx.fillRect(x, y, tileSize, 2);
            ctx.fillRect(x, y, 2, tileSize);

            // Add stone texture
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

            // Subtle pattern
            ctx.fillStyle = 'rgba(0,0,0,0.05)';
            if ((row + col) % 3 === 0) {
                ctx.fillRect(x + 2, y + 2, tileSize - 4, tileSize - 4);
            }
            break;

        case TILE.CARPET:
            ctx.fillStyle = COLORS.CARPET;
            ctx.fillRect(x, y, tileSize, tileSize);

            // Carpet texture
            ctx.fillStyle = 'rgba(0,0,0,0.1)';
            for (let i = 0; i < tileSize; i += 3) {
                ctx.fillRect(x + i, y, 1, tileSize);
            }
            break;

        case TILE.EXIT:
            // Glowing exit
            const glow = Math.sin(gameState.time * 4) * 0.3 + 0.7;
            ctx.fillStyle = COLORS.EXIT;
            ctx.fillRect(x, y, tileSize, tileSize);

            // Glow effect
            ctx.shadowColor = COLORS.EXIT;
            ctx.shadowBlur = 10 * glow;
            ctx.fillRect(x + 2, y + 2, tileSize - 4, tileSize - 4);
            ctx.shadowBlur = 0;

            // Arrow
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

            // Door frame
            ctx.fillStyle = 'rgba(0,0,0,0.3)';
            ctx.fillRect(x, y, 2, tileSize);
            ctx.fillRect(x + tileSize - 2, y, 2, tileSize);
            break;
    }

    ctx.restore();
}

function drawFire(row, col, intensity = 1) {
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
    radial.addColorStop(0, `rgba(255,100,20,0.25)`);
    radial.addColorStop(1, `rgba(255,100,20,0)`);
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

function drawSmoke(row, col, level) {
    if (level < 0.1) return;

    const x = col * tileSize + (gameState.shake?.x || 0);
    const y = row * tileSize + (gameState.shake?.y || 0);

    ctx.save();
    ctx.fillStyle = `rgba(70, 70, 80, ${Math.min(level * 0.6, 0.8)})`;
    ctx.fillRect(x, y, tileSize, tileSize);
    ctx.restore();
}

function drawFlood(flood) {
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

function drawDebris(debris) {
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
            3,
            3
        );
    }
    ctx.strokeStyle = '#1d1f24';
    ctx.strokeRect(x, y, tileSize, tileSize);
    ctx.restore();
}

function drawFurniture(item) {
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
    // Legs
    ctx.fillStyle = '#2a1a0f';
    ctx.fillRect(x + 2, y + tileSize - 4, 3, 4);
    ctx.fillRect(x + tileSize - 5, y + tileSize - 4, 3, 4);
    ctx.restore();
}

function drawBomb(bomb) {
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

function drawPerson(person) {
    if (!person.alive || person.escaped) return;

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

function renderPOV(person) {
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
            const other = gameState.people.find(p => p.row === nr && p.col === nc && p.alive && !p.escaped);
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

function drawSensor(sensor) {
    if (sensor.health <= 0) return;

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

function drawSensorZones() {
    if (!gameState.showSensorZones) return;
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

function drawPheromones() {
    if (!gameState.showPheromones) return;

    ctx.save();
    const shakeX = gameState.shake?.x || 0;
    const shakeY = gameState.shake?.y || 0;

    // Compute simple safe/danger maps based on exits and fires
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

function drawPredictions() {
    if (!gameState.showPredictions) return;

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

function drawSprinklers() {
    if (!gameState.showSprinklers || !gameState.sprinklers) return;
    const pulse = Math.sin(gameState.time * 8) * 0.3 + 0.7;
    gameState.sprinklers.forEach(s => {
        const x = s.col * tileSize + tileSize / 2 + (gameState.shake?.x || 0);
        const y = s.row * tileSize + tileSize / 2 + (gameState.shake?.y || 0);
        ctx.save();
        ctx.fillStyle = `rgba(80, 180, 255, ${0.4 + 0.2 * pulse})`;
        ctx.beginPath();
        ctx.arc(x, y, tileSize * 0.25, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = `rgba(140, 210, 255, ${0.6 + 0.2 * pulse})`;
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.restore();

        // Spray arcs if fire is nearby
        const hasFireNearby = gameState.fires.some(f =>
            Math.abs(f.row - s.row) + Math.abs(f.col - s.col) <= 5
        );
        if (hasFireNearby) {
            ctx.save();
            ctx.strokeStyle = `rgba(120, 200, 255, 0.35)`;
            ctx.lineWidth = 2;
            for (let angle = -Math.PI / 3; angle <= Math.PI / 3; angle += Math.PI / 6) {
                ctx.beginPath();
                ctx.moveTo(x, y);
                const len = tileSize * (1.5 + pulse);
                ctx.lineTo(x + Math.cos(angle) * len, y + Math.sin(angle) * len);
                ctx.stroke();
            }
            ctx.restore();
        }
    });
}

function drawStrobes() {
    if (!gameState.showStrobes) return;
    const pulse = Math.sin(gameState.time * 6) * 0.3 + 0.7;
    ctx.fillStyle = `rgba(80, 255, 180, ${0.05 + 0.15 * pulse})`;
    for (let r = 0; r < CONFIG.ROWS; r++) {
        for (let c = 0; c < CONFIG.COLS; c++) {
            if (gameState.maze[r]?.[c] === TILE.CORRIDOR) {
                const x = c * tileSize + (gameState.shake?.x || 0);
                const y = r * tileSize + (gameState.shake?.y || 0);
                ctx.fillRect(x, y, tileSize, tileSize);
                // running light line
                ctx.fillStyle = `rgba(140, 255, 210, ${0.12 + 0.1 * pulse})`;
                ctx.fillRect(x, y + tileSize * 0.45, tileSize, tileSize * 0.1);
                ctx.fillStyle = `rgba(80, 255, 180, ${0.05 + 0.15 * pulse})`;
            }
        }
    }
}

function drawPredictionCones() {
    if (!gameState.showPredictionBeams) return;
    ctx.save();
    const shakeX = gameState.shake?.x || 0;
    const shakeY = gameState.shake?.y || 0;
    const pulse = Math.sin(gameState.time * 4) * 0.2 + 0.3;

    const directions = [
        [-1, 0],
        [1, 0],
        [0, -1],
        [0, 1]
    ];

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

function drawParticles() {
    ctx.save();

    gameState.fires.forEach(fire => {
        // Emit particles occasionally
        if (Math.random() < 0.3) {
            const x = fire.col * tileSize + tileSize / 2 + (Math.random() - 0.5) * tileSize;
            const y = fire.row * tileSize + (Math.random() * tileSize * 0.5);
            const size = 2 + Math.random() * 3;

            ctx.fillStyle = Math.random() > 0.5 ? COLORS.FIRE_MID : COLORS.FIRE_OUTER;
            ctx.beginPath();
            ctx.arc(x + (gameState.shake?.x || 0), y + (gameState.shake?.y || 0), size, 0, Math.PI * 2);
            ctx.fill();
        }
    });

    ctx.restore();
}

function drawGrid() {
    if (!ctx) return;
    ctx.save();
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.15)';
    ctx.lineWidth = 1;
    const shakeX = gameState.shake?.x || 0;
    const shakeY = gameState.shake?.y || 0;
    for (let r = 0; r <= CONFIG.ROWS; r++) {
        ctx.beginPath();
        ctx.moveTo(shakeX, r * tileSize + shakeY);
        ctx.lineTo(canvasWidth + shakeX, r * tileSize + shakeY);
        ctx.stroke();
    }
    for (let c = 0; c <= CONFIG.COLS; c++) {
        ctx.beginPath();
        ctx.moveTo(c * tileSize + shakeX, shakeY);
        ctx.lineTo(c * tileSize + shakeX, canvasHeight + shakeY);
        ctx.stroke();
    }
    ctx.restore();
}

function drawMeshOverlay() {
    if (!gameState.showMesh) return;
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

function drawRLArrows() {
    if (!gameState.showRLOverlay) return;
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
function drawEarthquakeEffect() {
    if (!gameState.earthquakeActive || !gameState.earthquakeEpicenter) return;

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

function render() {
    if (!ctx || !canvas) return;

    ctx.clearRect(0, 0, canvasWidth, canvasHeight);

    // Background
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    // Grid overlay for depth
    drawGrid();

    // Ensure shake has default values
    const shakeX = gameState.shake?.x || 0;
    const shakeY = gameState.shake?.y || 0;

    // Draw tiles - always draw something
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

    // Sprinklers overlay
    drawSprinklers();

    // Draw floods and bombs
    gameState.floods.forEach(drawFlood);
    gameState.bombs.forEach(drawBomb);
    gameState.debris.forEach(drawDebris);
    gameState.furniture.forEach(drawFurniture);

    // Draw smoke
    Object.entries(gameState.smoke).forEach(([key, level]) => {
        const [r, c] = key.split(',').map(Number);
        drawSmoke(r, c, level);
    });

    // Floor strobes to guide even through smoke (drawn above smoke)
    drawStrobes();

    // Draw fires
    gameState.fires.forEach(fire => {
        drawFire(fire.row, fire.col, fire.intensity || 1);
    });

    // Draw particles
    drawParticles();

    // Draw sensors
    if (gameState.showSensors) {
        gameState.sensors.forEach(drawSensor);
    }
    drawSensorZones();

    // Mesh overlay
    drawMeshOverlay();

    // RL arrows
    drawRLArrows();

    // Draw people (sorted by Y for proper overlapping)
    const peopleSorted = [...gameState.people].sort((a, b) => a.row - b.row);
    peopleSorted.forEach(drawPerson);

    // POV dim + dialogue bubble like Minecraft when a person is selected
    if (gameState.povDimEnabled && gameState.selectedPersonId !== null) {
        const person = gameState.people.find(p => p.id === gameState.selectedPersonId && p.alive && !p.escaped);
        if (person) {
            ctx.fillStyle = 'rgba(0, 0, 0, 0.78)'; // heavier dim when enabled
            ctx.fillRect(0, 0, canvasWidth, canvasHeight);

            // Re-draw the selected person on top of the dim
            drawPerson(person);

            // Dialogue bubble above
            const px = person.col * tileSize + tileSize / 2 + shakeX;
            const py = person.row * tileSize + tileSize / 2 + shakeY - tileSize;
            const text = person.state === STATE.PANICKING
                ? "Help! Need a clear path!"
                : person.state === STATE.EVACUATING
                    ? "I see the exit—keep going!"
                    : "Moving out!";
            ctx.font = `${Math.floor(tileSize * 0.9)}px VT323, monospace`;
            const metrics = ctx.measureText(text);
            const w = metrics.width + 12;
            const h = tileSize + 6;
            ctx.fillStyle = 'rgba(0, 120, 90, 0.9)';
            ctx.fillRect(px - w / 2, py - h, w, h);
            ctx.strokeStyle = 'rgba(0, 200, 150, 1)';
            ctx.lineWidth = 2;
            ctx.strokeRect(px - w / 2, py - h, w, h);
            ctx.fillStyle = '#fff';
            ctx.fillText(text, px - w / 2 + 6, py - h / 2 + 4);
        }
    }

    // Earthquake rings
    drawEarthquakeEffect();

    // Alarm flash overlay
    if (gameState.stats.alarmActive) {
        const flash = Math.sin(gameState.time * 8) > 0;
        if (flash) {
            ctx.fillStyle = 'rgba(255, 0, 0, 0.05)';
            ctx.fillRect(0, 0, canvasWidth, canvasHeight);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SIMULATION UPDATE
// ═══════════════════════════════════════════════════════════════════════════

function updateSimulation(dt) {
    if (gameState.paused) return;

    gameState.time += dt;

    // Ensure shake object exists
    if (!gameState.shake) {
        gameState.shake = { x: 0, y: 0, intensity: 0 };
    }

    // Update shake
    gameState.shake.intensity *= 0.9;
    if (gameState.shake.intensity > 0.01) {
        gameState.shake.x = (Math.random() - 0.5) * gameState.shake.intensity * 10;
        gameState.shake.y = (Math.random() - 0.5) * gameState.shake.intensity * 10;
    } else {
        gameState.shake.x = 0;
        gameState.shake.y = 0;
    }

    // Backend-driven mode: skip local sim updates (backend is authoritative)
    if (CONFIG.USE_BACKEND) {
        return;
    }

    // Update fires
    updateFires(dt);

    // Update other hazards
    updateFloods(dt);
    updateBombs(dt);
    updateEarthquake(dt);

    // Update people
    updatePeople(dt);

    // Update sensors
    updateSensors(dt);

    // Update neural predictions
    updateNeuralPredictions(dt);
    updateRLOverlay(dt);

    // Update stats
    updateStats();

    // Check victory condition
    checkVictory();
}

function updateFires(dt) {
    // Fire spread logic
    gameState.fires.forEach(fire => {
        fire.age = (fire.age || 0) + dt;

        // Generate smoke
        for (let dr = -4; dr <= 4; dr++) {
            for (let dc = -4; dc <= 4; dc++) {
                const nr = fire.row + dr;
                const nc = fire.col + dc;
                if (nr >= 0 && nr < CONFIG.ROWS && nc >= 0 && nc < CONFIG.COLS) {
                    const dist = Math.abs(dr) + Math.abs(dc);
                    const key = `${nr},${nc}`;
                    gameState.smoke[key] = Math.min(
                        (gameState.smoke[key] || 0) + 0.02 / (dist + 1),
                        1.5
                    );
                }
            }
        }

        // Spread fire
        if (fire.age > 5 && Math.random() < 0.005 * dt) {
            const dirs = [[-1, 0], [1, 0], [0, -1], [0, 1]];
            const [dr, dc] = dirs[Math.floor(Math.random() * 4)];
            const nr = fire.row + dr;
            const nc = fire.col + dc;

            if (nr > 0 && nr < CONFIG.ROWS - 1 && nc > 0 && nc < CONFIG.COLS - 1) {
                const tile = gameState.maze[nr]?.[nc];
                if (tile !== TILE.WALL && tile !== TILE.EXIT) {
                    if (!gameState.fires.some(f => f.row === nr && f.col === nc)) {
                        gameState.fires.push({ row: nr, col: nc, age: 0, intensity: 0.8 });
                        gameState.shake.intensity = Math.min(gameState.shake.intensity + 0.5, 3);
                        addChatMessage('fire', `Fire spreading to (${nr}, ${nc})!`);
                    }
                }
            }
        }
    });

    // Smoke decay
    Object.keys(gameState.smoke).forEach(key => {
        gameState.smoke[key] *= 0.995;
        if (gameState.smoke[key] < 0.02) {
            delete gameState.smoke[key];
        }
    });

    // Trigger alarm if fire exists
    if (gameState.fires.length > 0 && !gameState.stats.alarmActive) {
        triggerAlarm();
    }
}

function updateFloods(dt) {
    const newFloods = [];
    const seen = new Set(gameState.floods.map(f => `${f.row},${f.col}`));
    let extinguished = false;

    gameState.floods.forEach(flood => {
        flood.depth = Math.min(1.5, (flood.depth || 0.6) + 0.25 * dt);

        // Spread gently to neighbors
        if (Math.random() < 0.45 * dt) {
            const dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]];
            const [dr, dc] = dirs[Math.floor(Math.random() * dirs.length)];
            const nr = flood.row + dr;
            const nc = flood.col + dc;

            if (inBounds(nr, nc) && gameState.maze[nr]?.[nc] !== TILE.WALL) {
                const key = `${nr},${nc}`;
                if (!seen.has(key)) {
                    seen.add(key);
                    newFloods.push({ row: nr, col: nc, depth: 0.5 });
                }
            }
        }

        // Extinguish fires on water tiles
        const fireIndex = gameState.fires.findIndex(f => f.row === flood.row && f.col === flood.col);
        if (fireIndex !== -1) {
            gameState.fires.splice(fireIndex, 1);
            extinguished = true;
        }

        // Thin smoke on flooded cells
        const floodKey = `${flood.row},${flood.col}`;
        if (gameState.smoke[floodKey]) {
            gameState.smoke[floodKey] = Math.max(0, gameState.smoke[floodKey] - 0.6);
            if (gameState.smoke[floodKey] < 0.05) delete gameState.smoke[floodKey];
        }

        // Damage sensors submerged in water
        gameState.sensors.forEach(sensor => {
            if (sensor.row === flood.row && sensor.col === flood.col) {
                sensor.health -= 8 * dt;
            }
        });
    });

    if (extinguished) {
        addChatMessage('system', 'Flooded tiles doused nearby flames.');
    }

    gameState.floods.push(...newFloods);
}

function updateBombs(dt) {
    const activeBombs = [];

    gameState.bombs.forEach(bomb => {
        bomb.timer = (bomb.timer ?? 3) - dt;
        bomb.radius = bomb.radius || 3;

        if (bomb.timer <= 0) {
            explodeBomb(bomb);
        } else {
            activeBombs.push(bomb);
        }
    });

    gameState.bombs = activeBombs;
}

function explodeBomb(bomb) {
    gameState.shake.intensity = Math.max(gameState.shake.intensity + 3, 4);
    addChatMessage('fire', `Bomb detonated at (${bomb.row}, ${bomb.col})!`);
    triggerAlarm();

    collapseArea(bomb.row, bomb.col, bomb.radius + 1, 0.5);

    // Damage people
    gameState.people.forEach(person => {
        if (!person.alive || person.escaped) return;

        const dist = Math.abs(person.row - bomb.row) + Math.abs(person.col - bomb.col);
        if (dist <= bomb.radius + 1) {
            const damage = Math.max(10, (bomb.radius - dist + 1) * 35);
            person.health -= damage;
            person.state = STATE.PANICKING;

            if (person.health <= 0) {
                person.alive = false;
                addChatMessage('death', `Person ${person.id} was lost in an explosion!`);
            }
        }
    });

    // Damage sensors
    gameState.sensors.forEach(sensor => {
        const dist = Math.abs(sensor.row - bomb.row) + Math.abs(sensor.col - bomb.col);
        if (dist <= bomb.radius + 1) {
            sensor.health -= 40 / Math.max(1, dist);
            sensor.triggered = true;
        }
    });

    // Scorch area and add smoke
    for (let dr = -bomb.radius; dr <= bomb.radius; dr++) {
        for (let dc = -bomb.radius; dc <= bomb.radius; dc++) {
            const nr = bomb.row + dr;
            const nc = bomb.col + dc;
            if (!inBounds(nr, nc)) continue;

            const dist = Math.abs(dr) + Math.abs(dc);
            if (dist <= bomb.radius && gameState.maze[nr]?.[nc] !== TILE.WALL && gameState.maze[nr]?.[nc] !== TILE.EXIT) {
                if (!gameState.fires.some(f => f.row === nr && f.col === nc) && Math.random() < 0.35) {
                    gameState.fires.push({ row: nr, col: nc, age: 0, intensity: 1 });
                }
                const key = `${nr},${nc}`;
                gameState.smoke[key] = (gameState.smoke[key] || 0) + 0.7;
            }
        }
    }
}

function updateEarthquake(dt) {
    if (!gameState.earthquakeActive || !gameState.earthquakeEpicenter) return;

    gameState.earthquakeTimer -= dt;
    const epic = gameState.earthquakeEpicenter;
    const baseImpact = Math.max(0, gameState.earthquakeTimer);

    // Intensify shake while active
    gameState.shake.intensity = Math.max(
        gameState.shake.intensity,
        2.5 + Math.sin(gameState.time * 12) * 0.5 + baseImpact * 0.6
    );

    // Damage sensors and people based on proximity
    gameState.sensors.forEach(sensor => {
        const dist = Math.abs(sensor.row - epic.row) + Math.abs(sensor.col - epic.col);
        const impact = Math.max(0, 1 - dist / 12);
        if (impact > 0) {
            sensor.health -= impact * 12 * dt;
            sensor.triggered = true;
        }
    });

    gameState.people.forEach(person => {
        if (!person.alive || person.escaped) return;

        const dist = Math.abs(person.row - epic.row) + Math.abs(person.col - epic.col);
        const impact = Math.max(0, 1 - dist / 14);

        if (impact > 0 && Math.random() < 0.8 * dt) {
            person.health -= impact * 18;
            if (person.state !== STATE.WARDEN) {
                person.state = STATE.PANICKING;
            }
        }
    });

    // Collapse environment
    if (Math.random() < 0.35 * dt) {
        const radius = 4 + Math.floor(Math.random() * 4);
        collapseArea(epic.row, epic.col, radius, 0.25);
    }

    if (gameState.earthquakeTimer <= 0) {
        gameState.earthquakeActive = false;
        gameState.earthquakeEpicenter = null;
        addChatMessage('system', 'Earthquake tremors subside.');
    }
}

function updatePeople(dt) {
    gameState.people.forEach(person => {
        if (!person.alive || person.escaped) return;

        // Check fire damage
        const inFire = gameState.fires.some(f => f.row === person.row && f.col === person.col);
        if (inFire) {
            person.health -= 30 * dt;
            person.state = STATE.PANICKING;
        }

        // Check smoke damage
        const smokeLevel = gameState.smoke[`${person.row},${person.col}`] || 0;
        if (smokeLevel > 0.5) {
            person.health -= smokeLevel * 10 * dt;
        }

        // Check flood effects
        if (isFlooded(person.row, person.col)) {
            person.health -= 6 * dt;
            if (!person.isWarden) {
                person.state = STATE.PANICKING;
            }
        }

        // Debris hazards
        if (isDebris(person.row, person.col)) {
            person.health -= 15 * dt;
            person.state = STATE.PANICKING;
        }

        // Panic near ticking bombs
        const nearBomb = gameState.bombs.some(b =>
            Math.abs(b.row - person.row) + Math.abs(b.col - person.col) <= 2
        );
        if (nearBomb && !person.isWarden) {
            person.state = STATE.PANICKING;
        }

        // Earthquake shake damage
        if (gameState.earthquakeActive && gameState.earthquakeEpicenter) {
            const dist = Math.abs(person.row - gameState.earthquakeEpicenter.row) +
                Math.abs(person.col - gameState.earthquakeEpicenter.col);
            const impact = Math.max(0, 1 - dist / 18);
            if (impact > 0 && Math.random() < 0.6 * dt) {
                person.health -= impact * 12;
                if (!person.isWarden) {
                    person.state = STATE.PANICKING;
                }
            }
        }

        // Check death
        if (person.health <= 0) {
            person.alive = false;
            addChatMessage('death', `Person ${person.id} has perished!`);
            if (gameState.selectedPersonId === person.id) {
                setDefaultPOV();
            }
            return;
        }

        // Check escape
        if (gameState.maze[person.row]?.[person.col] === TILE.EXIT) {
            person.escaped = true;
            addChatMessage('escape', `Person ${person.id} escaped safely!`);
            if (gameState.selectedPersonId === person.id) {
                setDefaultPOV();
            }
            return;
        }

        // Update awareness
        if (gameState.stats.alarmActive) {
            if (person.state === STATE.WORKING) {
                person.awareness = (person.awareness || 0) + 0.15 * dt;
            } else if (person.state === STATE.HEADPHONES) {
                person.awareness = (person.awareness || 0) + 0.02 * dt;
            }

            if (person.awareness >= 0.7 && person.state !== STATE.EVACUATING && person.state !== STATE.WARDEN) {
                person.state = STATE.EVACUATING;
            }
        }

        // Movement towards exit
        if (person.state === STATE.EVACUATING || person.state === STATE.WARDEN || person.state === STATE.PANICKING) {
            const moved = moveTowardsExit(person, dt);
            if (moved) {
                person.stuckTimer = 0;
            } else {
                person.stuckTimer = (person.stuckTimer || 0) + dt;
                if (person.stuckTimer > 2.0) {
                    if (createLadderToCorridor(person)) {
                        person.stuckTimer = 0;
                    }
                }
            }
        }
    });
}

function moveTowardsExit(person, dt) {
    let moved = false;
    // Find nearest exit
    let nearestExit = null;
    let nearestDist = Infinity;

    gameState.exits.forEach(exit => {
        const dist = Math.abs(exit[0] - person.row) + Math.abs(exit[1] - person.col);
        // Check if exit is blocked by fire
        const blocked = gameState.fires.some(f =>
            Math.abs(f.row - exit[0]) < 3 && Math.abs(f.col - exit[1]) < 3
        );
        if (!blocked && dist < nearestDist) {
            nearestDist = dist;
            nearestExit = exit;
        }
    });

    if (!nearestExit) return;

    // Smarter move selection: pick the lowest-cost safe neighbor
    const dirs = [
        [1, 0], [-1, 0], [0, 1], [0, -1],
        [0, 0] // option to wait if everything is dangerous
    ];

    const fireDistCache = {};
    function distToNearestFire(r, c) {
        const key = `${r},${c}`;
        if (fireDistCache[key] !== undefined) return fireDistCache[key];
        let best = Infinity;
        gameState.fires.forEach(f => {
            const d = Math.abs(f.row - r) + Math.abs(f.col - c);
            if (d < best) best = d;
        });
        fireDistCache[key] = best;
        return best;
    }

    let bestMove = null;
    let bestCost = Infinity;

    dirs.forEach(([mr, mc]) => {
        const newRow = person.row + mr;
        const newCol = person.col + mc;

        if (newRow < 0 || newRow >= CONFIG.ROWS || newCol < 0 || newCol >= CONFIG.COLS) return;

        const tile = gameState.maze[newRow]?.[newCol];
        if (tile === TILE.WALL) return;
        if (isDebris(newRow, newCol)) return;
        if (hasBomb(newRow, newCol)) return;
        if (gameState.fires.some(f => f.row === newRow && f.col === newCol)) return;

        const distExit = Math.abs(nearestExit[0] - newRow) + Math.abs(nearestExit[1] - newCol);
        const fireDist = distToNearestFire(newRow, newCol);
        let firePenalty = 0;
        if (fireDist < 2) firePenalty = 100;
        else if (fireDist < 4) firePenalty = 30;
        else if (fireDist < 6) firePenalty = 10;

        // Prefer corridors/exits/doors/ladders
        let tileBias = 0;
        if (tile === TILE.CORRIDOR || tile === TILE.EXIT) tileBias -= 2;
        if (tile === TILE.DOOR || tile === TILE.LADDER) tileBias -= 1;

        const cost = distExit + firePenalty + tileBias;
        if (cost < bestCost) {
            bestCost = cost;
            bestMove = [newRow, newCol];
        }
    });

    if (bestMove) {
        const floodSlow = isFlooded(bestMove[0], bestMove[1]) ? 0.5 : 1;
        const speed = (person.isWarden ? 1.2 : (person.state === STATE.PANICKING ? 1.3 : 1.0)) * floodSlow;
        if (Math.random() < 0.1 * speed * dt * 60) {
            person.row = bestMove[0];
            person.col = bestMove[1];
            moved = true;
        }
    }
    return moved;
}

function createLadderToCorridor(person) {
    // Find nearest corridor in straight or short Manhattan range and carve a ladder/door path
    let best = null;
    let bestDist = Infinity;
    for (let r = 0; r < CONFIG.ROWS; r++) {
        for (let c = 0; c < CONFIG.COLS; c++) {
            if (gameState.maze[r]?.[c] === TILE.CORRIDOR) {
                const dist = Math.abs(r - person.row) + Math.abs(c - person.col);
                if (dist < bestDist) {
                    bestDist = dist;
                    best = { r, c };
                }
            }
        }
    }
    if (!best || bestDist > 12) return false;

    // Carve a simple Manhattan line: first rows then cols
    let curR = person.row;
    let curC = person.col;
    while (curR !== best.r || curC !== best.c) {
        if (curR < best.r) curR++;
        else if (curR > best.r) curR--;
        else if (curC < best.c) curC++;
        else if (curC > best.c) curC--;

        if (gameState.maze[curR]?.[curC] === TILE.WALL) {
            gameState.maze[curR][curC] = TILE.LADDER;
        } else if (gameState.maze[curR]?.[curC] === TILE.CARPET || gameState.maze[curR]?.[curC] === TILE.FLOOR) {
            gameState.maze[curR][curC] = TILE.CORRIDOR;
        }
    }
    return true;
}

function updateSensors(dt) {
    gameState.sensors.forEach(sensor => {
        if (sensor.health <= 0) return;

        // Check for nearby fire
        let value = 0;
        gameState.fires.forEach(fire => {
            const dist = Math.abs(sensor.row - fire.row) + Math.abs(sensor.col - fire.col);
            if (dist < 10) {
                if (sensor.type === 'temperature') {
                    value = Math.max(value, 100 / (dist + 1) + 22);
                } else if (sensor.type === 'smoke') {
                    value = Math.max(value, gameState.smoke[`${sensor.row},${sensor.col}`] || 0);
                } else if (sensor.type === 'co') {
                    value = Math.max(value, 50 / (dist + 1));
                }
            }
        });

        // Motion detection
        if (sensor.type === 'motion') {
            gameState.people.forEach(p => {
                if (p.alive && !p.escaped) {
                    const dist = Math.abs(sensor.row - p.row) + Math.abs(sensor.col - p.col);
                    if (dist < 5) {
                        value += 1 / (dist + 1);
                    }
                }
            });
        }

        sensor.value = value;

        // Check threshold
        const thresholds = {
            temperature: 45,
            smoke: 0.3,
            co: 35,
            motion: 0.5
        };

        sensor.triggered = value > (thresholds[sensor.type] || 0.5);

        // Damage sensor in fire
        if (gameState.fires.some(f => f.row === sensor.row && f.col === sensor.col)) {
            sensor.health -= 5 * dt;
        }
    });
}

function updateNeuralPredictions(dt) {
    // Simulate neural network predictions
    if (gameState.fires.length > 0) {
        gameState.neural.confidence = Math.min(gameState.neural.confidence + 0.05 * dt, 0.95);

        // Generate predictions based on fire positions
        gameState.predictions = [];
        gameState.fires.forEach(fire => {
            const directions = [[-1, 0], [1, 0], [0, -1], [0, 1]];
            directions.forEach(([dr, dc]) => {
                for (let step = 1; step <= 3; step++) {
                    const nr = fire.row + dr * step;
                    const nc = fire.col + dc * step;
                    if (nr > 0 && nr < CONFIG.ROWS - 1 && nc > 0 && nc < CONFIG.COLS - 1) {
                        if (gameState.maze[nr]?.[nc] !== TILE.WALL) {
                            gameState.predictions.push({
                                row: nr,
                                col: nc,
                                prob: 0.7 * Math.pow(0.7, step)
                            });
                        }
                    }
                }
            });
        });
    } else {
        gameState.neural.confidence *= 0.95;
    }

    // Update pheromone averages
    gameState.neural.safePheromone = 0.1 + gameState.stats.escaped * 0.05;
    gameState.neural.dangerPheromone = gameState.fires.length * 0.2;

    // RL decisions
    if (gameState.stats.alarmActive && Math.random() < 0.01 * dt) {
        gameState.rl.decisions++;
        gameState.rl.avgReward = (gameState.rl.avgReward * 0.9) + (Math.random() * 5);
    }
}

function updateRLOverlay(dt) {
    // Decay arrows
    gameState.rlOverlay.arrows.forEach(a => a.ttl -= dt);
    gameState.rlOverlay.arrows = gameState.rlOverlay.arrows.filter(a => a.ttl > 0);

    if (!gameState.showRLOverlay) return;
    const wardens = gameState.people.filter(p => p.isWarden && p.alive && !p.escaped);
    if (wardens.length === 0) return;

    if (gameState.stats.alarmActive && Math.random() < 0.8 * dt) {
        const w = wardens[Math.floor(Math.random() * wardens.length)];
        const dirs = [
            { dx: 1, dy: 0, label: 'reroute east' },
            { dx: -1, dy: 0, label: 'hold west' },
            { dx: 0, dy: 1, label: 'push south' },
            { dx: 0, dy: -1, label: 'pull north' }
        ];
        const pick = dirs[Math.floor(Math.random() * dirs.length)];
        gameState.rlOverlay.arrows.push({
            row: w.row,
            col: w.col,
            dx: pick.dx,
            dy: pick.dy,
            ttl: 2.5
        });
        const entry = `Warden ${w.id}: ${pick.label}`;
        gameState.rlOverlay.log.unshift(entry);
        if (gameState.rlOverlay.log.length > 6) {
            gameState.rlOverlay.log.pop();
        }
    }
}

function updateStats() {
    gameState.stats.escaped = gameState.people.filter(p => p.escaped).length;
    gameState.stats.deaths = gameState.people.filter(p => !p.alive).length;
}

function checkVictory() {
    const alive = gameState.people.filter(p => p.alive && !p.escaped).length;
    if (alive === 0 && gameState.people.length > 0) {
        showVictoryScreen();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// UI UPDATES
// ═══════════════════════════════════════════════════════════════════════════

function updateUI() {
    // Stats
    document.getElementById('alive-count').textContent =
        gameState.stats.total - gameState.stats.escaped - gameState.stats.deaths;
    document.getElementById('escaped-count').textContent = gameState.stats.escaped;
    document.getElementById('death-count').textContent = gameState.stats.deaths;

    // Alarm indicator
    const alarmIndicator = document.getElementById('alarm-indicator');
    if (gameState.stats.alarmActive) {
        alarmIndicator.className = 'alarm-on';
        alarmIndicator.querySelector('.alarm-text').textContent = 'ALARM ACTIVE!';
    } else {
        alarmIndicator.className = 'alarm-off';
        alarmIndicator.querySelector('.alarm-text').textContent = 'ALARM OFF';
    }

    // Neural confidence
    const confidence = gameState.neural.confidence;
    document.getElementById('neural-confidence-bar').style.width = (confidence * 100) + '%';
    document.getElementById('neural-confidence').textContent = Math.round(confidence * 100) + '%';

    // Speed
    document.getElementById('speed-display').textContent = gameState.speed.toFixed(1) + 'x';

    // Sensors
    const tempSensors = gameState.sensors.filter(s => s.type === 'temperature' && s.health > 0);
    const avgTemp = tempSensors.length > 0
        ? tempSensors.reduce((sum, s) => sum + s.value, 0) / tempSensors.length
        : 22;
    document.getElementById('sensor-temp').textContent = avgTemp.toFixed(1) + 'C';

    const smokeSensors = gameState.sensors.filter(s => s.type === 'smoke' && s.health > 0);
    const avgSmoke = smokeSensors.length > 0
        ? smokeSensors.reduce((sum, s) => sum + s.value, 0) / smokeSensors.length
        : 0;
    document.getElementById('sensor-smoke').textContent = avgSmoke.toFixed(2);

    const coSensors = gameState.sensors.filter(s => s.type === 'co' && s.health > 0);
    const avgCo = coSensors.length > 0
        ? coSensors.reduce((sum, s) => sum + s.value, 0) / coSensors.length
        : 0;
    document.getElementById('sensor-co').textContent = Math.round(avgCo) + ' ppm';

    const motionSensors = gameState.sensors.filter(s => s.type === 'motion' && s.triggered);
    document.getElementById('sensor-motion').textContent = motionSensors.length;

    const aliveSensors = gameState.sensors.filter(s => s.health > 0);
    const coverage = (aliveSensors.length / gameState.sensors.length) * 100;
    document.getElementById('sensor-coverage').style.width = coverage + '%';
    document.getElementById('sensor-coverage-text').textContent = Math.round(coverage) + '%';

    // RL stats
    document.getElementById('rl-decisions').textContent = gameState.rl.decisions;
    document.getElementById('rl-reward').textContent = gameState.rl.avgReward.toFixed(2);
    document.getElementById('rl-epsilon').textContent = Math.round(gameState.rl.epsilon * 100) + '%';

    // Fire stats
    document.getElementById('fire-count').textContent = gameState.fires.length;
    document.getElementById('smoke-zones').textContent = Object.keys(gameState.smoke).length;
    document.getElementById('bomb-count').textContent = gameState.bombs.length;
    document.getElementById('flood-tiles').textContent = gameState.floods.length;
    document.getElementById('earthquake-status').textContent =
        gameState.earthquakeActive ? 'Tremors' : 'Stable';

    const dangerLevel = Math.min(
        gameState.fires.length * 10 +
        Object.keys(gameState.smoke).length / 10 +
        gameState.bombs.length * 8 +
        gameState.floods.length * 4 +
        (gameState.earthquakeActive ? 20 : 0),
        100
    );
    document.getElementById('danger-level').style.width = dangerLevel + '%';

    // Pheromones
    document.getElementById('safe-pheromone').style.width = Math.min(gameState.neural.safePheromone * 10, 100) + '%';
    document.getElementById('danger-pheromone').style.width = Math.min(gameState.neural.dangerPheromone * 10, 100) + '%';
    document.getElementById('tracked-paths').textContent = gameState.stats.escaped * 3;

    // Predictions
    document.querySelector('#prediction-zones .prediction-count').textContent = gameState.predictions.length;

    // Confidence ring
    const circumference = 2 * Math.PI * 40;
    const offset = circumference * (1 - confidence);
    document.getElementById('confidence-ring').style.strokeDashoffset = offset;
    document.getElementById('confidence-percent').textContent = Math.round(confidence * 100) + '%';

    updatePythonTelemetryUI();
    updatePOVUI();
    updateRLLogUI();

    // Pause overlay
    document.getElementById('pause-overlay').classList.toggle('hidden', !gameState.paused);
}

function updatePythonTelemetryUI() {
    const statusEl = document.getElementById('py-status');
    if (!statusEl) return;

    const data = gameState.pythonTelemetry.stats;
    const loading = gameState.pythonTelemetry.loading;
    const error = gameState.pythonTelemetry.error;

    statusEl.textContent = loading ? 'Running...' : (error ? 'Error' : (data ? 'Complete' : 'Idle'));

    const fields = [
        ['py-steps', data?.steps_run],
        ['py-escaped', data?.escaped],
        ['py-deaths', data?.deaths],
        ['py-alive', data?.alive],
        ['py-fires', data?.fires_active],
        ['py-confidence', data ? `${Math.round((data.neural_confidence || 0) * 100)}%` : undefined],
        ['py-rl', data?.rl_decisions],
        ['py-coverage', data ? `${Math.round((data.sensor_coverage || 0) * 100)}%` : undefined],
        ['py-temp', data?.avg_temp ? data.avg_temp.toFixed(1) + 'C' : undefined],
        ['py-mesh-nodes', data?.mesh_nodes],
        ['py-mesh-links', data?.mesh_links],
        ['py-mesh-degree', data?.mesh_avg_degree ? data.mesh_avg_degree.toFixed(1) : undefined],
        ['py-mesh-broadcasts', data?.mesh_broadcast_steps]
    ];

    fields.forEach(([id, val]) => {
        const el = document.getElementById(id);
        if (el) {
            el.textContent = val !== undefined && val !== null ? val : '-';
        }
    });

    const btn = document.getElementById('run-python-btn');
    if (btn) {
        btn.disabled = loading;
        btn.textContent = loading ? 'Running...' : 'Run Python Sim';
    }

    const errorEl = document.getElementById('py-error');
    if (errorEl) {
        errorEl.textContent = error || '';
        errorEl.classList.toggle('hidden', !error);
    }
}

function updatePOVUI() {
    const person = gameState.people.find(p => p.id === gameState.selectedPersonId);
    const nameEl = document.getElementById('pov-name');
    const statusEl = document.getElementById('pov-status');
    const dialogueEl = document.getElementById('pov-dialogue');
    const healthEl = document.getElementById('pov-health');
    const visionEl = document.getElementById('pov-vision');

    if (!nameEl || !statusEl || !dialogueEl || !healthEl || !visionEl) return;

    if (!person || !person.alive || person.escaped) {
        nameEl.textContent = 'No POV selected';
        statusEl.textContent = '-';
        dialogueEl.textContent = 'Select a person with tool 6 (POV)';
        healthEl.textContent = '-';
        visionEl.textContent = '-';
        const povCtx = getPOVContext();
        if (povCtx) {
            const povCanvas = document.getElementById('pov-canvas');
            povCtx.clearRect(0, 0, povCanvas.width, povCanvas.height);
        }
        return;
    }

    nameEl.textContent = `Person ${person.id}${person.isWarden ? ' (Warden)' : ''}`;
    statusEl.textContent = `${person.state} | row ${person.row}, col ${person.col}`;
    healthEl.textContent = `${Math.max(0, Math.round(person.health))}%`;

    const smoke = gameState.smoke[`${person.row},${person.col}`] || 0;
    const fireNearby = gameState.fires.some(f => Math.abs(f.row - person.row) + Math.abs(f.col - person.col) <= 2);
    const debrisHere = isDebris(person.row, person.col);
    const floodHere = isFlooded(person.row, person.col);

    const visionBits = [];
    if (smoke > 0.5) visionBits.push('smoke');
    if (fireNearby) visionBits.push('fire nearby');
    if (floodHere) visionBits.push('water');
    if (debrisHere) visionBits.push('rubble');
    visionEl.textContent = visionBits.length ? visionBits.join(', ') : 'clear';

    dialogueEl.textContent = buildDialogue(person, { smoke, fireNearby, debrisHere, floodHere });
    renderPOV(person);
}

function buildDialogue(person, context) {
    if (!person.alive) return '...';
    if (person.escaped) return 'Made it out!';

    const lines = [];
    if (person.isWarden) {
        lines.push('“Stay calm, follow me!”');
    } else if (person.state === STATE.PANICKING) {
        lines.push('“I have to move now!”');
    } else if (person.state === STATE.EVACUATING) {
        lines.push('“I see the exit—keep going!”');
    } else if (person.state === STATE.HEADPHONES) {
        lines.push('“Huh? What’s happening?”');
    } else {
        lines.push('“What was that noise?”');
    }

    if (context.fireNearby) lines.push('“I smell smoke—danger close!”');
    if (context.smoke > 0.7) lines.push('“Can’t see, smoke everywhere!”');
    if (context.debrisHere) lines.push('“Rubble here—watch your step!”');
    if (context.floodHere) lines.push('“Feet are soaked—water incoming!”');
    if (person.health < 40) lines.push('“I’m hurt... need help.”');

    return lines.slice(0, 2).join(' ');
}

function updateRLLogUI() {
    const list = document.getElementById('rl-log-list');
    if (!list) return;
    list.innerHTML = '';
    if (!gameState.rlOverlay.log.length) {
        const item = document.createElement('div');
        item.className = 'rl-log-item';
        item.textContent = 'No decisions yet';
        list.appendChild(item);
        return;
    }
    gameState.rlOverlay.log.forEach(entry => {
        const item = document.createElement('div');
        item.className = 'rl-log-item';
        item.textContent = entry;
        list.appendChild(item);
    });
}
function setDefaultPOV() {
    const candidate = gameState.people.find(p => p.alive && !p.escaped);
    if (candidate) {
        gameState.selectedPersonId = candidate.id;
    }
}

function populateFurniture() {
    gameState.furniture = [];
    for (let r = 2; r < CONFIG.ROWS - 2; r++) {
        for (let c = 2; c < CONFIG.COLS - 2; c++) {
            if (gameState.maze[r][c] === TILE.CARPET && Math.random() < 0.06) {
                gameState.furniture.push({ row: r, col: c, type: 'desk' });
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GAME ACTIONS
// ═══════════════════════════════════════════════════════════════════════════

function addFire(row, col) {
    if (row < 1 || row >= CONFIG.ROWS - 1 || col < 1 || col >= CONFIG.COLS - 1) return;
    if (gameState.maze[row]?.[col] === TILE.WALL) return;
    if (gameState.fires.some(f => f.row === row && f.col === col)) return;

    if (CONFIG.USE_BACKEND) {
        sendCommand({ action: 'add_fire', row, col });
    } else {
        gameState.fires.push({ row, col, age: 0, intensity: 1 });
        gameState.shake.intensity = 2;
        addChatMessage('fire', `Fire started at (${row}, ${col})!`);
        if (!gameState.stats.alarmActive) triggerAlarm();
    }
}

function addBomb(row, col) {
    if (!inBounds(row, col) || gameState.maze[row]?.[col] === TILE.EXIT) return;
    if (CONFIG.USE_BACKEND) {
        sendCommand({ action: 'add_bomb', row, col });
        addChatMessage('fire', `Bomb triggered at (${row}, ${col}) [backend]`);
    } else {
        gameState.bombs.push({ row, col, timer: 3, radius: 3 });
        gameState.shake.intensity = Math.max(gameState.shake.intensity, 1.5);
        addChatMessage('fire', `Bomb armed at (${row}, ${col}) - detonation imminent!`);
        if (!gameState.stats.alarmActive) {
            triggerAlarm();
        }
    }
}

function addFlood(row, col) {
    if (!inBounds(row, col)) return;
    if (CONFIG.USE_BACKEND) {
        sendCommand({ action: 'add_flood', row, col });
        addChatMessage('fire', `Flood triggered at (${row}, ${col}) [backend]`);
    } else {
        if (isFlooded(row, col)) return;
        gameState.floods.push({ row, col, depth: 0.7 });
        addChatMessage('system', `Flood surge released at (${row}, ${col}).`);
        if (!gameState.stats.alarmActive) {
            triggerAlarm();
        }
    }
}

function collapseArea(row, col, radius, chance = 0.4) {
    for (let dr = -radius; dr <= radius; dr++) {
        for (let dc = -radius; dc <= radius; dc++) {
            const nr = row + dr;
            const nc = col + dc;
            if (!inBounds(nr, nc)) continue;
            const dist = Math.abs(dr) + Math.abs(dc);
            if (dist > radius) continue;
            if (gameState.maze[nr]?.[nc] === TILE.EXIT) continue;
            if (isDebris(nr, nc)) continue;

            const tile = gameState.maze[nr]?.[nc];
            const collapseBonus = tile === TILE.WALL ? 0.25 : 0;
            if (Math.random() < chance + collapseBonus) {
                gameState.debris.push({ row: nr, col: nc });

                // Damage sensors buried
                gameState.sensors.forEach(sensor => {
                    if (sensor.row === nr && sensor.col === nc) {
                        sensor.health -= 60;
                        sensor.triggered = true;
                    }
                });
            }
        }
    }
}

function getNextSensorId() {
    if (!gameState.sensors.length) return 0;
    return Math.max(...gameState.sensors.map(s => s.id)) + 1;
}

function addSensor(row, col) {
    if (!inBounds(row, col)) return;
    if (gameState.maze[row]?.[col] === TILE.WALL || gameState.maze[row]?.[col] === TILE.EXIT) return;
    if (gameState.sensors.some(s => s.row === row && s.col === col && s.health > 0)) return;

    const sensorTypes = ['temperature', 'smoke', 'co', 'motion'];
    const id = getNextSensorId();
    const type = sensorTypes[id % sensorTypes.length];

    gameState.sensors.push({
        id,
        row,
        col,
        type,
        value: type === 'temperature' ? 22 : 0,
        triggered: false,
        health: 100
    });
    addChatMessage('system', `Sensor placed at (${row}, ${col}).`);
}

function selectPersonAt(row, col) {
    let closest = null;
    let bestDist = Infinity;

    gameState.people.forEach(p => {
        if (!p.alive || p.escaped) return;
        const dist = Math.abs(p.row - row) + Math.abs(p.col - col);
        if (dist < bestDist) {
            bestDist = dist;
            closest = p;
        }
    });

    if (closest && bestDist <= 3) {
        gameState.selectedPersonId = closest.id;
        addChatMessage('system', `POV: Person ${closest.id}${closest.isWarden ? ' (Warden)' : ''}`);
        updatePOVUI();
    }
}

function addEarthquake(row, col) {
    if (!inBounds(row, col)) return;
    if (CONFIG.USE_BACKEND) {
        sendCommand({ action: 'add_quake', row, col });
        addChatMessage('alarm', `Earthquake epicenter set at (${row}, ${col}) [backend].`);
        // Client-side visual hint while backend processes
        gameState.earthquakeActive = true;
        gameState.earthquakeTimer = 5;
        gameState.earthquakeEpicenter = { row, col };
        gameState.shake.intensity = Math.max(gameState.shake.intensity, 4);
        return;
    }

    gameState.earthquakeActive = true;
    gameState.earthquakeTimer = 5;
    gameState.earthquakeEpicenter = { row, col };
    gameState.shake.intensity = 4;
    addChatMessage('alarm', `Earthquake epicenter set at (${row}, ${col})!`);
    triggerAlarm();
}

async function runPythonSimulation() {
    if (gameState.pythonTelemetry.loading) return;

    gameState.pythonTelemetry.loading = true;
    gameState.pythonTelemetry.error = '';
    updatePythonTelemetryUI();

    try {
        const response = await fetch(CONFIG.PY_SIM_API_URL);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();
        if (!data.success) {
            throw new Error(data.error || 'Simulation failed');
        }

        gameState.pythonTelemetry.stats = data.stats || null;
        gameState.pythonTelemetry.lastRun = Date.now();
        addChatMessage('system', 'Python headless simulation completed.');
    } catch (err) {
        gameState.pythonTelemetry.error = err.message || 'Unknown error';
        addChatMessage('system', `Python sim error: ${gameState.pythonTelemetry.error}`);
    } finally {
        gameState.pythonTelemetry.loading = false;
        updatePythonTelemetryUI();
    }
}

function triggerAlarm() {
    if (gameState.stats.alarmActive) return;

    if (CONFIG.USE_BACKEND) {
        sendCommand({ action: 'trigger_alarm' });
        addChatMessage('alarm', 'EMERGENCY! ALARM ACTIVATED! (backend)');
    } else {
        gameState.stats.alarmActive = true;
        addChatMessage('alarm', 'EMERGENCY! ALARM ACTIVATED!');
        // Wake up all wardens
        gameState.people.forEach(p => {
            if (p.isWarden && p.alive) {
                p.state = STATE.WARDEN;
                p.awareness = 1;
            }
        });
    }
}

function resetGame() {
    // Show loading
    document.getElementById('loading-screen').style.display = 'flex';
    document.getElementById('game-container').classList.add('hidden');
    document.getElementById('victory-screen').classList.add('hidden');

    // Reset loading bar
    document.querySelector('.loading-bar').style.animation = 'none';
    setTimeout(() => {
        document.querySelector('.loading-bar').style.animation = 'loading 2s ease-out forwards';
    }, 10);

    // Reinitialize
    setTimeout(() => {
        initGame();
    }, 100);
}

function togglePause() {
    gameState.paused = !gameState.paused;
    if (gameState.paused) {
        addChatMessage('system', 'Game paused');
    } else {
        addChatMessage('system', 'Game resumed');
    }
}

function changeSpeed(delta) {
    gameState.speed = Math.max(0.5, Math.min(5, gameState.speed + delta));
}

function addChatMessage(type, text) {
    const container = document.getElementById('chat-messages');
    const msg = document.createElement('div');
    msg.className = `chat-message ${type}`;
    msg.textContent = `[${type.charAt(0).toUpperCase() + type.slice(1)}] ${text}`;
    container.appendChild(msg);

    // Keep only last 5 messages
    while (container.children.length > 5) {
        container.removeChild(container.firstChild);
    }

    // Auto-remove after animation
    setTimeout(() => {
        if (msg.parentNode) {
            msg.remove();
        }
    }, 10000);
}

function changeZoom(delta) {
    gameState.zoom = Math.max(0.8, Math.min(1.8, gameState.zoom + delta));
    resizeCanvas();
}

function showVictoryScreen() {
    gameState.paused = true;

    const victoryScreen = document.getElementById('victory-screen');
    victoryScreen.classList.remove('hidden');

    // Update stats
    document.getElementById('victory-escaped').textContent = gameState.stats.escaped;
    document.getElementById('victory-deaths').textContent = gameState.stats.deaths;
    document.getElementById('victory-critical').textContent = gameState.rescue?.criticalSaves || 0;
    document.getElementById('victory-prevented').textContent = gameState.rescue?.deathsPrevented || 0;
    const safeAvg = (gameState.neural?.safePheromone || 0).toFixed(2);
    const dangerAvg = (gameState.neural?.dangerPheromone || 0).toFixed(2);
    document.getElementById('victory-pheromone').textContent = `${safeAvg} / ${dangerAvg}`;
    document.getElementById('victory-rl-decisions').textContent = gameState.rl?.decisions || 0;
    document.getElementById('victory-coverage').textContent = `${gameState.neural?.coverage || 0}%`;

    const elapsed = Math.floor(gameState.time);
    const minutes = Math.floor(elapsed / 60);
    const seconds = elapsed % 60;
    document.getElementById('victory-time').textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;

    // Calculate rating (stars)
    const ratingContainer = document.getElementById('victory-rating');
    ratingContainer.innerHTML = '';

    let stars = 0;
    if (gameState.stats.deaths === 0) stars = 3;
    else if (gameState.stats.deaths <= 3) stars = 2;
    else if (gameState.stats.escaped > gameState.stats.deaths) stars = 1;

    for (let i = 0; i < 3; i++) {
        const star = document.createElement('div');
        star.className = 'star' + (i < stars ? ' earned' : '');
        star.style.animationDelay = (i * 0.2) + 's';
        ratingContainer.appendChild(star);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// EVENT HANDLERS
// ═══════════════════════════════════════════════════════════════════════════

function setupEventListeners() {
    // Canvas click
    canvas.addEventListener('click', (e) => {
        if (gameState.paused) return;

        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;

        const x = (e.clientX - rect.left) * scaleX;
        const y = (e.clientY - rect.top) * scaleY;

        const col = Math.floor(x / tileSize);
        const row = Math.floor(y / tileSize);

        switch (gameState.selectedTool) {
            case 'fire':
                addFire(row, col);
                break;
            case 'bomb':
                addBomb(row, col);
                break;
            case 'earthquake':
                addEarthquake(row, col);
                break;
            case 'flood':
                addFlood(row, col);
                break;
            case 'sensor':
                addSensor(row, col);
                break;
            case 'inspect':
                selectPersonAt(row, col);
                break;
        }
    });

    // Keyboard
    document.addEventListener('keydown', (e) => {
        switch (e.key.toLowerCase()) {
            case ' ':
                e.preventDefault();
                togglePause();
                break;
            case 'a':
                triggerAlarm();
                break;
            case 'p':
                gameState.showPredictions = !gameState.showPredictions;
                addChatMessage('system', `Predictions: ${gameState.showPredictions ? 'ON' : 'OFF'}`);
                break;
            case 't':
                gameState.showPheromones = !gameState.showPheromones;
                addChatMessage('system', `Pheromones: ${gameState.showPheromones ? 'ON' : 'OFF'}`);
                break;
            case 's':
                gameState.showSensors = !gameState.showSensors;
                addChatMessage('system', `Sensors: ${gameState.showSensors ? 'ON' : 'OFF'}`);
                break;
            case 'f':
                gameState.showStrobes = !gameState.showStrobes;
                addChatMessage('system', `Floor strobes: ${gameState.showStrobes ? 'ON' : 'OFF'}`);
                break;
            case 'h':
                gameState.showSprinklers = !gameState.showSprinklers;
                addChatMessage('system', `Sprinklers: ${gameState.showSprinklers ? 'ON' : 'OFF'}`);
                break;
            case 'm':
                gameState.showMesh = !gameState.showMesh;
                addChatMessage('system', `Mesh overlay: ${gameState.showMesh ? 'ON' : 'OFF'}`);
                break;
            case 'l':
                gameState.showRLOverlay = !gameState.showRLOverlay;
                addChatMessage('system', `RL overlay: ${gameState.showRLOverlay ? 'ON' : 'OFF'}`);
                break;
            case 'o':
                gameState.showPredictionBeams = !gameState.showPredictionBeams;
                addChatMessage('system', `Predictions: ${gameState.showPredictionBeams ? 'ON' : 'OFF'}`);
                break;
            case 'z':
                gameState.showSensorZones = !gameState.showSensorZones;
                addChatMessage('system', `Sensor zones: ${gameState.showSensorZones ? 'ON' : 'OFF'}`);
                break;
            case '[':
                changeZoom(-0.1);
                addChatMessage('system', `Zoom: ${(gameState.zoom * 100).toFixed(0)}%`);
                break;
            case ']':
                changeZoom(0.1);
                addChatMessage('system', `Zoom: ${(gameState.zoom * 100).toFixed(0)}%`);
                break;
            case 'r':
                resetGame();
                break;
            case '=':
            case '+':
                changeSpeed(0.5);
                break;
            case '-':
                changeSpeed(-0.5);
                break;
            case 'v':
                gameState.povDimEnabled = !gameState.povDimEnabled;
                addChatMessage('system', `POV dim: ${gameState.povDimEnabled ? 'ON' : 'OFF'}`);
                break;
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
                selectHotbarSlot(parseInt(e.key) - 1);
                break;
        }
    });

    // Hotbar clicks
    document.querySelectorAll('.hotbar-slot[data-tool]').forEach(slot => {
        slot.addEventListener('click', () => {
            const tool = slot.dataset.tool;
            if (tool === 'reset') {
                resetGame();
            } else {
                selectHotbarSlot(Array.from(slot.parentNode.children).indexOf(slot));
            }
        });
    });

    // Restart button
    document.getElementById('restart-btn').addEventListener('click', resetGame);

    // Python simulation trigger
    const pythonBtn = document.getElementById('run-python-btn');
    if (pythonBtn) {
        pythonBtn.addEventListener('click', runPythonSimulation);
    }
}

function selectHotbarSlot(index) {
    const slots = document.querySelectorAll('.hotbar-slot');
    slots.forEach((slot, i) => {
        slot.classList.toggle('selected', i === index);
    });

    const selectedSlot = slots[index];
    if (selectedSlot && selectedSlot.dataset.tool) {
        gameState.selectedTool = selectedSlot.dataset.tool;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// INITIALIZATION
// ═══════════════════════════════════════════════════════════════════════════

async function fetchInitialState() {
    // Try headless backend first
    if (CONFIG.USE_BACKEND) {
        try {
            const response = await fetch(`${CONFIG.HEADLESS_URL}?steps=240&dt=0.033&extended=1`);
            const data = await response.json();
            if (data.success) {
                if (data.maze) gameState.maze = data.maze;
                if (data.exits) gameState.exits = data.exits;
                if (data.people) gameState.people = data.people;
                if (data.stats) {
                    gameState.stats.escaped = data.stats.escaped || 0;
                    gameState.stats.deaths = data.stats.deaths || 0;
                    gameState.stats.total = data.stats.total || CONFIG.TOTAL_PEOPLE;
                    gameState.stats.alarmActive = data.stats.alarmActive || false;
                }
                if (data.sensors) gameState.sensors = data.sensors;
                setDefaultPOV();
                populateFurniture();
                return true;
            }
        } catch (error) {
            console.warn('Headless API not available, will try simulate/local:', error);
        }
    }

    // Fallback to simulate API
    try {
        const response = await fetch(CONFIG.API_URL);
        const data = await response.json();

        if (data.success) {
            gameState.maze = data.maze;
            gameState.exits = data.exits;
            gameState.people = data.people;
            gameState.sensors = data.sensors;
            gameState.stats.total = data.config.totalPeople;
            setDefaultPOV();
            populateFurniture();
            return true;
        }
    } catch (error) {
        console.warn('Simulate API not available, generating locally:', error);
    }

    // Generate locally if API fails
    generateLocalState();
    return true;
}

async function pollBackendState() {
    if (!CONFIG.USE_BACKEND) return;
    try {
        // Smaller step count keeps transient effects (e.g., quake shake) visible between polls
        const response = await fetch(`${CONFIG.HEADLESS_URL}?steps=15&dt=0.033&full=1`);
        const data = await response.json();
        if (!data.success) return;

        // Apply snapshot
        if (data.maze) gameState.maze = data.maze;
        if (data.exits) gameState.exits = data.exits;
        if (data.people) gameState.people = data.people;
        if (data.hazards) {
            const fires = [];
            const floods = [];
            const debris = [];
            data.hazards.forEach(h => {
                if (h.type === 'fire') {
                    fires.push({ row: h.row, col: h.col, intensity: h.intensity || 1 });
                } else if (h.type === 'flood') {
                    floods.push({ row: h.row, col: h.col, depth: h.intensity || 0.7 });
                } else if (h.type === 'debris') {
                    debris.push({ row: h.row, col: h.col, age: h.age || 0 });
                }
            });
            gameState.fires = fires;
            gameState.floods = floods;
            gameState.debris = debris;
            gameState.bombs = [];
        } else if (data.fires) {
            gameState.fires = data.fires.map(f => ({ row: f[0], col: f[1], intensity: 1 }));
        }
        gameState.smoke = data.smoke || {};
        const quakeActive = (gameState.debris && gameState.debris.length > 0) || (typeof data.shake === 'number' && data.shake > 0.1);
        gameState.earthquakeActive = quakeActive;
        if (!quakeActive) {
            gameState.earthquakeTimer = 0;
        }
        gameState.stats.escaped = data.stats?.escaped || 0;
        gameState.stats.deaths = data.stats?.deaths || 0;
        gameState.stats.total = data.stats?.total || CONFIG.TOTAL_PEOPLE;
        gameState.stats.alarmActive = data.stats?.alarmActive || false;
        if (data.sensors) gameState.sensors = data.sensors;
        gameState.neural.confidence = data.neural?.confidence || 0;
        gameState.rl.decisions = data.rl?.decisions || 0;
        gameState.rl.avgReward = data.rl?.avg_reward || 0;
        if (typeof data.shake === 'number') {
            gameState.shake.intensity = Math.max(gameState.shake.intensity * 0.5, data.shake);
        }
        if (data.rescue) {
            gameState.rescue = data.rescue;
        }
        if (data.pheromones) {
            const safe = data.pheromones.safe || [];
            const danger = data.pheromones.danger || [];
            // Compute simple averages to drive UI bars
            const safeFlat = safe.flat ? safe.flat() : [];
            const dangerFlat = danger.flat ? danger.flat() : [];
            const safeAvg = safeFlat.length ? safeFlat.reduce((a, b) => a + b, 0) / safeFlat.length : 0;
            const dangerAvg = dangerFlat.length ? dangerFlat.reduce((a, b) => a + b, 0) / dangerFlat.length : 0;
            gameState.neural.safePheromone = safeAvg;
            gameState.neural.dangerPheromone = dangerAvg;
        }
        // Time for animations
        if (typeof data.time === 'number') {
            gameState.time = data.time;
        }
        // Sensor coverage for victory stats
        if (data.sensor_fusion && data.sensor_fusion.coverage !== undefined) {
            gameState.neural.coverage = data.sensor_fusion.coverage;
        }
    } catch (err) {
        console.warn('Backend poll failed:', err);
    }
}

function generateLocalState() {
    // Generate maze
    gameState.maze = [];
    for (let r = 0; r < CONFIG.ROWS; r++) {
        gameState.maze[r] = [];
        for (let c = 0; c < CONFIG.COLS; c++) {
            gameState.maze[r][c] = TILE.FLOOR;
        }
    }

    // Outer walls
    for (let r = 0; r < CONFIG.ROWS; r++) {
        gameState.maze[r][0] = TILE.WALL;
        gameState.maze[r][CONFIG.COLS - 1] = TILE.WALL;
    }
    for (let c = 0; c < CONFIG.COLS; c++) {
        gameState.maze[0][c] = TILE.WALL;
        gameState.maze[CONFIG.ROWS - 1][c] = TILE.WALL;
    }

    // Corridors
    const hCorr = [Math.floor(CONFIG.ROWS / 3), Math.floor(2 * CONFIG.ROWS / 3)];
    const vCorr = [Math.floor(CONFIG.COLS / 4), Math.floor(CONFIG.COLS / 2), Math.floor(3 * CONFIG.COLS / 4)];

    hCorr.forEach(hr => {
        for (let c = 1; c < CONFIG.COLS - 1; c++) {
            for (let r = hr - 1; r <= hr + 1; r++) {
                if (r > 0 && r < CONFIG.ROWS - 1) {
                    gameState.maze[r][c] = TILE.CORRIDOR;
                }
            }
        }
    });

    vCorr.forEach(vc => {
        for (let r = 1; r < CONFIG.ROWS - 1; r++) {
            for (let c = vc - 1; c <= vc + 1; c++) {
                if (c > 0 && c < CONFIG.COLS - 1) {
                    gameState.maze[r][c] = TILE.CORRIDOR;
                }
            }
        }
    });

    // Rooms
    const sections = [
        [2, hCorr[0] - 2, 2, vCorr[0] - 2],
        [2, hCorr[0] - 2, vCorr[0] + 2, vCorr[1] - 2],
        [2, hCorr[0] - 2, vCorr[1] + 2, vCorr[2] - 2],
        [hCorr[0] + 2, hCorr[1] - 2, 2, vCorr[0] - 2],
        [hCorr[1] + 2, CONFIG.ROWS - 3, 2, vCorr[0] - 2],
        [hCorr[1] + 2, CONFIG.ROWS - 3, vCorr[1] + 2, vCorr[2] - 2],
    ];

    sections.forEach(([r1, r2, c1, c2]) => {
        if (r2 - r1 > 3 && c2 - c1 > 3) {
            // Walls
            for (let r = r1; r <= r2; r++) {
                if (gameState.maze[r][c1] !== TILE.CORRIDOR) gameState.maze[r][c1] = TILE.WALL;
                if (gameState.maze[r][c2] !== TILE.CORRIDOR) gameState.maze[r][c2] = TILE.WALL;
            }
            for (let c = c1; c <= c2; c++) {
                if (gameState.maze[r1][c] !== TILE.CORRIDOR) gameState.maze[r1][c] = TILE.WALL;
                if (gameState.maze[r2][c] !== TILE.CORRIDOR) gameState.maze[r2][c] = TILE.WALL;
            }
            // Carpet interior
            for (let r = r1 + 1; r < r2; r++) {
                for (let c = c1 + 1; c < c2; c++) {
                    if (gameState.maze[r][c] !== TILE.CORRIDOR) {
                        gameState.maze[r][c] = TILE.CARPET;
                    }
                }
            }

            // Corridor-facing doors on every corridor-adjacent wall cell
            for (let c = c1 + 1; c < c2; c++) {
                if (r2 + 1 < CONFIG.ROWS && gameState.maze[r2 + 1][c] === TILE.CORRIDOR) {
                    gameState.maze[r2][c] = TILE.DOOR;
                }
                if (r1 - 1 > 0 && gameState.maze[r1 - 1][c] === TILE.CORRIDOR) {
                    gameState.maze[r1][c] = TILE.DOOR;
                }
            }
            for (let r = r1 + 1; r < r2; r++) {
                if (c2 + 1 < CONFIG.COLS && gameState.maze[r][c2 + 1] === TILE.CORRIDOR) {
                    gameState.maze[r][c2] = TILE.DOOR;
                }
                if (c1 - 1 > 0 && gameState.maze[r][c1 - 1] === TILE.CORRIDOR) {
                    gameState.maze[r][c1] = TILE.DOOR;
                }
            }

            // Interior doors for flow (center and thirds)
            if ((r2 - r1) > 4 && (c2 - c1) > 4) {
                const rMid = Math.floor((r1 + r2) / 2);
                const cMid = Math.floor((c1 + c2) / 2);
                const thirds = [
                    [r1 + Math.floor((r2 - r1) / 3), cMid],
                    [r1 + Math.floor(2 * (r2 - r1) / 3), cMid],
                    [rMid, c1 + Math.floor((c2 - c1) / 3)],
                    [rMid, c1 + Math.floor(2 * (c2 - c1) / 3)],
                ];
                const candidates = [
                    [rMid, cMid],
                    [rMid, cMid - 2],
                    [rMid, cMid + 2],
                    [rMid - 2, cMid],
                    [rMid + 2, cMid],
                    ...thirds,
                ];
                const dedup = {};
                candidates.forEach(([rr, cc]) => {
                    if (rr > r1 && rr < r2 && cc > c1 && cc < c2 && gameState.maze[rr][cc] === TILE.CARPET) {
                        dedup[`${rr},${cc}`] = true;
                    }
                });
                Object.keys(dedup).slice(0, 4).forEach(key => {
                    const [rr, cc] = key.split(',').map(Number);
                    gameState.maze[rr][cc] = TILE.DOOR;
                });
            }
        }
    });

    // Exits
    gameState.exits = [
        [hCorr[0], 1], [hCorr[1], 1],
        [hCorr[0], CONFIG.COLS - 2], [hCorr[1], CONFIG.COLS - 2],
        [1, vCorr[0]], [1, vCorr[1]], [1, vCorr[2]],
        [CONFIG.ROWS - 2, vCorr[0]], [CONFIG.ROWS - 2, vCorr[1]], [CONFIG.ROWS - 2, vCorr[2]]
    ];

    gameState.exits.forEach(([er, ec]) => {
        if (er > 0 && er < CONFIG.ROWS - 1 && ec > 0 && ec < CONFIG.COLS - 1) {
            gameState.maze[er][ec] = TILE.EXIT;
            // Clear around exits
            for (let dr = -1; dr <= 1; dr++) {
                for (let dc = -1; dc <= 1; dc++) {
                    const nr = er + dr;
                    const nc = ec + dc;
                    if (nr > 0 && nr < CONFIG.ROWS - 1 && nc > 0 && nc < CONFIG.COLS - 1) {
                        if (gameState.maze[nr][nc] === TILE.WALL) {
                            gameState.maze[nr][nc] = TILE.CORRIDOR;
                        }
                    }
                }
            }
        }
    });

    // Add more mid-edge and interior exits to reduce bottlenecks
    const extraExits = [
        [Math.floor(CONFIG.ROWS / 2), 1],
        [Math.floor(CONFIG.ROWS / 2), CONFIG.COLS - 2],
        [1, Math.floor(CONFIG.COLS / 2)],
        [CONFIG.ROWS - 2, Math.floor(CONFIG.COLS / 2)],
        [hCorr[0], vCorr[1]],
        [hCorr[1], vCorr[1]]
    ];
    extraExits.forEach(([er, ec]) => {
        if (er > 0 && er < CONFIG.ROWS - 1 && ec > 0 && ec < CONFIG.COLS - 1) {
            gameState.exits.push([er, ec]);
            gameState.maze[er][ec] = TILE.EXIT;
        }
    });

    // Sprinklers: grid on corridors
    gameState.sprinklers = [];
    for (let r = 2; r < CONFIG.ROWS - 2; r += CONFIG.SPRINKLER_SPACING) {
        for (let c = 2; c < CONFIG.COLS - 2; c += CONFIG.SPRINKLER_SPACING) {
            if (gameState.maze[r][c] === TILE.CORRIDOR) {
                gameState.sprinklers.push({ row: r, col: c });
            }
        }
    }

    // Generate people
    gameState.people = [];
    const spawns = [];

    for (let r = 2; r < CONFIG.ROWS - 2; r++) {
        for (let c = 2; c < CONFIG.COLS - 2; c++) {
            const tile = gameState.maze[r][c];
            if (tile === TILE.CARPET || tile === TILE.FLOOR || tile === TILE.CORRIDOR) {
                spawns.push([r, c]);
            }
        }
    }

    // Shuffle spawns
    for (let i = spawns.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [spawns[i], spawns[j]] = [spawns[j], spawns[i]];
    }

    const states = [STATE.WORKING, STATE.HEADPHONES];
    const corridorSpawns = spawns.filter(([r, c]) => gameState.maze[r][c] === TILE.CORRIDOR);

    // Wardens
    for (let i = 0; i < Math.min(CONFIG.NUM_WARDENS, corridorSpawns.length); i++) {
        const [r, c] = corridorSpawns[i];
    gameState.people.push({
        id: i,
        row: r,
        col: c,
        state: STATE.WARDEN,
        isWarden: true,
        health: 100,
        alive: true,
        escaped: false,
        awareness: 1,
        stuckTimer: 0
    });
}

    // Regular people
    for (let i = CONFIG.NUM_WARDENS; i < Math.min(CONFIG.TOTAL_PEOPLE, spawns.length); i++) {
        const [r, c] = spawns[i];
        gameState.people.push({
            id: i,
            row: r,
            col: c,
            state: states[Math.floor(Math.random() * states.length)],
            isWarden: false,
            health: 100,
            alive: true,
            escaped: false,
            awareness: 0,
            stuckTimer: 0
        });
    }

    // Generate sensors, stratified across the map to avoid clustering
    gameState.sensors = [];
    const sensorTypes = ['temperature', 'smoke', 'co', 'motion'];
    const bucketsR = 5;
    const bucketsC = 5;
    const bucketHeight = Math.floor(CONFIG.ROWS / bucketsR);
    const bucketWidth = Math.floor(CONFIG.COLS / bucketsC);

    let sensorId = 0;
    const leftover = [];

    for (let br = 0; br < bucketsR; br++) {
        for (let bc = 0; bc < bucketsC; bc++) {
            if (sensorId >= CONFIG.NUM_SENSORS) break;
            const rMin = br * bucketHeight;
            const rMax = (br === bucketsR - 1) ? CONFIG.ROWS - 1 : (br + 1) * bucketHeight;
            const cMin = bc * bucketWidth;
            const cMax = (bc === bucketsC - 1) ? CONFIG.COLS - 1 : (bc + 1) * bucketWidth;

            const candidates = spawns.filter(([r, c]) =>
                r >= rMin && r < rMax && c >= cMin && c < cMax &&
                gameState.maze[r][c] !== TILE.EXIT
            );

            if (candidates.length === 0) continue;

            const pick = candidates[Math.floor(Math.random() * candidates.length)];
            if (pick) {
                const type = sensorTypes[sensorId % sensorTypes.length];
                gameState.sensors.push({
                    id: sensorId,
                    row: pick[0],
                    col: pick[1],
                    type,
                    value: type === 'temperature' ? 22 : 0,
                    triggered: false,
                    health: 100
                });
                sensorId++;
            }

            // Collect extras to fill remaining slots if needed
            candidates.slice(1).forEach(item => leftover.push(item));
        }
    }

    // If we still need sensors, fill from leftovers
    for (let i = leftover.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [leftover[i], leftover[j]] = [leftover[j], leftover[i]];
    }

    for (const [r, c] of leftover) {
        if (sensorId >= CONFIG.NUM_SENSORS) break;
        const type = sensorTypes[sensorId % sensorTypes.length];
        gameState.sensors.push({
            id: sensorId,
            row: r,
            col: c,
            type,
            value: type === 'temperature' ? 22 : 0,
            triggered: false,
            health: 100
        });
        sensorId++;
    }

    setDefaultPOV();

    // Populate furniture (desks) in office tiles
    populateFurniture();
}

async function initGame() {
    // Reset state
    gameState.fires = [];
    gameState.bombs = [];
    gameState.floods = [];
    gameState.debris = [];
    gameState.earthquakeActive = false;
    gameState.earthquakeEpicenter = null;
    gameState.earthquakeTimer = 0;
    gameState.smoke = {};
    gameState.predictions = [];
    gameState.paused = false;
    gameState.speed = 1.0;
    gameState.time = 0;
    gameState.startTime = Date.now();
    gameState.shake = { x: 0, y: 0, intensity: 0 };
    gameState.selectedPersonId = null;
    gameState.povDimEnabled = false;
    gameState.pythonTelemetry = { loading: false, error: '', stats: null, lastRun: null };
    gameState.stats.escaped = 0;
    gameState.stats.deaths = 0;
    gameState.stats.alarmActive = false;
    gameState.neural.confidence = 0;
    gameState.rl.decisions = 0;
    gameState.rl.avgReward = 0;
    gameState.rlOverlay = { arrows: [], log: [] };
    // Backend sim reset
    if (CONFIG.USE_BACKEND) {
        try {
            await fetch(CONFIG.COMMAND_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ reset: true })
            });
        } catch (err) {
            console.warn('Backend reset failed (proceeding local fallback):', err);
        }
    }

    // Fetch or generate state
    await fetchInitialState();

    // Hide loading, show game
    setTimeout(() => {
        document.getElementById('loading-screen').style.display = 'none';
        document.getElementById('game-container').classList.remove('hidden');
        document.getElementById('victory-screen').classList.add('hidden');

        // Clear chat
        document.getElementById('chat-messages').innerHTML = '';
        addChatMessage('system', 'World loaded successfully!');
        addChatMessage('system', 'Use 1-6 to pick Fire/Bomb/Quake/Flood/Sensor/POV, then click');
        addChatMessage('system', 'POV auto-locks to the first survivor; press 6 and click someone to swap');
        addChatMessage('system', 'Goal: ZERO deaths — evacuate everyone safely');

        gameState.startTime = Date.now();
    }, 2000);
}

let lastTime = 0;

function gameLoop(currentTime) {
    const dt = Math.min((currentTime - lastTime) / 1000, 0.1) * gameState.speed;
    lastTime = currentTime;

    updateSimulation(dt);
    render();
    updateUI();

    requestAnimationFrame(gameLoop);
}

// Start the game
document.addEventListener('DOMContentLoaded', async () => {
    initCanvas();
    setupEventListeners();
    await initGame();

    // Start backend polling if enabled
    if (CONFIG.USE_BACKEND) {
        setInterval(pollBackendState, 500);
    }

    // Start game loop
    requestAnimationFrame(gameLoop);
});
