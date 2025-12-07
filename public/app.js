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
    TOTAL_PEOPLE: 60,
    NUM_WARDENS: 4,
    NUM_SENSORS: 25,
    API_URL: '/api/simulate'
};

// Tile types
const TILE = {
    FLOOR: 0,
    WALL: 1,
    EXIT: 2,
    DOOR: 3,
    CORRIDOR: 4,
    CARPET: 5
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

    // Hazards
    FIRE_CORE: '#ffffc8',
    FIRE_MID: '#ffdc50',
    FIRE_OUTER: '#ff6414',
    SMOKE: '#464650',

    // Pheromones
    SAFE_PHEROMONE: 'rgba(0, 255, 136, 0.3)',
    DANGER_PHEROMONE: 'rgba(255, 68, 68, 0.4)',
    PREDICTION: 'rgba(192, 101, 255, 0.4)',

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
    smoke: {},
    predictions: [],

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

    // Selected tool
    selectedTool: 'fire',

    // Screen shake
    shake: { x: 0, y: 0, intensity: 0 }
};

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
    tileSize = Math.max(4, Math.min(maxTileWidth, maxTileHeight, 14));

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
            break;

        case TILE.WALL:
            ctx.fillStyle = COLORS.WALL;
            ctx.fillRect(x, y, tileSize, tileSize);
            ctx.fillStyle = COLORS.WALL_HIGHLIGHT;
            ctx.fillRect(x, y, tileSize, 2);
            ctx.fillRect(x, y, 2, tileSize);
            break;

        case TILE.CORRIDOR:
            ctx.fillStyle = COLORS.CORRIDOR;
            ctx.fillRect(x, y, tileSize, tileSize);
            break;

        case TILE.CARPET:
            ctx.fillStyle = COLORS.CARPET;
            ctx.fillRect(x, y, tileSize, tileSize);
            break;

        case TILE.EXIT:
            const glow = Math.sin(gameState.time * 4) * 0.3 + 0.7;
            ctx.fillStyle = COLORS.EXIT;
            ctx.fillRect(x, y, tileSize, tileSize);
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1;
            ctx.strokeRect(x + 1, y + 1, tileSize - 2, tileSize - 2);
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

function drawPheromones() {
    if (!gameState.showPheromones) return;

    ctx.save();

    // Draw safe pheromone trails
    for (let r = 0; r < CONFIG.ROWS; r++) {
        for (let c = 0; c < CONFIG.COLS; c++) {
            const safe = gameState.neural.safePheromone;
            const danger = gameState.neural.dangerPheromone;

            if (safe > 0.3) {
                const x = c * tileSize + (gameState.shake?.x || 0);
                const y = r * tileSize + (gameState.shake?.y || 0);
                ctx.fillStyle = COLORS.SAFE_PHEROMONE;
                ctx.fillRect(x, y, tileSize, tileSize);
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

function render() {
    if (!ctx || !canvas) return;

    ctx.clearRect(0, 0, canvasWidth, canvasHeight);

    // Background
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

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

    // Draw sensors
    if (gameState.showSensors) {
        gameState.sensors.forEach(drawSensor);
    }

    // Draw people (sorted by Y for proper overlapping)
    [...gameState.people]
        .sort((a, b) => a.row - b.row)
        .forEach(drawPerson);

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

    // Update shake
    gameState.shake.intensity *= 0.9;
    if (gameState.shake.intensity > 0.01) {
        (gameState.shake?.x || 0) = (Math.random() - 0.5) * gameState.shake.intensity * 10;
        (gameState.shake?.y || 0) = (Math.random() - 0.5) * gameState.shake.intensity * 10;
    } else {
        (gameState.shake?.x || 0) = 0;
        (gameState.shake?.y || 0) = 0;
    }

    // Update fires
    updateFires(dt);

    // Update people
    updatePeople(dt);

    // Update sensors
    updateSensors(dt);

    // Update neural predictions
    updateNeuralPredictions(dt);

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

        // Check death
        if (person.health <= 0) {
            person.alive = false;
            addChatMessage('death', `Person ${person.id} has perished!`);
            return;
        }

        // Check escape
        if (gameState.maze[person.row]?.[person.col] === TILE.EXIT) {
            person.escaped = true;
            addChatMessage('escape', `Person ${person.id} escaped safely!`);
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
            moveTowardsExit(person, dt);
        }
    });
}

function moveTowardsExit(person, dt) {
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

    // Simple pathfinding - move towards exit
    const dr = nearestExit[0] - person.row;
    const dc = nearestExit[1] - person.col;

    // Check for obstacles
    const moves = [];
    if (dr > 0) moves.push([1, 0]);
    if (dr < 0) moves.push([-1, 0]);
    if (dc > 0) moves.push([0, 1]);
    if (dc < 0) moves.push([0, -1]);

    // Add random variation
    if (Math.random() < 0.2) {
        moves.push([0, 0]); // Sometimes stay still
    }

    // Try each move
    for (const [mr, mc] of moves) {
        const newRow = person.row + mr;
        const newCol = person.col + mc;

        // Check bounds
        if (newRow < 0 || newRow >= CONFIG.ROWS || newCol < 0 || newCol >= CONFIG.COLS) continue;

        // Check tile walkability
        const tile = gameState.maze[newRow]?.[newCol];
        if (tile === TILE.WALL) continue;

        // Check for fire
        if (gameState.fires.some(f => f.row === newRow && f.col === newCol)) continue;

        // Move with probability based on speed
        const speed = person.isWarden ? 1.2 : (person.state === STATE.PANICKING ? 1.3 : 1.0);
        if (Math.random() < 0.1 * speed * dt * 60) {
            person.row = newRow;
            person.col = newCol;
        }
        break;
    }
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

    const dangerLevel = Math.min(gameState.fires.length * 10 + Object.keys(gameState.smoke).length / 10, 100);
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

    // Pause overlay
    document.getElementById('pause-overlay').classList.toggle('hidden', !gameState.paused);
}

// ═══════════════════════════════════════════════════════════════════════════
// GAME ACTIONS
// ═══════════════════════════════════════════════════════════════════════════

function addFire(row, col) {
    if (row < 1 || row >= CONFIG.ROWS - 1 || col < 1 || col >= CONFIG.COLS - 1) return;
    if (gameState.maze[row]?.[col] === TILE.WALL) return;
    if (gameState.fires.some(f => f.row === row && f.col === col)) return;

    gameState.fires.push({ row, col, age: 0, intensity: 1 });
    gameState.shake.intensity = 2;
    addChatMessage('fire', `Fire started at (${row}, ${col})!`);
}

function triggerAlarm() {
    if (gameState.stats.alarmActive) return;

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

function showVictoryScreen() {
    gameState.paused = true;

    const victoryScreen = document.getElementById('victory-screen');
    victoryScreen.classList.remove('hidden');

    // Update stats
    document.getElementById('victory-escaped').textContent = gameState.stats.escaped;
    document.getElementById('victory-deaths').textContent = gameState.stats.deaths;

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

        if (gameState.selectedTool === 'fire') {
            addFire(row, col);
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
    try {
        const response = await fetch(CONFIG.API_URL);
        const data = await response.json();

        if (data.success) {
            gameState.maze = data.maze;
            gameState.exits = data.exits;
            gameState.people = data.people;
            gameState.sensors = data.sensors;
            gameState.stats.total = data.config.totalPeople;
            return true;
        }
    } catch (error) {
        console.warn('API not available, generating locally:', error);
    }

    // Generate locally if API fails
    generateLocalState();
    return true;
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
            awareness: 1
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
            awareness: 0
        });
    }

    // Generate sensors
    gameState.sensors = [];
    const sensorTypes = ['temperature', 'smoke', 'co', 'motion'];
    let sensorId = 0;

    for (let r = 3; r < CONFIG.ROWS - 3; r += 4) {
        for (let c = 3; c < CONFIG.COLS - 3; c += 4) {
            if (gameState.maze[r][c] !== TILE.WALL && sensorId < CONFIG.NUM_SENSORS) {
                gameState.sensors.push({
                    id: sensorId,
                    row: r,
                    col: c,
                    type: sensorTypes[sensorId % sensorTypes.length],
                    value: sensorTypes[sensorId % sensorTypes.length] === 'temperature' ? 22 : 0,
                    triggered: false,
                    health: 100
                });
                sensorId++;
            }
        }
    }
}

async function initGame() {
    // Reset state
    gameState.fires = [];
    gameState.smoke = {};
    gameState.predictions = [];
    gameState.paused = false;
    gameState.speed = 1.0;
    gameState.time = 0;
    gameState.startTime = Date.now();
    gameState.stats.escaped = 0;
    gameState.stats.deaths = 0;
    gameState.stats.alarmActive = false;
    gameState.neural.confidence = 0;
    gameState.rl.decisions = 0;
    gameState.rl.avgReward = 0;

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
        addChatMessage('system', 'Click anywhere to place fire blocks');

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

    // Start game loop
    requestAnimationFrame(gameLoop);
});
