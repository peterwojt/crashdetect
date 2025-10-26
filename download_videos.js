const https = require('https');
const fs = require('fs');
const path = require('path');

const MASTER_URL = 'https://trafficcams.bellevuewa.gov/traffic-edge/CCTV072L.stream/playlist.m3u8';
const BASE_URL = 'https://trafficcams.bellevuewa.gov/traffic-edge/CCTV072L.stream/';
const OUTPUT_DIR = 'traffic_cam_videos3/upload';
const PROCESSED_DIR = 'traffic_cam_videos3/processed';
const LOG_FILE = 'traffic_cam_videos3/download_log.csv';
const CHECK_INTERVAL_MS = 4000; // 4 seconds

const STOP_TIME = new Date('2026-10-21T03:00:00Z');

// Ensure directories exist
if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });
if (!fs.existsSync(PROCESSED_DIR)) fs.mkdirSync(PROCESSED_DIR, { recursive: true });

// Create CSV log file with header if not present
if (!fs.existsSync(LOG_FILE)) {
    fs.writeFileSync(LOG_FILE, 'filename,timestamp\n', 'utf8');
}

// Keep track of last 4 digits for already downloaded files
let downloadedSuffixes = new Set(
    fs.readdirSync(PROCESSED_DIR)
        .filter(f => f.endsWith('.ts'))
        .map(f => f.match(/(\d{4})\.ts$/)?.[1])
        .filter(Boolean)
);

async function mainLoop() {
    try {
        if (new Date() >= STOP_TIME) {
            console.log(`Stop time reached (${STOP_TIME.toISOString()}). Exiting loop.`);
            return;
        }
        const chunklistName = await fetchChunklistName(MASTER_URL);
        if (!chunklistName) return console.log('No chunklist found.');

        const chunklistUrl = BASE_URL + chunklistName;
        const tsFiles = await fetchTsFiles(chunklistUrl);

        if (tsFiles.length === 0) {
            console.log('No TS files found in chunklist.');
            return;
        }

        for (const tsFile of tsFiles) {
            const suffixMatch = tsFile.match(/(\d{4})\.ts$/);
            const suffix = suffixMatch ? suffixMatch[1] : null;

            if (!suffix || downloadedSuffixes.has(suffix)) continue;

            await downloadTsFile(BASE_URL + tsFile, tsFile, suffix);
            downloadedSuffixes.add(suffix);
        }

        cleanupUnprocessedFiles();
    } catch (err) {
        console.error('Error in loop:', err.message);
    } finally {
        setTimeout(mainLoop, CHECK_INTERVAL_MS);
    }
}

function fetchChunklistName(url) {
    return new Promise((resolve, reject) => {
        https.get(url, res => {
            let data = '';
            res.on('data', chunk => (data += chunk));
            res.on('end', () => {
                const match = data.match(/chunklist.*\.m3u8/g);
                resolve(match ? match[0] : null);
            });
        }).on('error', reject);
    });
}

function fetchTsFiles(url) {
    return new Promise((resolve, reject) => {
        https.get(url, res => {
            let data = '';
            res.on('data', chunk => (data += chunk));
            res.on('end', () => {
                const matches = data.match(/.*\.ts/g);
                resolve(matches || []);
            });
        }).on('error', reject);
    });
}

function downloadTsFile(url, filename, suffix) {
    return new Promise((resolve, reject) => {
        const filepath = path.join(OUTPUT_DIR, filename);
        const processedPath = path.join(PROCESSED_DIR, filename);
        const file = fs.createWriteStream(filepath);

        https.get(url, response => {
            response.pipe(file);

            file.on('finish', () => {
                file.close(() => {
                    console.log(`✅ Downloaded ${filename}`);

                    // Move file to processed folder
                    fs.rename(filepath, processedPath, err => {
                        if (err) {
                            console.error(`⚠️  Could not move ${filename}:`, err.message);
                        } else {
                            console.log(`📦 Moved ${filename} → processed/`);
                            logDownload(filename);
                        }
                        resolve();
                    });
                });
            });
        }).on('error', err => {
            fs.unlink(filepath, () => {});
            console.error(`❌ Failed ${filename}:`, err.message);
            reject(err);
        });
    });
}

function logDownload(filename) {
    const timestamp = new Date().toISOString();
    const line = `${filename},${timestamp}\n`;
    fs.appendFile(LOG_FILE, line, err => {
        if (err) console.error(`⚠️  Failed to write log for ${filename}:`, err.message);
        else console.log(`🕓 Logged ${filename} at ${timestamp}`);
    });
}


function cleanupUnprocessedFiles() {
    const items = fs.readdirSync(OUTPUT_DIR);
    for (const item of items) {
        const fullPath = path.join(OUTPUT_DIR, item);

        // Skip folders
        if (fs.statSync(fullPath).isDirectory()) {
            // skip crash folder entirely
            if (item.toLowerCase() === 'crash') continue;
            else continue; // don't delete subdirs either
        }

        if (item.endsWith('.ts')) {
            const processedPath = path.join(PROCESSED_DIR, item);
            const existsInProcessed = fs.existsSync(processedPath);

            if (!existsInProcessed) {
                try {
                    fs.unlinkSync(fullPath);
                    console.log(`🧹 Deleted leftover file: ${item}`);
                } catch (err) {
                    console.error(`⚠️  Failed to delete ${item}:`, err.message);
                }
            }
        }
    }
}

// Start loop
mainLoop();
