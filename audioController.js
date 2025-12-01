/**
 * Audio Controller Module
 * Manages layered audio playback using Web Audio API
 */

export class AudioController {
    constructor() {
        this.audioContext = null;
        this.tracks = {
            drum: null,
            bass: null,
            harmony: null
        };
        this.gainNodes = {
            drum: null,
            bass: null,
            harmony: null
        };
        this.sources = {
            drum: null,
            bass: null,
            harmony: null
        };
        this.buffers = {
            drum: null,
            bass: null,
            harmony: null
        };
        this.isPlaying = false;
        this.currentLayer = 0;
        this.isLoaded = false;
    }

    /**
     * Initialize audio context and load audio files
     * @param {Object} paths - Audio file paths { drum, bass, harmony }
     */
    async initialize(paths) {
        try {
            // Create audio context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();

            // Load all audio files
            await Promise.all([
                this.loadAudio('drum', paths.drum),
                this.loadAudio('bass', paths.bass),
                this.loadAudio('harmony', paths.harmony)
            ]);

            // Create gain nodes for volume control
            this.gainNodes.drum = this.audioContext.createGain();
            this.gainNodes.bass = this.audioContext.createGain();
            this.gainNodes.harmony = this.audioContext.createGain();

            // Connect gain nodes to destination
            this.gainNodes.drum.connect(this.audioContext.destination);
            this.gainNodes.bass.connect(this.audioContext.destination);
            this.gainNodes.harmony.connect(this.audioContext.destination);

            // Set initial volumes to 0
            this.gainNodes.drum.gain.value = 0;
            this.gainNodes.bass.gain.value = 0;
            this.gainNodes.harmony.gain.value = 0;

            this.isLoaded = true;
            console.log('✅ Audio system initialized successfully');
            return true;
        } catch (error) {
            console.error('❌ Failed to initialize audio system:', error);
            throw error;
        }
    }

    /**
     * Load an audio file
     * @param {string} name - Track name
     * @param {string} path - File path
     */
    async loadAudio(name, path) {
        try {
            const response = await fetch(path);
            const arrayBuffer = await response.arrayBuffer();
            const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
            this.buffers[name] = audioBuffer;
            console.log(`✅ Loaded ${name} audio`);
        } catch (error) {
            console.error(`❌ Failed to load ${name} audio:`, error);
            throw error;
        }
    }

    /**
     * Start playback of all tracks
     */
    startPlayback() {
        if (!this.isLoaded || this.isPlaying) {
            return;
        }

        // Resume audio context if suspended (browser autoplay policy)
        if (this.audioContext.state === 'suspended') {
            this.audioContext.resume();
        }

        // Create and start sources for all tracks
        ['drum', 'bass', 'harmony'].forEach(trackName => {
            const source = this.audioContext.createBufferSource();
            source.buffer = this.buffers[trackName];
            source.loop = true;
            source.connect(this.gainNodes[trackName]);
            source.start(0);
            this.sources[trackName] = source;
        });

        this.isPlaying = true;
        this.currentLayer = 0;
        console.log('🎵 Music system started (layered mode)');
    }

    /**
     * Stop playback of all tracks
     */
    stopPlayback() {
        if (!this.isPlaying) {
            return;
        }

        // Stop all sources
        ['drum', 'bass', 'harmony'].forEach(trackName => {
            if (this.sources[trackName]) {
                try {
                    this.sources[trackName].stop();
                } catch (e) {
                    // Source might already be stopped
                }
                this.sources[trackName] = null;
            }
            // Reset gain to 0
            if (this.gainNodes[trackName]) {
                this.gainNodes[trackName].gain.value = 0;
            }
        });

        this.isPlaying = false;
        this.currentLayer = 0;
        console.log('🔇 Music stopped');
    }

    /**
     * Set volume based on layered system
     * @param {number} avgDistance - Average distance between hands
     * @param {number} stability - Pose stability (0.0 to 1.0)
     * @param {number} maxDistance - Maximum distance threshold
     */
    setVolumeLayered(avgDistance, stability, maxDistance = 0.30) {
        if (!this.isPlaying || !this.isLoaded) {
            return;
        }

        // Calculate base volume from distance
        const baseVolume = Math.max(0.0, Math.min(1.0, 1.0 - (avgDistance / maxDistance)));

        // Determine layer based on stability
        let newLayer = 0;
        if (stability > 0.75) {
            newLayer = 3; // Very stable: all 3 layers
        } else if (stability > 0.45) {
            newLayer = 2; // Moderately stable: 2 layers
        } else {
            newLayer = 1; // Slight movement: 1 layer
        }

        // Log layer changes
        if (newLayer !== this.currentLayer) {
            const layerNames = {
                1: 'Drums',
                2: 'Drums, Bass',
                3: 'Drums, Bass, Harmony'
            };
            console.log(`🎶 Music intensity changed: ${this.currentLayer} layers → ${newLayer} layers (${layerNames[newLayer]})`);
            this.currentLayer = newLayer;
        }

        // Set volumes for each track
        const fadeTime = 0.1; // Smooth fade transition
        const currentTime = this.audioContext.currentTime;

        // Drum (Layer 1)
        if (newLayer >= 1) {
            this.gainNodes.drum.gain.linearRampToValueAtTime(baseVolume * 0.8, currentTime + fadeTime);
        } else {
            this.gainNodes.drum.gain.linearRampToValueAtTime(0, currentTime + fadeTime);
        }

        // Bass (Layer 2)
        if (newLayer >= 2) {
            this.gainNodes.bass.gain.linearRampToValueAtTime(baseVolume * 0.7, currentTime + fadeTime);
        } else {
            this.gainNodes.bass.gain.linearRampToValueAtTime(0, currentTime + fadeTime);
        }

        // Harmony (Layer 3)
        if (newLayer >= 3) {
            this.gainNodes.harmony.gain.linearRampToValueAtTime(baseVolume * 0.6, currentTime + fadeTime);
        } else {
            this.gainNodes.harmony.gain.linearRampToValueAtTime(0, currentTime + fadeTime);
        }

        return { volume: baseVolume, layers: newLayer };
    }

    /**
     * Get current layer count
     * @returns {number} Current layer count
     */
    getCurrentLayer() {
        return this.currentLayer;
    }

    /**
     * Cleanup resources
     */
    cleanup() {
        this.stopPlayback();
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
    }
}
