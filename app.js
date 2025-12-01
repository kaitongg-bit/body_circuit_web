/**
 * Body Circuit Band - Main Application
 * Web version using TensorFlow.js MoveNet for pose detection
 */

import { PoseDetector, SoloMode } from './poseDetection.js';
import { CircuitDetector } from './circuitDetector.js';
import { AudioController } from './audioController.js';
import { VisualFeedback } from './visualFeedback.js';

class BodyCircuitBandApp {
    constructor() {
        // DOM elements
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.loadingText = document.getElementById('loadingText');
        this.statusDot = document.getElementById('statusDot');
        this.appStatus = document.getElementById('appStatus');

        // Status panel elements
        this.circuitStatus = document.getElementById('circuitStatus');
        this.avgDistanceEl = document.getElementById('avgDistance');
        this.stabilityEl = document.getElementById('stability');
        this.volumeSection = document.getElementById('volumeSection');
        this.volumeBar = document.getElementById('volumeBar');
        this.volumePercent = document.getElementById('volumePercent');
        this.layerInfo = document.getElementById('layerInfo');

        // Core modules
        this.poseDetector = new PoseDetector();
        this.soloMode = new SoloMode();
        this.circuitDetector = new CircuitDetector(0.30, 10);
        this.audioController = new AudioController();
        this.visualFeedback = new VisualFeedback(this.canvas);

        // State
        this.isRunning = false;
        this.stream = null;
        this.animationFrameId = null;
        this.frameCount = 0;

        // Bind event listeners
        this.startBtn.addEventListener('click', () => this.start());
        this.stopBtn.addEventListener('click', () => this.stop());
    }

    /**
     * Show loading overlay
     */
    showLoading(message) {
        this.loadingText.textContent = message;
        this.loadingOverlay.classList.add('active');
    }

    /**
     * Hide loading overlay
     */
    hideLoading() {
        this.loadingOverlay.classList.remove('active');
    }

    /**
     * Update status indicator
     */
    updateStatus(message, isActive = false) {
        this.appStatus.textContent = message;
        if (isActive) {
            this.statusDot.classList.add('active');
        } else {
            this.statusDot.classList.remove('active');
        }
    }

    /**
     * Initialize camera
     */
    async initCamera() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'user'
                },
                audio: false
            });

            this.video.srcObject = this.stream;

            // Wait for video to be ready
            await new Promise((resolve) => {
                this.video.onloadedmetadata = () => {
                    resolve();
                };
            });

            // Set canvas size to match video
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;

            console.log('✅ Camera initialized');
            return true;
        } catch (error) {
            console.error('❌ Camera initialization failed:', error);
            alert('Failed to access camera. Please ensure camera permissions are granted.');
            return false;
        }
    }

    /**
     * Start the application
     */
    async start() {
        if (this.isRunning) return;

        try {
            this.startBtn.disabled = true;
            this.showLoading('Initializing camera...');

            // Initialize camera
            const cameraOk = await this.initCamera();
            if (!cameraOk) {
                this.hideLoading();
                this.startBtn.disabled = false;
                return;
            }

            // Load pose detection model
            this.showLoading('Loading pose detection model...');
            await this.poseDetector.initialize();

            // Load audio files
            this.showLoading('Loading audio files...');
            await this.audioController.initialize({
                drum: 'audio_samples_v2/drum.wav',
                bass: 'audio_samples_v2/bass.wav',
                harmony: 'audio_samples_v2/harmony.wav'
            });

            this.hideLoading();
            this.isRunning = true;
            this.stopBtn.disabled = false;
            this.updateStatus('Running - Detecting poses...', true);

            // Start detection loop
            this.detectLoop();

            console.log('🚀 Application started');
        } catch (error) {
            console.error('❌ Failed to start application:', error);
            alert('Failed to start application: ' + error.message);
            this.hideLoading();
            this.startBtn.disabled = false;
        }
    }

    /**
     * Stop the application
     */
    stop() {
        if (!this.isRunning) return;

        this.isRunning = false;

        // Stop animation loop
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }

        // Stop audio
        this.audioController.stopPlayback();

        // Stop camera
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }

        // Clear canvas
        this.visualFeedback.clear();

        // Reset UI
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        this.updateStatus('Stopped', false);
        this.circuitStatus.textContent = 'CIRCUIT OPEN';
        this.circuitStatus.className = 'circuit-status open';
        this.volumeSection.style.display = 'none';

        // Reset modules
        this.soloMode.reset();
        this.circuitDetector.reset();

        console.log('🛑 Application stopped');
    }

    /**
     * Main detection loop
     */
    async detectLoop() {
        if (!this.isRunning) return;

        this.frameCount++;

        try {
            // Detect poses
            const poses = await this.poseDetector.detectPoses(this.video);
            const person = this.poseDetector.extractPerson(
                poses,
                this.video.videoWidth,
                this.video.videoHeight
            );

            // Clear canvas
            this.visualFeedback.clear();

            if (person) {
                // Create virtual persons (solo mode)
                const { persons, stability } = this.soloMode.createVirtualPersons(person);

                // Update circuit detector
                const { state, avgDistance, distances, stateChanged } =
                    this.circuitDetector.update(persons);

                // Handle state changes
                if (stateChanged) {
                    if (state) {
                        console.log('🔌 Circuit changed from open to closed - starting music!');
                        this.audioController.startPlayback();
                    } else {
                        console.log('🔓 Circuit changed from closed to open - stopping music!');
                        this.audioController.stopPlayback();
                    }
                }

                // Update audio volume if circuit is closed
                let volumeInfo = { volume: 0, layers: 0 };
                if (state) {
                    volumeInfo = this.audioController.setVolumeLayered(
                        avgDistance,
                        stability,
                        0.30
                    );
                }

                // Draw visual feedback
                persons.forEach(p => {
                    this.visualFeedback.drawPersonLandmarks(
                        p,
                        this.canvas.width,
                        this.canvas.height
                    );
                });

                this.visualFeedback.drawConnections(
                    persons,
                    distances,
                    state,
                    this.canvas.width,
                    this.canvas.height,
                    this.circuitDetector.distanceThreshold
                );

                this.visualFeedback.drawStabilityIndicator(
                    stability,
                    this.canvas.width
                );

                // Update UI
                this.updateUI(state, avgDistance, stability, volumeInfo);

                // Debug logging every 30 frames
                if (this.frameCount % 30 === 0) {
                    const handsRaised = this.poseDetector.isHandsRaised(person);
                    console.log(`Frame ${this.frameCount}: Person detected, hands_raised=${handsRaised}`);
                    console.log(`   Stability: ${stability.toFixed(2)} (1.0=still, 0.0=unstable)`);
                    console.log(`   Distance: A→B=${distances[0].toFixed(3)}, B→C=${distances[1].toFixed(3)}, C→A=${distances[2].toFixed(3)}`);
                    console.log(`   Avg distance: ${avgDistance.toFixed(3)}, Circuit: ${state ? 'closed' : 'open'}`);
                }
            } else {
                // No person detected
                this.visualFeedback.drawNoPersonMessage(
                    this.canvas.width,
                    this.canvas.height
                );

                // Update UI
                this.circuitStatus.textContent = 'CIRCUIT OPEN';
                this.circuitStatus.className = 'circuit-status open';
                this.avgDistanceEl.textContent = '--';
                this.stabilityEl.textContent = '--';
                this.volumeSection.style.display = 'none';

                if (this.frameCount % 30 === 0) {
                    console.log(`Frame ${this.frameCount}: No person detected`);
                }
            }
        } catch (error) {
            console.error('Error in detection loop:', error);
        }

        // Schedule next frame
        this.animationFrameId = requestAnimationFrame(() => this.detectLoop());
    }

    /**
     * Update UI elements
     */
    updateUI(circuitClosed, avgDistance, stability, volumeInfo) {
        // Circuit status
        if (circuitClosed) {
            this.circuitStatus.textContent = 'CIRCUIT CLOSED - MUSIC PLAYING';
            this.circuitStatus.className = 'circuit-status closed';
            this.volumeSection.style.display = 'block';
        } else {
            this.circuitStatus.textContent = 'CIRCUIT OPEN';
            this.circuitStatus.className = 'circuit-status open';
            this.volumeSection.style.display = 'none';
        }

        // Distance and stability
        this.avgDistanceEl.textContent = avgDistance.toFixed(4);
        this.stabilityEl.textContent = `${(stability * 100).toFixed(0)}%`;

        // Volume and layers
        if (circuitClosed) {
            const volumePercent = (volumeInfo.volume * 100).toFixed(0);
            this.volumePercent.textContent = `${volumePercent}%`;
            this.volumeBar.style.width = `${volumePercent}%`;

            const layerNames = {
                1: 'Drums',
                2: 'Drums, Bass',
                3: 'Drums, Bass, Harmony'
            };
            this.layerInfo.textContent = `Layers: ${volumeInfo.layers} (${layerNames[volumeInfo.layers] || 'None'})`;
        }
    }

    /**
     * Cleanup resources
     */
    cleanup() {
        this.stop();
        this.poseDetector.dispose();
        this.audioController.cleanup();
    }
}

// Initialize app when DOM is ready
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new BodyCircuitBandApp();
    console.log('🎵 Body Circuit Band initialized');
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (app) {
        app.cleanup();
    }
});
