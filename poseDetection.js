/**
 * Pose Detection Module
 * Uses TensorFlow.js MoveNet for browser-based pose detection
 */

export class PoseDetector {
    constructor() {
        this.detector = null;
        this.model = null;
    }

    /**
     * Initialize the pose detection model
     */
    async initialize() {
        try {
            // Load MoveNet model (SinglePose Lightning for speed)
            this.model = poseDetection.SupportedModels.MoveNet;
            const detectorConfig = {
                modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING
            };

            this.detector = await poseDetection.createDetector(this.model, detectorConfig);
            console.log('✅ Pose detection model loaded successfully');
            return true;
        } catch (error) {
            console.error('❌ Failed to load pose detection model:', error);
            throw error;
        }
    }

    /**
     * Detect poses in a video frame
     * @param {HTMLVideoElement} video - Video element
     * @returns {Promise<Array>} Array of detected poses
     */
    async detectPoses(video) {
        if (!this.detector) {
            throw new Error('Detector not initialized');
        }

        try {
            const poses = await this.detector.estimatePoses(video);
            return poses;
        } catch (error) {
            console.error('Error detecting poses:', error);
            return [];
        }
    }

    /**
     * Extract person data from pose detection results
     * @param {Array} poses - Detected poses
     * @param {number} videoWidth - Video width
     * @param {number} videoHeight - Video height
     * @returns {Object|null} Person object or null
     */
    extractPerson(poses, videoWidth, videoHeight) {
        if (!poses || poses.length === 0) {
            return null;
        }

        const pose = poses[0]; // Get first detected person
        const keypoints = pose.keypoints;

        // MoveNet keypoint indices:
        // 5: left_shoulder, 6: right_shoulder
        // 9: left_wrist, 10: right_wrist
        const leftShoulder = keypoints[5];
        const rightShoulder = keypoints[6];
        const leftWrist = keypoints[9];
        const rightWrist = keypoints[10];

        // Check confidence scores (minimum 0.3)
        const minConfidence = 0.3;
        if (leftShoulder.score < minConfidence ||
            rightShoulder.score < minConfidence ||
            leftWrist.score < minConfidence ||
            rightWrist.score < minConfidence) {
            return null;
        }

        // Normalize coordinates to 0-1 range
        const person = {
            leftShoulder: {
                x: leftShoulder.x / videoWidth,
                y: leftShoulder.y / videoHeight
            },
            rightShoulder: {
                x: rightShoulder.x / videoWidth,
                y: rightShoulder.y / videoHeight
            },
            leftWrist: {
                x: leftWrist.x / videoWidth,
                y: leftWrist.y / videoHeight
            },
            rightWrist: {
                x: rightWrist.x / videoWidth,
                y: rightWrist.y / videoHeight
            },
            xCenter: (leftShoulder.x + rightShoulder.x) / 2 / videoWidth
        };

        return person;
    }

    /**
     * Check if hands are raised above shoulders
     * @param {Object} person - Person object
     * @returns {boolean} True if both hands are raised
     */
    isHandsRaised(person) {
        if (!person) return false;

        const leftRaised = person.leftWrist.y < person.leftShoulder.y;
        const rightRaised = person.rightWrist.y < person.rightShoulder.y;

        return leftRaised && rightRaised;
    }

    /**
     * Cleanup resources
     */
    dispose() {
        if (this.detector) {
            this.detector.dispose();
            this.detector = null;
        }
    }
}

/**
 * Solo Mode - Creates virtual persons from detected person
 */
export class SoloMode {
    constructor() {
        this.baseOffset = 0.12;
        this.wristHistory = [];
        this.historySize = 10;
    }

    /**
     * Calculate pose stability based on wrist position changes
     * @param {Object} person - Person object
     * @returns {number} Stability score (0.0 to 1.0)
     */
    calculateStability(person) {
        if (!person) return 0.5;

        const currentPos = {
            x: (person.leftWrist.x + person.rightWrist.x) / 2,
            y: (person.leftWrist.y + person.rightWrist.y) / 2
        };

        this.wristHistory.push(currentPos);

        if (this.wristHistory.length > this.historySize) {
            this.wristHistory.shift();
        }

        if (this.wristHistory.length < 3) {
            return 0.5;
        }

        // Calculate standard deviation
        const positions = this.wristHistory;
        const meanX = positions.reduce((sum, p) => sum + p.x, 0) / positions.length;
        const meanY = positions.reduce((sum, p) => sum + p.y, 0) / positions.length;

        const variance = positions.reduce((sum, p) => {
            return sum + Math.pow(p.x - meanX, 2) + Math.pow(p.y - meanY, 2);
        }, 0) / positions.length;

        const stdDev = Math.sqrt(variance);

        // Convert to stability score (0.0 = very unstable, 1.0 = completely still)
        const stability = Math.max(0.0, Math.min(1.0, 1.0 - (stdDev / 0.05)));

        return stability;
    }

    /**
     * Create 3 virtual persons from detected person
     * @param {Object} person - Detected person
     * @returns {Object} { persons: Array, stability: number }
     */
    createVirtualPersons(person) {
        if (!person) {
            return { persons: [], stability: 0 };
        }

        const stability = this.calculateStability(person);

        // Dynamic offset based on stability
        const dynamicOffset = this.baseOffset + (0.04 * (1.0 - stability));

        const offsets = [
            { x: -dynamicOffset, y: 0, id: 'A' },
            { x: 0, y: 0, id: 'B' },
            { x: dynamicOffset, y: 0, id: 'C' }
        ];

        const virtualPersons = offsets.map(offset => ({
            id: offset.id,
            xCenter: person.xCenter + offset.x,
            leftWrist: {
                x: person.leftWrist.x + offset.x,
                y: person.leftWrist.y + offset.y
            },
            rightWrist: {
                x: person.rightWrist.x + offset.x,
                y: person.rightWrist.y + offset.y
            },
            leftShoulder: {
                x: person.leftShoulder.x + offset.x,
                y: person.leftShoulder.y + offset.y
            },
            rightShoulder: {
                x: person.rightShoulder.x + offset.x,
                y: person.rightShoulder.y + offset.y
            }
        }));

        return { persons: virtualPersons, stability };
    }

    /**
     * Reset history (useful when restarting)
     */
    reset() {
        this.wristHistory = [];
    }
}
