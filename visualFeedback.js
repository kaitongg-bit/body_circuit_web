/**
 * Visual Feedback Module
 * Renders pose landmarks, connections, and status information on canvas
 */

export class VisualFeedback {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');

        this.colors = {
            A: '#FF0000', // Red
            B: '#00FF00', // Green
            C: '#0000FF'  // Blue
        };
    }

    /**
     * Clear the canvas
     */
    clear() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }

    /**
     * Draw person landmarks
     * @param {Object} person - Person object
     * @param {number} width - Canvas width
     * @param {number} height - Canvas height
     */
    drawPersonLandmarks(person, width, height) {
        const color = this.colors[person.id] || '#FFFFFF';

        // Draw keypoints
        const points = [
            { name: 'left_wrist', pos: person.leftWrist },
            { name: 'right_wrist', pos: person.rightWrist },
            { name: 'left_shoulder', pos: person.leftShoulder },
            { name: 'right_shoulder', pos: person.rightShoulder }
        ];

        points.forEach(point => {
            const x = point.pos.x * width;
            const y = point.pos.y * height;

            // Draw filled circle
            this.ctx.fillStyle = color;
            this.ctx.beginPath();
            this.ctx.arc(x, y, 10, 0, 2 * Math.PI);
            this.ctx.fill();

            // Draw white border
            this.ctx.strokeStyle = '#FFFFFF';
            this.ctx.lineWidth = 2;
            this.ctx.beginPath();
            this.ctx.arc(x, y, 12, 0, 2 * Math.PI);
            this.ctx.stroke();
        });

        // Draw skeleton lines
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 3;

        // Shoulder line
        this.ctx.beginPath();
        this.ctx.moveTo(person.leftShoulder.x * width, person.leftShoulder.y * height);
        this.ctx.lineTo(person.rightShoulder.x * width, person.rightShoulder.y * height);
        this.ctx.stroke();

        // Left arm
        this.ctx.beginPath();
        this.ctx.moveTo(person.leftShoulder.x * width, person.leftShoulder.y * height);
        this.ctx.lineTo(person.leftWrist.x * width, person.leftWrist.y * height);
        this.ctx.stroke();

        // Right arm
        this.ctx.beginPath();
        this.ctx.moveTo(person.rightShoulder.x * width, person.rightShoulder.y * height);
        this.ctx.lineTo(person.rightWrist.x * width, person.rightWrist.y * height);
        this.ctx.stroke();

        // Draw person label
        const textX = person.xCenter * width;
        const textY = person.leftShoulder.y * height - 30;

        this.ctx.font = 'bold 24px Inter, sans-serif';
        this.ctx.fillStyle = color;
        this.ctx.textAlign = 'center';
        this.ctx.fillText(`Person ${person.id}`, textX, textY);

        // Draw hand status
        const leftRaised = person.leftWrist.y < person.leftShoulder.y;
        const rightRaised = person.rightWrist.y < person.rightShoulder.y;
        const handsRaised = leftRaised && rightRaised;

        this.ctx.font = '16px Inter, sans-serif';
        this.ctx.fillStyle = handsRaised ? '#10b981' : '#ef4444';
        this.ctx.fillText(handsRaised ? 'Hands Up' : 'Hands Down', textX, textY + 25);
    }

    /**
     * Draw connections between persons
     * @param {Array} persons - Array of 3 persons [A, B, C]
     * @param {Array} distances - Array of 3 distances
     * @param {boolean} circuitClosed - Circuit state
     * @param {number} width - Canvas width
     * @param {number} height - Canvas height
     * @param {number} threshold - Distance threshold
     */
    drawConnections(persons, distances, circuitClosed, width, height, threshold) {
        const [personA, personB, personC] = persons;
        const [d1, d2, d3] = distances;

        const connections = [
            { p1: personA.rightWrist, p2: personB.leftWrist, dist: d1, label: 'A→B' },
            { p1: personB.rightWrist, p2: personC.leftWrist, dist: d2, label: 'B→C' },
            { p1: personC.rightWrist, p2: personA.leftWrist, dist: d3, label: 'C→A' }
        ];

        connections.forEach(conn => {
            const x1 = conn.p1.x * width;
            const y1 = conn.p1.y * height;
            const x2 = conn.p2.x * width;
            const y2 = conn.p2.y * height;

            // Determine line color and thickness
            let color, lineWidth;
            if (circuitClosed) {
                color = '#10b981'; // Green - circuit closed
                lineWidth = 8;
            } else if (conn.dist < threshold) {
                color = '#f59e0b'; // Yellow - close but not closed
                lineWidth = 5;
            } else {
                color = '#ef4444'; // Red - too far
                lineWidth = 3;
            }

            // Draw connection line
            this.ctx.strokeStyle = color;
            this.ctx.lineWidth = lineWidth;
            this.ctx.beginPath();
            this.ctx.moveTo(x1, y1);
            this.ctx.lineTo(x2, y2);
            this.ctx.stroke();

            // Draw distance label
            const midX = (x1 + x2) / 2;
            const midY = (y1 + y2) / 2;

            this.ctx.font = 'bold 16px Inter, sans-serif';
            this.ctx.fillStyle = '#FFFFFF';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(`${conn.label}: ${conn.dist.toFixed(3)}`, midX, midY - 10);
        });
    }

    /**
     * Draw "No person detected" message
     * @param {number} width - Canvas width
     * @param {number} height - Canvas height
     */
    drawNoPersonMessage(width, height) {
        this.ctx.font = 'bold 36px Inter, sans-serif';
        this.ctx.fillStyle = '#ef4444';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('No person detected', width / 2, height / 2 - 30);

        this.ctx.font = '24px Inter, sans-serif';
        this.ctx.fillStyle = '#f59e0b';
        this.ctx.fillText('Stand in front of camera', width / 2, height / 2 + 20);
    }

    /**
     * Draw stability indicator
     * @param {number} stability - Stability score (0.0 to 1.0)
     * @param {number} width - Canvas width
     */
    drawStabilityIndicator(stability, width) {
        const barWidth = 200;
        const barHeight = 20;
        const x = width - barWidth - 20;
        const y = 20;

        // Background
        this.ctx.fillStyle = 'rgba(30, 41, 59, 0.8)';
        this.ctx.fillRect(x, y, barWidth, barHeight);

        // Fill based on stability
        const fillWidth = barWidth * stability;
        const gradient = this.ctx.createLinearGradient(x, y, x + barWidth, y);
        gradient.addColorStop(0, '#ef4444');
        gradient.addColorStop(0.5, '#f59e0b');
        gradient.addColorStop(1, '#10b981');

        this.ctx.fillStyle = gradient;
        this.ctx.fillRect(x, y, fillWidth, barHeight);

        // Border
        this.ctx.strokeStyle = '#FFFFFF';
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(x, y, barWidth, barHeight);

        // Label
        this.ctx.font = '14px Inter, sans-serif';
        this.ctx.fillStyle = '#FFFFFF';
        this.ctx.textAlign = 'right';
        this.ctx.fillText(`Stability: ${(stability * 100).toFixed(0)}%`, x - 10, y + 15);
    }
}
