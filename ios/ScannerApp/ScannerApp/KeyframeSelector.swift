import ARKit
import Foundation

struct RecordingPolicy {
    let minimumFrameInterval: TimeInterval = 0.20
    let translationThresholdMeters: Float = 0.02
    let rotationThresholdRadians: Float = 3.0 * Float.pi / 180.0
    var rotationThresholdDegrees: Double {
        Double(rotationThresholdRadians * 180.0 / .pi)
    }

    var metadata: RecordingPolicyMetadata {
        RecordingPolicyMetadata(
            minimumFrameIntervalSeconds: minimumFrameInterval,
            translationThresholdMeters: Double(translationThresholdMeters),
            rotationThresholdDegrees: rotationThresholdDegrees
        )
    }
}

final class KeyframeSelector {
    let policy: RecordingPolicy
    private var previousTransform: simd_float4x4?
    private var previousTimestamp: TimeInterval?

    init(policy: RecordingPolicy = RecordingPolicy()) {
        self.policy = policy
    }

    func reset() {
        previousTransform = nil
        previousTimestamp = nil
    }

    func shouldCapture(
        timestamp: TimeInterval,
        transform: simd_float4x4,
        trackingIsNormal: Bool,
        hasDepth: Bool,
        hasConfidence: Bool
    ) -> Bool {
        guard trackingIsNormal, hasDepth, hasConfidence else { return false }

        guard let previousTransform, let previousTimestamp else {
            self.previousTransform = transform
            self.previousTimestamp = timestamp
            return true
        }

        guard timestamp - previousTimestamp >= policy.minimumFrameInterval else {
            return false
        }

        let relativeTransform = simd_inverse(previousTransform) * transform
        let translation = SIMD3<Float>(
            relativeTransform.columns.3.x,
            relativeTransform.columns.3.y,
            relativeTransform.columns.3.z
        )
        let rotation = rotationAngle(of: relativeTransform)

        guard simd_length(translation) >= policy.translationThresholdMeters ||
                rotation >= policy.rotationThresholdRadians else {
            return false
        }

        self.previousTransform = transform
        self.previousTimestamp = timestamp
        return true
    }

    private func rotationAngle(of transform: simd_float4x4) -> Float {
        let trace = transform.columns.0.x + transform.columns.1.y + transform.columns.2.z
        let cosine = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
        return acos(cosine)
    }
}
