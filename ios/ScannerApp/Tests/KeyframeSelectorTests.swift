import ARKit
import XCTest

@testable import ScannerApp

final class KeyframeSelectorTests: XCTestCase {
    private let policy = RecordingPolicy()

    func testFirstValidFrameIsCaptured() {
        let selector = KeyframeSelector(policy: policy)
        XCTAssertTrue(selector.shouldCapture(
            timestamp: 0,
            transform: matrix_identity_float4x4,
            trackingIsNormal: true,
            hasDepth: true,
            hasConfidence: true
        ))
    }

    func testNoMovementIsRejected() {
        let selector = KeyframeSelector(policy: policy)
        _ = selector.shouldCapture(timestamp: 0, transform: matrix_identity_float4x4, trackingIsNormal: true, hasDepth: true, hasConfidence: true)
        XCTAssertFalse(selector.shouldCapture(timestamp: 0.3, transform: matrix_identity_float4x4, trackingIsNormal: true, hasDepth: true, hasConfidence: true))
    }

    func testMinimumIntervalRejectsEvenLargeMovement() {
        let selector = KeyframeSelector(policy: policy)
        _ = selector.shouldCapture(timestamp: 0, transform: matrix_identity_float4x4, trackingIsNormal: true, hasDepth: true, hasConfidence: true)
        XCTAssertFalse(selector.shouldCapture(timestamp: 0.1, transform: translated(0.1), trackingIsNormal: true, hasDepth: true, hasConfidence: true))
    }

    func testTranslationThresholdCapturesAfterInterval() {
        let selector = KeyframeSelector(policy: policy)
        _ = selector.shouldCapture(timestamp: 0, transform: matrix_identity_float4x4, trackingIsNormal: true, hasDepth: true, hasConfidence: true)
        XCTAssertTrue(selector.shouldCapture(timestamp: 0.3, transform: translated(0.03), trackingIsNormal: true, hasDepth: true, hasConfidence: true))
    }

    func testRotationThresholdCapturesAfterInterval() {
        let selector = KeyframeSelector(policy: policy)
        _ = selector.shouldCapture(timestamp: 0, transform: matrix_identity_float4x4, trackingIsNormal: true, hasDepth: true, hasConfidence: true)
        XCTAssertTrue(selector.shouldCapture(timestamp: 0.3, transform: rotated(policy.rotationThresholdRadians * 1.5), trackingIsNormal: true, hasDepth: true, hasConfidence: true))
    }

    func testSmallMotionIsRejected() {
        let selector = KeyframeSelector(policy: policy)
        _ = selector.shouldCapture(timestamp: 0, transform: matrix_identity_float4x4, trackingIsNormal: true, hasDepth: true, hasConfidence: true)
        XCTAssertFalse(selector.shouldCapture(timestamp: 0.3, transform: translated(0.005), trackingIsNormal: true, hasDepth: true, hasConfidence: true))
    }

    func testNearlyIdenticalRotationIsStable() {
        let selector = KeyframeSelector(policy: policy)
        _ = selector.shouldCapture(timestamp: 0, transform: matrix_identity_float4x4, trackingIsNormal: true, hasDepth: true, hasConfidence: true)
        XCTAssertFalse(selector.shouldCapture(timestamp: 0.3, transform: rotated(0.000001), trackingIsNormal: true, hasDepth: true, hasConfidence: true))
    }

    private func translated(_ x: Float) -> simd_float4x4 {
        var transform = matrix_identity_float4x4
        transform.columns.3.x = x
        return transform
    }

    private func rotated(_ angle: Float) -> simd_float4x4 {
        let cosine = cos(angle)
        let sine = sin(angle)
        return simd_float4x4(
            SIMD4<Float>(cosine, 0, -sine, 0),
            SIMD4<Float>(0, 1, 0, 0),
            SIMD4<Float>(sine, 0, cosine, 0),
            SIMD4<Float>(0, 0, 0, 1)
        )
    }
}
