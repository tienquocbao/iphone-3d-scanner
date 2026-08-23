import Foundation
import ARKit
import Combine

final class ARSessionManager: NSObject, ObservableObject, ARSessionDelegate {

    let session = ARSession()

    @Published var lidarSupported = false
    @Published var trackingStatus = "Starting..."
    @Published var depthResolution = "-"
    @Published var centerDepth: Float?
    @Published var confidence = "-"

    override init() {
        super.init()
        session.delegate = self
    }

    func start() {
        guard ARWorldTrackingConfiguration.isSupported else {
            DispatchQueue.main.async {
                self.trackingStatus = "World tracking unsupported"
            }
            return
        }

        let configuration = ARWorldTrackingConfiguration()

        let supportsDepth =
            ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)

        DispatchQueue.main.async {
            self.lidarSupported = supportsDepth
        }

        guard supportsDepth else {
            DispatchQueue.main.async {
                self.trackingStatus = "sceneDepth unsupported"
            }

            session.run(
                configuration,
                options: [.resetTracking, .removeExistingAnchors]
            )
            return
        }

        configuration.frameSemantics.insert(.sceneDepth)
        configuration.worldAlignment = .gravity

        session.run(
            configuration,
            options: [.resetTracking, .removeExistingAnchors]
        )
    }

    func pause() {
        session.pause()
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let tracking = trackingDescription(frame.camera.trackingState)

        guard let sceneDepth = frame.sceneDepth else {
            DispatchQueue.main.async {
                self.trackingStatus = tracking
                self.depthResolution = "waiting..."
            }
            return
        }

        let depthMap = sceneDepth.depthMap

        let width = CVPixelBufferGetWidth(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)

        let depth = readCenterDepth(depthMap)

        var confidenceText = "-"

        if let confidenceMap = sceneDepth.confidenceMap {
            confidenceText = readCenterConfidence(confidenceMap)
        }

        DispatchQueue.main.async {
            self.trackingStatus = tracking
            self.depthResolution = "\(width) × \(height)"
            self.centerDepth = depth
            self.confidence = confidenceText
        }
    }

    private func trackingDescription(
        _ state: ARCamera.TrackingState
    ) -> String {

        switch state {
        case .normal:
            return "NORMAL"

        case .notAvailable:
            return "NOT AVAILABLE"

        case .limited(let reason):
            switch reason {
            case .initializing:
                return "LIMITED — initializing"

            case .excessiveMotion:
                return "LIMITED — excessive motion"

            case .insufficientFeatures:
                return "LIMITED — insufficient features"

            case .relocalizing:
                return "LIMITED — relocalizing"

            @unknown default:
                return "LIMITED"
            }
        }
    }

    private func readCenterDepth(
        _ pixelBuffer: CVPixelBuffer
    ) -> Float? {

        CVPixelBufferLockBaseAddress(
            pixelBuffer,
            .readOnly
        )

        defer {
            CVPixelBufferUnlockBaseAddress(
                pixelBuffer,
                .readOnly
            )
        }

        guard let baseAddress =
                CVPixelBufferGetBaseAddress(pixelBuffer)
        else {
            return nil
        }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        let bytesPerRow =
            CVPixelBufferGetBytesPerRow(pixelBuffer)

        let rowStride =
            bytesPerRow / MemoryLayout<Float32>.size

        let values =
            baseAddress.assumingMemoryBound(to: Float32.self)

        let x = width / 2
        let y = height / 2

        let value = values[y * rowStride + x]

        guard value.isFinite, value > 0 else {
            return nil
        }

        return value
    }

    private func readCenterConfidence(
        _ pixelBuffer: CVPixelBuffer
    ) -> String {

        CVPixelBufferLockBaseAddress(
            pixelBuffer,
            .readOnly
        )

        defer {
            CVPixelBufferUnlockBaseAddress(
                pixelBuffer,
                .readOnly
            )
        }

        guard let baseAddress =
                CVPixelBufferGetBaseAddress(pixelBuffer)
        else {
            return "-"
        }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        let bytesPerRow =
            CVPixelBufferGetBytesPerRow(pixelBuffer)

        let values =
            baseAddress.assumingMemoryBound(to: UInt8.self)

        let x = width / 2
        let y = height / 2

        let rawValue =
            values[y * bytesPerRow + x]

        switch rawValue {
        case 0:
            return "LOW"
        case 1:
            return "MEDIUM"
        case 2:
            return "HIGH"
        default:
            return "UNKNOWN (\(rawValue))"
        }
    }
}
