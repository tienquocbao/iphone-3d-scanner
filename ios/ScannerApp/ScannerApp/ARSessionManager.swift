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
    @Published var captureStatus = "Ready"

    private let captureService = FrameCaptureService()
    private let captureStateLock = NSLock()
    private let sessionID = UUID().uuidString.lowercased()
    private var captureRequested = false
    private var nextFrameIndex = 0

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

    func requestCapture() {
        captureStateLock.lock()
        captureRequested = true
        captureStateLock.unlock()

        DispatchQueue.main.async {
            self.captureStatus = "Waiting for valid RGB-D frame..."
        }
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

        if captureIsRequested {
            captureFrame(frame, sceneDepthAvailable: true)
        }

        DispatchQueue.main.async {
            self.trackingStatus = tracking
            self.depthResolution = "\(width) x \(height)"
            self.centerDepth = depth
            self.confidence = confidenceText
        }
    }

    private var captureIsRequested: Bool {
        captureStateLock.lock()
        defer { captureStateLock.unlock() }
        return captureRequested
    }

    private func captureFrame(
        _ frame: ARFrame,
        sceneDepthAvailable: Bool
    ) {
        guard sceneDepthAvailable else { return }

        guard frame.sceneDepth?.confidenceMap != nil else {
            _ = consumeCaptureRequest()
            reportCaptureFailure("Capture failed: confidence map unavailable")
            return
        }

        let capturedFrame: CapturedFrame
        do {
            capturedFrame = try captureService.extract(frame: frame)
        } catch {
            _ = consumeCaptureRequest()
            reportCaptureFailure("Capture failed: \(error.localizedDescription)")
            return
        }

        guard let frameIndex = consumeCaptureRequest() else { return }
        let currentSessionID = sessionID

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let result = try self.captureService.persist(
                    capturedFrame,
                    sessionID: currentSessionID,
                    frameIndex: frameIndex
                )
                DispatchQueue.main.async {
                    self.captureStatus = "Captured frame \(result.frameIndex) | RGB \(result.rgbWidth) x \(result.rgbHeight) | depth \(result.depthWidth) x \(result.depthHeight) | \(result.depthBytes) B depth, \(result.confidenceBytes) B confidence"
                }
            } catch {
                self.reportCaptureFailure("Capture failed: \(error.localizedDescription)")
            }
        }
    }

    private func consumeCaptureRequest() -> Int? {
        captureStateLock.lock()
        defer { captureStateLock.unlock() }

        guard captureRequested else { return nil }
        captureRequested = false
        let frameIndex = nextFrameIndex
        nextFrameIndex += 1
        return frameIndex
    }

    private func reportCaptureFailure(_ message: String) {
        DispatchQueue.main.async {
            self.captureStatus = message
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
                return "LIMITED - initializing"

            case .excessiveMotion:
                return "LIMITED - excessive motion"

            case .insufficientFeatures:
                return "LIMITED - insufficient features"

            case .relocalizing:
                return "LIMITED - relocalizing"

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
