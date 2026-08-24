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
    @Published var scanState: ScanState = .idle
    @Published var capturedFrameCount = 0

    private let captureService = FrameCaptureService()
    private let captureStateLock = NSLock()
    private let captureQueue = DispatchQueue(label: "com.local.iphone3dscanner.capture")
    private var sessionID = UUID().uuidString.lowercased()
    private var recording = false
    private var lastCaptureTimestamp: TimeInterval?
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

    func startScan() {
        captureStateLock.lock()
        guard !recording else {
            captureStateLock.unlock()
            return
        }
        sessionID = UUID().uuidString.lowercased()
        nextFrameIndex = 0
        lastCaptureTimestamp = nil
        recording = true
        captureStateLock.unlock()

        DispatchQueue.main.async {
            self.scanState = .recording
            self.capturedFrameCount = 0
            self.captureStatus = "Recording synchronized RGB-D frames..."
        }
    }

    func stopScan() {
        captureStateLock.lock()
        guard recording else {
            captureStateLock.unlock()
            return
        }
        recording = false
        let finishingSessionID = sessionID
        let finishingFrameCount = nextFrameIndex
        captureStateLock.unlock()

        DispatchQueue.main.async {
            self.scanState = .finalizing
            self.captureStatus = "Finalizing session..."
        }

        captureQueue.async {
            do {
                let result = try self.captureService.finalizeSession(
                    sessionID: finishingSessionID,
                    frameCount: finishingFrameCount
                )
                DispatchQueue.main.async {
                    self.scanState = .readyToTransfer
                    self.captureStatus = "Session ready: \(result.frameCount) frames validated"
                }
            } catch {
                DispatchQueue.main.async {
                    self.scanState = .idle
                    self.captureStatus = "Finalization failed: \(error.localizedDescription)"
                }
            }
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

        captureFrameIfRecording(frame)

        DispatchQueue.main.async {
            self.trackingStatus = tracking
            self.depthResolution = "\(width) x \(height)"
            self.centerDepth = depth
            self.confidence = confidenceText
        }
    }

    private func captureFrameIfRecording(_ frame: ARFrame) {
        captureStateLock.lock()
        guard recording else {
            captureStateLock.unlock()
            return
        }

        if let lastCaptureTimestamp,
           frame.timestamp - lastCaptureTimestamp < 0.2 {
            captureStateLock.unlock()
            return
        }

        guard frame.sceneDepth?.confidenceMap != nil else {
            captureStateLock.unlock()
            return
        }

        let capturedFrame: CapturedFrame
        do {
            capturedFrame = try captureService.extract(frame: frame)
        } catch {
            captureStateLock.unlock()
            return
        }

        let frameIndex = nextFrameIndex
        nextFrameIndex += 1
        lastCaptureTimestamp = frame.timestamp
        let currentSessionID = sessionID
        captureStateLock.unlock()

        captureQueue.async {
            do {
                _ = try self.captureService.persist(
                    capturedFrame,
                    sessionID: currentSessionID,
                    frameIndex: frameIndex
                )
                DispatchQueue.main.async {
                    self.capturedFrameCount = frameIndex + 1
                    self.captureStatus = "Recording frame \(frameIndex)"
                }
            } catch {
                self.reportCaptureFailure("Capture failed: \(error.localizedDescription)")
            }
        }
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
