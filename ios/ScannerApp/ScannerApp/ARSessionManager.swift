import Foundation
import ARKit
import Combine
import UIKit

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
    @Published var durationText = "00:00"
    @Published var storageText = "0 B"
    @Published var serverURLText = TransferSettings.serverURL
    @Published var authTokenText = ""
    @Published var isTransferring = false
    @Published var uploadedFrameCount = 0
    @Published var processedFrameCount = 0
    @Published var uploadBacklog = 0
    @Published var liveConnectionStatus = "NOT CONFIGURED"

    private let captureService = FrameCaptureService()
    private lazy var transferService = SessionTransferService(captureService: captureService)
    private let recordingPolicy = RecordingPolicy()
    private let storagePolicy = StoragePolicy.default
    private let keyframeSelector = KeyframeSelector()
    private let captureStateLock = NSLock()
    private let captureQueue = DispatchQueue(label: "com.local.iphone3dscanner.capture")
    private var sessionID = UUID().uuidString.lowercased()
    private var recording = false
    private var nextFrameIndex = 0
    private var sessionBytes: Int64 = 0
    private var sessionStartedAt = Date()
    private var writePending = false
    private var liveUploadService: LiveUploadService?
    private var liveStatusTask: Task<Void, Never>?

    override init() {
        super.init()
        authTokenText = KeychainStore.load() ?? ""
        session.delegate = self
        let incompleteCount = captureService.incompleteSessionIDs().count
        if incompleteCount > 0 {
            captureStatus = "Incomplete session found; kept for manual cleanup"
        }
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
        captureStateLock.lock()
        let wasRecording = recording
        captureStateLock.unlock()
        if wasRecording {
            stopScan(message: "App paused; finalizing session...")
        }
        session.pause()
    }

    func deleteCompletedSession() {
        guard scanState == .completed else { return }
        let completedSessionID = sessionID
        do {
            try captureService.deleteSession(sessionID: completedSessionID)
            DispatchQueue.main.async {
                self.scanState = .idle
                self.capturedFrameCount = 0
                self.durationText = "00:00"
                self.storageText = "0 B"
                self.captureStatus = "Completed session deleted"
            }
        } catch {
            reportScanError("Delete failed: \(error.localizedDescription)")
        }
    }

    func saveServerURL() {
        TransferSettings.serverURL = serverURLText
        serverURLText = TransferSettings.serverURL
    }

    func testConnection() {
        saveServerURL()
        do { try KeychainStore.save(authTokenText) } catch {
            captureStatus = error.localizedDescription
            return
        }
        captureStatus = "Testing receiver..."
        Task {
            do {
                try await transferService.testConnection(serverURLString: TransferSettings.serverURL, authToken: authTokenText)
                await MainActor.run { self.captureStatus = "Receiver connection verified" }
            } catch {
                await MainActor.run { self.captureStatus = "Connection failed: \(error.localizedDescription)" }
            }
        }
    }

    func transferCompletedSession() {
        guard scanState == .completed, !isTransferring else { return }
        saveServerURL()
        do { try KeychainStore.save(authTokenText) } catch {
            captureStatus = error.localizedDescription
            return
        }
        guard !TransferSettings.serverURL.isEmpty else {
            captureStatus = "Enter the Windows receiver URL first"
            return
        }
        let currentSessionID = sessionID
        isTransferring = true
        captureStatus = "Transferring session; local copy is protected..."
        Task {
            do {
                let result = try await transferService.transfer(
                    sessionID: currentSessionID,
                    serverURLString: TransferSettings.serverURL,
                    authToken: authTokenText
                )
                await MainActor.run {
                    self.isTransferring = false
                    self.scanState = .idle
                    self.capturedFrameCount = 0
                    self.durationText = "00:00"
                    self.storageText = "0 B"
                    self.captureStatus = "Transfer verified; deleted \(result.fileCount) local files"
                }
            } catch {
                await MainActor.run {
                    self.isTransferring = false
                    self.captureStatus = "Transfer failed - LOCAL SESSION PRESERVED\n\(error.localizedDescription)"
                }
            }
        }
    }

    func startScan() {
        captureStateLock.lock()
        let alreadyRecording = recording
        captureStateLock.unlock()
        guard !alreadyRecording else { return }

        guard ARWorldTrackingConfiguration.isSupported,
              ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) else {
            reportScanError("Cannot start: LiDAR sceneDepth is unavailable")
            return
        }

        let documentsURL = FileManager.default.urls(
            for: .documentDirectory,
            in: .userDomainMask
        )[0]
        guard storagePolicy.canStart(at: documentsURL) else {
            reportScanError("Cannot start: less than 2 GB free storage")
            return
        }

        let newSessionID = UUID().uuidString.lowercased()
        let startDate = Date()
        do {
            _ = try captureService.initializeSession(
                sessionID: newSessionID,
                startedAt: startDate,
                policy: recordingPolicy
            )
        } catch {
            reportScanError("Cannot create session: \(error.localizedDescription)")
            return
        }

        captureStateLock.lock()
        guard !recording else {
            captureStateLock.unlock()
            return
        }
        sessionID = newSessionID
        nextFrameIndex = 0
        sessionBytes = 0
        sessionStartedAt = startDate
        writePending = false
        keyframeSelector.reset()
        recording = true
        captureStateLock.unlock()

        DispatchQueue.main.async {
            UIApplication.shared.isIdleTimerDisabled = true
            self.scanState = .recording
            self.capturedFrameCount = 0
            self.durationText = "00:00"
            self.storageText = "0 B"
            self.captureStatus = "Recording synchronized RGB-D frames..."
        }

        if let live = try? LiveUploadService(serverURLString: TransferSettings.serverURL, authToken: authTokenText) {
            liveUploadService = live
            Task {
                do {
                    try await live.start(sessionID: newSessionID)
                    await MainActor.run { self.liveConnectionStatus = "CONNECTED" }
                    self.liveStatusTask = Task { [weak self] in
                        while !Task.isCancelled {
                            try? await Task.sleep(for: .seconds(1))
                            guard let self else { return }
                            do {
                                let status = try await live.status(sessionID: newSessionID)
                                await MainActor.run {
                                    self.uploadedFrameCount = status.uploadedFiles / 4
                                    self.processedFrameCount = status.processedFrames
                                    self.uploadBacklog = status.uploadBacklogFiles
                                    self.liveConnectionStatus = "CONNECTED"
                                }
                            } catch {
                                await MainActor.run { self.liveConnectionStatus = "RECONNECTING" }
                            }
                        }
                    }
                } catch {
                    await MainActor.run { self.liveConnectionStatus = "OFFLINE; LOCAL CAPTURE CONTINUES" }
                }
            }
        } else {
            liveConnectionStatus = "NOT CONFIGURED"
        }
    }

    func stopScan() {
        stopScan(message: "Finalizing session...")
    }

    private func stopScan(message: String) {
        captureStateLock.lock()
        guard recording else {
            captureStateLock.unlock()
            return
        }
        recording = false
        let finishingSessionID = sessionID
        let finishingStartedAt = sessionStartedAt
        captureStateLock.unlock()

        DispatchQueue.main.async {
            UIApplication.shared.isIdleTimerDisabled = false
            self.scanState = .finalizing
            self.captureStatus = message
        }

        captureQueue.async {
            Task {
                await self.liveUploadService?.finishAndWait()
                do {
                    let result = try self.captureService.finalizeSession(
                        sessionID: finishingSessionID,
                        frameCount: self.successfulFrameCount,
                        totalBytes: self.sessionBytes,
                        startedAt: finishingStartedAt
                    )
                    DispatchQueue.main.async {
                        self.liveStatusTask?.cancel()
                        self.scanState = .completed
                        self.capturedFrameCount = result.frameCount
                        self.captureStatus = "Session complete: \(result.frameCount) frames validated"
                    }
                } catch {
                    DispatchQueue.main.async {
                        self.liveStatusTask?.cancel()
                        self.scanState = .error
                        self.captureStatus = "Finalization failed: \(error.localizedDescription)"
                    }
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

        guard !writePending else {
            captureStateLock.unlock()
            return
        }

        let hasDepth = frame.sceneDepth != nil
        let hasConfidence = frame.sceneDepth?.confidenceMap != nil
        let trackingIsNormal: Bool
        if case .normal = frame.camera.trackingState {
            trackingIsNormal = true
        } else {
            trackingIsNormal = false
        }

        guard keyframeSelector.shouldCapture(
            timestamp: frame.timestamp,
            transform: frame.camera.transform,
            trackingIsNormal: trackingIsNormal,
            hasDepth: hasDepth,
            hasConfidence: hasConfidence
        ) else {
            captureStateLock.unlock()
            return
        }

        let documentsURL = FileManager.default.urls(
            for: .documentDirectory,
            in: .userDomainMask
        )[0]
        guard storagePolicy.canContinue(at: documentsURL, sessionBytes: sessionBytes) else {
            captureStateLock.unlock()
            stopScan(message: "Storage limit reached; finalizing session...")
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
        let currentSessionID = sessionID
        writePending = true
        captureStateLock.unlock()

        captureQueue.async {
            do {
                let result = try self.captureService.persist(
                    capturedFrame,
                    sessionID: currentSessionID,
                    frameIndex: frameIndex
                )
                self.captureStateLock.lock()
                self.nextFrameIndex += 1
                self.sessionBytes += result.totalBytes
                self.writePending = false
                let frameCount = self.nextFrameIndex
                let totalBytes = self.sessionBytes
                let startedAt = self.sessionStartedAt
                self.captureStateLock.unlock()
                if let live = self.liveUploadService {
                    let enqueueComplete = DispatchSemaphore(value: 0)
                    Task {
                        await live.enqueueFrame(
                            sessionID: currentSessionID,
                            frameDirectory: result.directory,
                            frameIndex: frameIndex
                        )
                        enqueueComplete.signal()
                    }
                    enqueueComplete.wait()
                }
                DispatchQueue.main.async {
                    self.capturedFrameCount = frameCount
                    self.durationText = self.formatDuration(Date().timeIntervalSince(startedAt))
                    self.storageText = self.formatBytes(totalBytes)
                    self.captureStatus = "Recording frame \(frameIndex)"
                }
            } catch {
                self.captureStateLock.lock()
                self.writePending = false
                self.captureStateLock.unlock()
                self.reportCaptureFailure("Capture failed: \(error.localizedDescription)")
                self.stopScan(message: "Capture failed; finalizing session...")
            }
        }
    }

    private var successfulFrameCount: Int {
        captureStateLock.lock()
        defer { captureStateLock.unlock() }
        return nextFrameIndex
    }

    private func reportScanError(_ message: String) {
        DispatchQueue.main.async {
            self.scanState = .error
            self.captureStatus = message
        }
    }

    private func formatDuration(_ seconds: TimeInterval) -> String {
        String(format: "%02d:%02d", Int(seconds) / 60, Int(seconds) % 60)
    }

    private func formatBytes(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
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
