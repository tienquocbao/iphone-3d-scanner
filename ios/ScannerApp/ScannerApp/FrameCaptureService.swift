import ARKit
import CoreImage
import Foundation
import ImageIO

enum FrameCaptureError: LocalizedError {
    case sceneDepthUnavailable
    case confidenceUnavailable
    case invalidDimensions
    case invalidRGBData
    case frameAlreadyExists
    case invalidDepthDataSize
    case invalidConfidenceDataSize
    case nonFiniteCameraMatrix
    case jpegConversionFailed

    var errorDescription: String? {
        switch self {
        case .sceneDepthUnavailable:
            return "sceneDepth is unavailable"
        case .confidenceUnavailable:
            return "confidence map is unavailable"
        case .invalidDimensions:
            return "depth and confidence dimensions do not match"
        case .invalidRGBData:
            return "RGB JPEG data is missing or empty"
        case .frameAlreadyExists:
            return "Frame directory already exists"
        case .invalidDepthDataSize:
            return "depth data size is invalid"
        case .invalidConfidenceDataSize:
            return "confidence data size is invalid"
        case .nonFiniteCameraMatrix:
            return "camera matrix contains a non-finite value"
        case .jpegConversionFailed:
            return "JPEG conversion failed"
        }
    }
}

final class FrameCaptureService {
    private let ciContext = CIContext()
    private let fileManager = FileManager.default

    func extract(frame: ARFrame) throws -> CapturedFrame {
        guard let sceneDepth = frame.sceneDepth else {
            throw FrameCaptureError.sceneDepthUnavailable
        }

        guard let confidenceMap = sceneDepth.confidenceMap else {
            throw FrameCaptureError.confidenceUnavailable
        }

        let depthMap = sceneDepth.depthMap
        let depthWidth = CVPixelBufferGetWidth(depthMap)
        let depthHeight = CVPixelBufferGetHeight(depthMap)
        let confidenceWidth = CVPixelBufferGetWidth(confidenceMap)
        let confidenceHeight = CVPixelBufferGetHeight(confidenceMap)

        guard depthWidth == confidenceWidth, depthHeight == confidenceHeight else {
            throw FrameCaptureError.invalidDimensions
        }

        let rgbImage = CIImage(cvPixelBuffer: frame.capturedImage)
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)
        let jpegOptions: [CIImageRepresentationOption: Any] = [
            CIImageRepresentationOption(
                rawValue: kCGImageDestinationLossyCompressionQuality as String
            ): 0.92
        ]

        guard let colorSpace,
              let rgbJPEG = ciContext.jpegRepresentation(
                  of: rgbImage,
                  colorSpace: colorSpace,
                  options: jpegOptions
              )
        else {
            throw FrameCaptureError.jpegConversionFailed
        }

        let depthData = try copyDepthData(
            depthMap,
            width: depthWidth,
            height: depthHeight
        )
        let confidenceData = try copyConfidenceData(
            confidenceMap,
            width: confidenceWidth,
            height: confidenceHeight
        )

        let intrinsics = frame.camera.intrinsics
        let transform = frame.camera.transform
        let intrinsicsRows: [[Float]] = [
            [intrinsics.columns.0.x, intrinsics.columns.1.x, intrinsics.columns.2.x],
            [intrinsics.columns.0.y, intrinsics.columns.1.y, intrinsics.columns.2.y],
            [intrinsics.columns.0.z, intrinsics.columns.1.z, intrinsics.columns.2.z]
        ]
        let transformRows: [[Float]] = [
            [transform.columns.0.x, transform.columns.1.x, transform.columns.2.x, transform.columns.3.x],
            [transform.columns.0.y, transform.columns.1.y, transform.columns.2.y, transform.columns.3.y],
            [transform.columns.0.z, transform.columns.1.z, transform.columns.2.z, transform.columns.3.z],
            [transform.columns.0.w, transform.columns.1.w, transform.columns.2.w, transform.columns.3.w]
        ]

        let matrixValues = intrinsicsRows.flatMap { $0 } + transformRows.flatMap { $0 }
        guard matrixValues.allSatisfy(\.isFinite) else {
            throw FrameCaptureError.nonFiniteCameraMatrix
        }

        return CapturedFrame(
            rgbJPEG: rgbJPEG,
            rgbWidth: CVPixelBufferGetWidth(frame.capturedImage),
            rgbHeight: CVPixelBufferGetHeight(frame.capturedImage),
            depthData: depthData,
            depthWidth: depthWidth,
            depthHeight: depthHeight,
            confidenceData: confidenceData,
            timestamp: frame.timestamp,
            imageWidth: Int(frame.camera.imageResolution.width.rounded()),
            imageHeight: Int(frame.camera.imageResolution.height.rounded()),
            intrinsicsRows: intrinsicsRows,
            transformRows: transformRows
        )
    }

    func persist(
        _ frame: CapturedFrame,
        sessionID: String,
        frameIndex: Int
    ) throws -> CaptureResult {
        let documentsURL = fileManager.urls(
            for: .documentDirectory,
            in: .userDomainMask
        )[0]
        let frameDirectory = documentsURL
            .appendingPathComponent("Scans", isDirectory: true)
            .appendingPathComponent("session_\(sessionID)", isDirectory: true)
            .appendingPathComponent("frames", isDirectory: true)
            .appendingPathComponent(String(format: "%06d", frameIndex), isDirectory: true)
        let temporaryDirectory = frameDirectory.deletingLastPathComponent()
            .appendingPathComponent(".tmp_\(String(format: "%06d", frameIndex))", isDirectory: true)

        try fileManager.createDirectory(
            at: temporaryDirectory,
            withIntermediateDirectories: true
        )
        var committed = false
        defer {
            if !committed {
                try? fileManager.removeItem(at: temporaryDirectory)
            }
        }

        let metadata = try JSONEncoder().encodedMetadata(
            frame.metadata(frameIndex: frameIndex)
        )

        try frame.rgbJPEG.write(
            to: temporaryDirectory.appendingPathComponent("rgb.jpg"),
            options: .atomic
        )
        try frame.depthData.write(
            to: temporaryDirectory.appendingPathComponent("depth.f32"),
            options: .atomic
        )
        try frame.confidenceData.write(
            to: temporaryDirectory.appendingPathComponent("confidence.u8"),
            options: .atomic
        )
        try metadata.write(
            to: temporaryDirectory.appendingPathComponent("frame.json"),
            options: .atomic
        )
        guard !fileManager.fileExists(atPath: frameDirectory.path) else {
            throw FrameCaptureError.frameAlreadyExists
        }
        try fileManager.moveItem(at: temporaryDirectory, to: frameDirectory)
        committed = true

        return CaptureResult(
            frameIndex: frameIndex,
            directory: frameDirectory,
            rgbBytes: frame.rgbJPEG.count,
            rgbWidth: frame.rgbWidth,
            rgbHeight: frame.rgbHeight,
            depthBytes: frame.depthData.count,
            depthWidth: frame.depthWidth,
            depthHeight: frame.depthHeight,
            confidenceBytes: frame.confidenceData.count,
            totalBytes: Int64(frame.rgbJPEG.count + frame.depthData.count + frame.confidenceData.count + metadata.count)
        )
    }

    func initializeSession(
        sessionID: String,
        startedAt: Date,
        policy: RecordingPolicy
    ) throws -> URL {
        let sessionDirectory = try sessionDirectory(for: sessionID)
        let metadata = SessionMetadata(
            schemaVersion: 1,
            sessionID: sessionID,
            status: "recording",
            startedAtUTC: ISO8601DateFormatter().string(from: startedAt),
            endedAtUTC: nil,
            durationSeconds: 0,
            frameCount: 0,
            totalBytes: 0,
            captureMode: "keyframe",
            recordingPolicy: policy.metadata,
            sensor: SensorMetadata(depthUnit: "meters", coordinateSystem: "ARKit"),
            validation: nil
        )
        try JSONEncoder().encodedSessionMetadata(metadata).write(
            to: sessionDirectory.appendingPathComponent("session.json"),
            options: .atomic
        )
        return sessionDirectory
    }

    func finalizeSession(
        sessionID: String,
        frameCount: Int,
        totalBytes: Int64,
        startedAt: Date
    ) throws -> SessionFinalizationResult {
        let sessionDirectory = try sessionDirectory(for: sessionID)
        let framesDirectory = sessionDirectory.appendingPathComponent(
            "frames",
            isDirectory: true
        )
        try fileManager.createDirectory(
            at: framesDirectory,
            withIntermediateDirectories: true
        )

        var checkedFrames = 0
        for frameIndex in 0..<frameCount {
            let frameDirectory = framesDirectory.appendingPathComponent(
                String(format: "%06d", frameIndex),
                isDirectory: true
            )
            try validateFrameDirectory(frameDirectory)
            checkedFrames += 1
        }

        let metadata = SessionMetadata(
            schemaVersion: 1,
            sessionID: sessionID,
            status: "completed",
            startedAtUTC: ISO8601DateFormatter().string(from: startedAt),
            endedAtUTC: ISO8601DateFormatter().string(from: Date()),
            durationSeconds: max(0, Date().timeIntervalSince(startedAt)),
            frameCount: frameCount,
            totalBytes: totalBytes,
            captureMode: "keyframe",
            recordingPolicy: RecordingPolicy().metadata,
            sensor: SensorMetadata(depthUnit: "meters", coordinateSystem: "ARKit"),
            validation: SessionValidation(
                valid: true,
                checkedFrames: checkedFrames,
                message: "All frame files and metadata validated"
            )
        )
        let sessionData = try JSONEncoder().encodedSessionMetadata(metadata)
        try sessionData.write(
            to: sessionDirectory.appendingPathComponent("session.json"),
            options: .atomic
        )

        return SessionFinalizationResult(
            sessionID: sessionID,
            frameCount: frameCount,
            directory: sessionDirectory
        )
    }

    func deleteSession(sessionID: String) throws {
        let directory = try sessionDirectory(for: sessionID)
        guard fileManager.fileExists(atPath: directory.path) else { return }
        try fileManager.removeItem(at: directory)
    }

    func incompleteSessionIDs() -> [String] {
        let documentsURL = fileManager.urls(
            for: .documentDirectory,
            in: .userDomainMask
        )[0]
        let scansDirectory = documentsURL.appendingPathComponent("Scans", isDirectory: true)
        guard let entries = try? fileManager.contentsOfDirectory(
            at: scansDirectory,
            includingPropertiesForKeys: [.isDirectoryKey]
        ) else { return [] }

        return entries.compactMap { directory in
            let metadataURL = directory.appendingPathComponent("session.json")
            guard let data = try? Data(contentsOf: metadataURL),
                  let metadata = try? JSONDecoder().decode(SessionMetadata.self, from: data),
                  metadata.status == "recording" else { return nil }
            return metadata.sessionID
        }
    }

    func sessionDirectory(for sessionID: String) throws -> URL {
        let documentsURL = fileManager.urls(
            for: .documentDirectory,
            in: .userDomainMask
        )[0]
        let scansDirectory = documentsURL.appendingPathComponent("Scans", isDirectory: true)
        let sessionDirectory = scansDirectory.appendingPathComponent(
            "session_\(sessionID)",
            isDirectory: true
        )
        try fileManager.createDirectory(
            at: sessionDirectory.appendingPathComponent("frames", isDirectory: true),
            withIntermediateDirectories: true
        )
        return sessionDirectory
    }

    private func validateFrameDirectory(_ directory: URL) throws {
        let metadataData = try Data(contentsOf: directory.appendingPathComponent("frame.json"))
        let metadata = try JSONDecoder().decode(FrameMetadata.self, from: metadataData)
        let depthData = try Data(contentsOf: directory.appendingPathComponent(metadata.depth.file))
        let confidenceData = try Data(contentsOf: directory.appendingPathComponent(metadata.confidence.file))
        let rgbData = try Data(contentsOf: directory.appendingPathComponent(metadata.rgb.file))

        guard !rgbData.isEmpty else { throw FrameCaptureError.invalidRGBData }
        guard depthData.count == metadata.depth.width * metadata.depth.height * 4 else {
            throw FrameCaptureError.invalidDepthDataSize
        }
        guard confidenceData.count == metadata.confidence.width * metadata.confidence.height,
              metadata.confidence.width == metadata.depth.width,
              metadata.confidence.height == metadata.depth.height else {
            throw FrameCaptureError.invalidConfidenceDataSize
        }
    }

    private func copyDepthData(
        _ pixelBuffer: CVPixelBuffer,
        width: Int,
        height: Int
    ) throws -> Data {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw FrameCaptureError.invalidDepthDataSize
        }

        let rowStride = CVPixelBufferGetBytesPerRow(pixelBuffer) / MemoryLayout<Float32>.size
        let values = baseAddress.assumingMemoryBound(to: Float32.self)
        var data = Data(capacity: width * height * MemoryLayout<Float32>.size)

        for y in 0..<height {
            for x in 0..<width {
                var bits = values[y * rowStride + x].bitPattern.littleEndian
                withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
            }
        }

        guard data.count == width * height * 4 else {
            throw FrameCaptureError.invalidDepthDataSize
        }
        return data
    }

    private func copyConfidenceData(
        _ pixelBuffer: CVPixelBuffer,
        width: Int,
        height: Int
    ) throws -> Data {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw FrameCaptureError.invalidConfidenceDataSize
        }

        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let bytes = baseAddress.assumingMemoryBound(to: UInt8.self)
        var data = Data(capacity: width * height)

        for y in 0..<height {
            data.append(bytes.advanced(by: y * bytesPerRow), count: width)
        }

        guard data.count == width * height else {
            throw FrameCaptureError.invalidConfidenceDataSize
        }
        return data
    }
}

private extension JSONEncoder {
    func encodedMetadata(_ metadata: FrameMetadata) throws -> Data {
        outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encode(metadata)
    }

    func encodedSessionMetadata(_ metadata: SessionMetadata) throws -> Data {
        outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encode(metadata)
    }
}
