import Foundation

enum ScanState: String {
    case idle = "IDLE"
    case recording = "RECORDING"
    case finalizing = "FINALIZING"
    case completed = "COMPLETED"
    case error = "ERROR"
    case betweenPasses = "BETWEEN PASSES"
}

enum ObjectScanState: Equatable {
    case idle
    case recordingPass(id: Int)
    case betweenPasses(completedPasses: Int)
}

enum ScanMode: String, Codable, CaseIterable, Identifiable {
    case scene
    case object

    var id: String { rawValue }
    var displayName: String { rawValue.capitalized }
}

struct ScanPassMetadata: Codable, Equatable {
    let id: Int
    let startFrame: Int
    let endFrame: Int

    enum CodingKeys: String, CodingKey {
        case id
        case startFrame = "start_frame"
        case endFrame = "end_frame"
    }
}

struct FrameMetadata: Codable {
    let schemaVersion: Int
    let frameIndex: Int
    let timestampSeconds: Double
    let timestampOrigin: String
    let rgb: RGBMetadata
    let depth: DepthMetadata
    let confidence: ConfidenceMetadata
    let camera: CameraMetadata

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case frameIndex = "frame_index"
        case timestampSeconds = "timestamp_seconds"
        case timestampOrigin = "timestamp_origin"
        case rgb
        case depth
        case confidence
        case camera
    }
}

struct RGBMetadata: Codable {
    let file: String
    let width: Int
    let height: Int
}

struct DepthMetadata: Codable {
    let file: String
    let width: Int
    let height: Int
    let dtype: String
    let endianness: String
    let unit: String
}

struct ConfidenceMetadata: Codable {
    let file: String
    let width: Int
    let height: Int
    let dtype: String
    let encoding: [String: String]
}

struct CameraMetadata: Codable {
    let imageWidth: Int
    let imageHeight: Int
    let intrinsicsRows: [[Float]]
    let transformSemantics: String
    let transformRows: [[Float]]
    let coordinateSystem: String
    let units: String
    let forwardAxis: String

    enum CodingKeys: String, CodingKey {
        case imageWidth = "image_width"
        case imageHeight = "image_height"
        case intrinsicsRows = "intrinsics_rows"
        case transformSemantics = "transform_semantics"
        case transformRows = "transform_rows"
        case coordinateSystem = "coordinate_system"
        case units
        case forwardAxis = "forward_axis"
    }
}

struct CapturedFrame {
    let rgbJPEG: Data
    let rgbWidth: Int
    let rgbHeight: Int
    let depthData: Data
    let depthWidth: Int
    let depthHeight: Int
    let confidenceData: Data
    let timestamp: Double
    let imageWidth: Int
    let imageHeight: Int
    let intrinsicsRows: [[Float]]
    let transformRows: [[Float]]

    func metadata(frameIndex: Int) -> FrameMetadata {
        FrameMetadata(
            schemaVersion: 1,
            frameIndex: frameIndex,
            timestampSeconds: timestamp,
            timestampOrigin: "ARSession",
            rgb: RGBMetadata(
                file: "rgb.jpg",
                width: rgbWidth,
                height: rgbHeight
            ),
            depth: DepthMetadata(
                file: "depth.f32",
                width: depthWidth,
                height: depthHeight,
                dtype: "float32",
                endianness: "little",
                unit: "meters"
            ),
            confidence: ConfidenceMetadata(
                file: "confidence.u8",
                width: depthWidth,
                height: depthHeight,
                dtype: "uint8",
                encoding: [
                    "0": "low",
                    "1": "medium",
                    "2": "high"
                ]
            ),
            camera: CameraMetadata(
                imageWidth: imageWidth,
                imageHeight: imageHeight,
                intrinsicsRows: intrinsicsRows,
                transformSemantics: "world_from_camera",
                transformRows: transformRows,
                coordinateSystem: "ARKit",
                units: "meters",
                forwardAxis: "-Z"
            )
        )
    }
}

struct CaptureResult {
    let frameIndex: Int
    let directory: URL
    let rgbBytes: Int
    let rgbWidth: Int
    let rgbHeight: Int
    let depthBytes: Int
    let depthWidth: Int
    let depthHeight: Int
    let confidenceBytes: Int
    let totalBytes: Int64
}

struct SessionValidation: Codable {
    let valid: Bool
    let checkedFrames: Int
    let message: String
}

struct RecordingPolicyMetadata: Codable {
    let minimumFrameIntervalSeconds: Double
    let translationThresholdMeters: Double
    let rotationThresholdDegrees: Double

    enum CodingKeys: String, CodingKey {
        case minimumFrameIntervalSeconds = "minimum_frame_interval_seconds"
        case translationThresholdMeters = "translation_threshold_meters"
        case rotationThresholdDegrees = "rotation_threshold_degrees"
    }
}

struct SensorMetadata: Codable {
    let depthUnit: String
    let coordinateSystem: String

    enum CodingKeys: String, CodingKey {
        case depthUnit = "depth_unit"
        case coordinateSystem = "coordinate_system"
    }
}

struct SessionMetadata: Codable {
    let schemaVersion: Int
    let sessionID: String
    let status: String
    let startedAtUTC: String
    let endedAtUTC: String?
    let durationSeconds: Double
    let frameCount: Int
    let totalBytes: Int64
    let captureMode: String
    let recordingPolicy: RecordingPolicyMetadata
    let sensor: SensorMetadata
    let validation: SessionValidation?
    let scanMode: ScanMode?
    let passes: [ScanPassMetadata]?

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case sessionID = "session_id"
        case status
        case startedAtUTC = "started_at_utc"
        case endedAtUTC = "ended_at_utc"
        case durationSeconds = "duration_seconds"
        case frameCount = "frame_count"
        case totalBytes = "total_bytes"
        case captureMode = "capture_mode"
        case recordingPolicy = "recording_policy"
        case sensor
        case validation
        case scanMode = "scan_mode"
        case passes
    }
}

struct SessionFinalizationResult {
    let sessionID: String
    let frameCount: Int
    let directory: URL
}
