import Foundation

enum ScanState: String {
    case idle = "IDLE"
    case recording = "RECORDING"
    case finalizing = "FINALIZING"
    case readyToTransfer = "READY_TO_TRANSFER"
    case transferredVerified = "TRANSFERRED + VERIFIED"
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
}

struct SessionValidation: Codable {
    let valid: Bool
    let checkedFrames: Int
    let message: String
}

struct SessionMetadata: Codable {
    let schemaVersion: Int
    let sessionID: String
    let frameCount: Int
    let framesDirectory: String
    let finalizedAtUTC: String
    let validation: SessionValidation

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case sessionID = "session_id"
        case frameCount = "frame_count"
        case framesDirectory = "frames_directory"
        case finalizedAtUTC = "finalized_at_utc"
        case validation
    }
}

struct SessionFinalizationResult {
    let sessionID: String
    let frameCount: Int
    let directory: URL
}
