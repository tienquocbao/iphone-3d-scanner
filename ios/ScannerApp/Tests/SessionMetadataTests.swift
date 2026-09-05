import Foundation
import XCTest

@testable import ScannerApp

final class SessionMetadataTests: XCTestCase {
    func testObjectModeAndPassMetadataRoundTrip() throws {
        let metadata = SessionMetadata(
            schemaVersion: 1,
            sessionID: "object-session",
            status: "completed",
            startedAtUTC: "2026-09-05T00:00:00Z",
            endedAtUTC: "2026-09-05T00:00:01Z",
            durationSeconds: 1,
            frameCount: 2,
            totalBytes: 42,
            captureMode: "keyframe",
            recordingPolicy: RecordingPolicy().metadata,
            sensor: SensorMetadata(depthUnit: "meters", coordinateSystem: "ARKit"),
            validation: nil,
            scanMode: .object,
            passes: [ScanPassMetadata(id: 0, startFrame: 0, endFrame: 1)]
        )
        let decoded = try JSONDecoder().decode(SessionMetadata.self, from: JSONEncoder().encode(metadata))
        XCTAssertEqual(decoded.scanMode, .object)
        XCTAssertEqual(decoded.passes, [ScanPassMetadata(id: 0, startFrame: 0, endFrame: 1)])
    }

    func testLegacySessionWithoutScanModeLoads() throws {
        let data = Data("{\"schema_version\":1,\"session_id\":\"legacy\",\"status\":\"completed\",\"started_at_utc\":\"2026-09-05T00:00:00Z\",\"duration_seconds\":0,\"frame_count\":0,\"total_bytes\":0,\"capture_mode\":\"keyframe\",\"recording_policy\":{\"minimum_frame_interval_seconds\":0.2,\"translation_threshold_meters\":0.02,\"rotation_threshold_degrees\":3},\"sensor\":{\"depth_unit\":\"meters\",\"coordinate_system\":\"ARKit\"}}".utf8)
        let decoded = try JSONDecoder().decode(SessionMetadata.self, from: data)
        XCTAssertNil(decoded.scanMode)
        XCTAssertNil(decoded.passes)
    }
}
