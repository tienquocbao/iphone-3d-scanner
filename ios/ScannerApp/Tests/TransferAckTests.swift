import XCTest

final class TransferAckTests: XCTestCase {
    func testCanonicalManifestHashMatchesSharedFixture() {
        let manifest = TransferManifest(
            protocolVersion: 1,
            sessionID: "abc",
            files: [
                TransferFile(path: "session.json", size: 123, sha256: String(repeating: "0", count: 64)),
                TransferFile(path: "frames/000000/rgb.jpg", size: 456, sha256: String(repeating: "1", count: 64)),
                TransferFile(path: "frames/000000/depth.f32", size: 789, sha256: String(repeating: "2", count: 64)),
                TransferFile(path: "frames/000000/confidence.u8", size: 12, sha256: String(repeating: "3", count: 64)),
                TransferFile(path: "frames/000000/frame.json", size: 345, sha256: String(repeating: "4", count: 64))
            ]
        )
        XCTAssertEqual(manifestSHA256(manifest), "2ce25b90aa5a5b787a3dd14d3c1209a6db4f6a6250eeba167459f8bceb0ee50d")
    }

    private let json = """
    {
      "protocol_version": 1,
      "status": "verified",
      "session_id": "session-test",
      "manifest_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "verified_file_count": 25,
      "verified_total_bytes": 5021785
    }
    """.data(using: .utf8)!

    func testCanonicalVerifiedAcknowledgementDecodes() throws {
        let response = try JSONDecoder().decode(VerifiedResponse.self, from: json)
        XCTAssertEqual(response.verifiedFileCount, 25)
        XCTAssertEqual(response.verifiedTotalBytes, 5021785)
        XCTAssertTrue(SessionTransferService.isValidVerifiedAcknowledgement(
            response,
            sessionID: "session-test",
            manifestHash: String(repeating: "a", count: 64),
            fileCount: 25,
            totalBytes: 5021785
        ))
    }

    func testMismatchedFileCountBlocksAcknowledgement() throws {
        let response = try JSONDecoder().decode(VerifiedResponse.self, from: json)
        XCTAssertFalse(SessionTransferService.isValidVerifiedAcknowledgement(
            response,
            sessionID: "session-test",
            manifestHash: String(repeating: "a", count: 64),
            fileCount: 24,
            totalBytes: 5021785
        ))
    }

    func testMismatchedByteCountBlocksAcknowledgement() throws {
        let response = try JSONDecoder().decode(VerifiedResponse.self, from: json)
        XCTAssertFalse(SessionTransferService.isValidVerifiedAcknowledgement(
            response,
            sessionID: "session-test",
            manifestHash: String(repeating: "a", count: 64),
            fileCount: 25,
            totalBytes: 5021784
        ))
    }

    func testMismatchedHashAndSessionBlockAcknowledgement() throws {
        let response = try JSONDecoder().decode(VerifiedResponse.self, from: json)
        XCTAssertFalse(SessionTransferService.isValidVerifiedAcknowledgement(
            response,
            sessionID: "other-session",
            manifestHash: String(repeating: "a", count: 64),
            fileCount: 25,
            totalBytes: 5021785
        ))
        XCTAssertFalse(SessionTransferService.isValidVerifiedAcknowledgement(
            response,
            sessionID: "session-test",
            manifestHash: String(repeating: "b", count: 64),
            fileCount: 25,
            totalBytes: 5021785
        ))
    }
}
