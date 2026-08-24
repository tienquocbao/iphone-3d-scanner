import XCTest

final class TransferAckTests: XCTestCase {
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
