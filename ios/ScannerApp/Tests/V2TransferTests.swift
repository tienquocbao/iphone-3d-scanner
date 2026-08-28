import XCTest

final class V2TransferTests: XCTestCase {
    func testVerifiedSameSessionAllowsDeletion() throws {
        let data = Data("{\"protocol_version\":2,\"status\":\"verified\",\"session_id\":\"session-a\"}".utf8)
        let ack = try JSONDecoder().decode(FinalizeAck.self, from: data)
        XCTAssertTrue(SessionTransferService.canDeleteLocalSession(localSessionID: "session-a", ack: ack))
    }

    func testInvalidV2AcknowledgementPreservesLocalSession() {
        XCTAssertFalse(SessionTransferService.canDeleteLocalSession(localSessionID: "session-a", ack: FinalizeAck(protocolVersion: 2, status: "ready", sessionID: "session-a")))
        XCTAssertFalse(SessionTransferService.canDeleteLocalSession(localSessionID: "session-a", ack: FinalizeAck(protocolVersion: 1, status: "verified", sessionID: "session-a")))
        XCTAssertFalse(SessionTransferService.canDeleteLocalSession(localSessionID: "session-a", ack: FinalizeAck(protocolVersion: 2, status: "verified", sessionID: "session-b")))
    }

    func testDefaultBatchTargetIsSixteenMiB() {
        XCTAssertEqual(BatchTransferPolicy.default.targetBodyBytes, 16 * 1024 * 1024)
    }
}
