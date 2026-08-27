import XCTest

final class TransferAckTests: XCTestCase {
    private let json = """
    {
      "protocol_version": 1,
      "status": "verified",
      "session_id": "session-test"
    }
    """.data(using: .utf8)!

    func testVerifiedAcknowledgementAllowsDeletion() throws {
        let response = try JSONDecoder().decode(FinalizeAck.self, from: json)
        XCTAssertTrue(SessionTransferService.canDeleteLocalSession(localSessionID: "session-test", ack: response))
    }

    func testWrongSessionBlocksDeletion() throws {
        let response = try JSONDecoder().decode(FinalizeAck.self, from: json)
        XCTAssertFalse(SessionTransferService.canDeleteLocalSession(localSessionID: "other-session", ack: response))
    }

    func testMalformedOrNonVerifiedAcknowledgementBlocksDeletion() throws {
        let malformed = Data("{}".utf8)
        XCTAssertThrowsError(try JSONDecoder().decode(FinalizeAck.self, from: malformed))

        let wrongStatus = Data("{\"protocol_version\":1,\"status\":\"ready\",\"session_id\":\"session-test\"}".utf8)
        let response = try JSONDecoder().decode(FinalizeAck.self, from: wrongStatus)
        XCTAssertFalse(SessionTransferService.canDeleteLocalSession(localSessionID: "session-test", ack: response))

        let wrongProtocol = Data("{\"protocol_version\":2,\"status\":\"verified\",\"session_id\":\"session-test\"}".utf8)
        let protocolResponse = try JSONDecoder().decode(FinalizeAck.self, from: wrongProtocol)
        XCTAssertFalse(SessionTransferService.canDeleteLocalSession(localSessionID: "session-test", ack: protocolResponse))
    }
}
