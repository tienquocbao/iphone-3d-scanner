import XCTest

final class LiveUploadRequestTests: XCTestCase {
    func testAllLiveRequestsUseTheActualBearerToken() throws {
        let service = try LiveUploadService(serverURLString: "https://receiver.example", authToken: "test-token")
        let item = LiveUploadItem(
            sessionID: "session-test",
            relativePath: "frames/000000/rgb.jpg",
            fileURL: URL(fileURLWithPath: "/tmp/rgb.jpg"),
            size: 3,
            sha256: String(repeating: "a", count: 64)
        )

        let start = service.makeRequest(
            path: ["api", "v1", "live", "sessions", "session-test", "start"],
            method: "POST",
            contentType: "application/json"
        )
        let put = service.makeLiveFileRequest(item)
        let status = service.makeRequest(
            path: ["api", "v1", "live", "sessions", "session-test", "status"],
            method: "GET",
            contentType: "application/json"
        )

        XCTAssertEqual(start.value(forHTTPHeaderField: "Authorization"), "Bearer test-token")
        XCTAssertEqual(put.value(forHTTPHeaderField: "Authorization"), "Bearer test-token")
        XCTAssertEqual(status.value(forHTTPHeaderField: "Authorization"), "Bearer test-token")
    }

    func testEmptyTokenOmitsAuthorizationForEveryLiveRequest() throws {
        let service = try LiveUploadService(serverURLString: "https://receiver.example", authToken: "")
        let item = LiveUploadItem(
            sessionID: "session-test",
            relativePath: "frames/000000/rgb.jpg",
            fileURL: URL(fileURLWithPath: "/tmp/rgb.jpg"),
            size: 3,
            sha256: String(repeating: "a", count: 64)
        )
        let requests = [
            service.makeRequest(path: ["api", "v1", "live", "sessions", "session-test", "start"], method: "POST", contentType: "application/json"),
            service.makeLiveFileRequest(item),
            service.makeRequest(path: ["api", "v1", "live", "sessions", "session-test", "status"], method: "GET", contentType: "application/json")
        ]
        for request in requests {
            XCTAssertNil(request.value(forHTTPHeaderField: "Authorization"))
        }
    }

    func testFailedLiveItemIsRecordedAfterBoundedRetries() async {
        let item = LiveUploadItem(
            sessionID: "session-test",
            relativePath: "frames/000000/rgb.jpg",
            fileURL: URL(fileURLWithPath: "/tmp/rgb.jpg"),
            size: 3,
            sha256: String(repeating: "a", count: 64)
        )
        let queue = LiveUploadQueue(maxConcurrent: 4) { _ in
            throw TransferError(stage: .upload, message: "simulated network failure")
        }
        await queue.enqueue(item)
        await queue.finishAndWait()
        let status = await queue.status()
        XCTAssertEqual(status.backlog, 0)
        XCTAssertEqual(status.failed, 1)
    }

    func testReadyFramesDriveCompleteUploadedFrameCount() {
        let status = LiveReceiverStatus(
            state: "recording",
            uploadedFiles: 7,
            uploadedBytes: 100,
            readyFrames: 1,
            processedFrames: 0,
            uploadBacklogFiles: 3
        )
        XCTAssertEqual(ARSessionManager.completeUploadedFrameCount(for: status), 1)
    }
}
