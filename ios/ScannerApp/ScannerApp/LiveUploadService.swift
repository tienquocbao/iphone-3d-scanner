import CryptoKit
import Foundation

struct LiveUploadItem: Sendable {
    let sessionID: String
    let relativePath: String
    let fileURL: URL
    let size: Int64
    let sha256: String
}

struct LiveReceiverStatus: Decodable, Sendable {
    let state: String
    let uploadedFiles: Int
    let uploadedBytes: Int64
    let readyFrames: Int
    let processedFrames: Int
    let uploadBacklogFiles: Int

    enum CodingKeys: String, CodingKey {
        case state
        case uploadedFiles = "uploaded_files"
        case uploadedBytes = "uploaded_bytes"
        case readyFrames = "ready_frames"
        case processedFrames = "processed_frames"
        case uploadBacklogFiles = "upload_backlog_files"
    }
}

actor LiveUploadQueue {
    static let defaultConcurrency = 4

    private let maxConcurrent: Int
    private let upload: @Sendable (LiveUploadItem) async throws -> Void
    private var pending: [LiveUploadItem] = []
    private var active = 0
    private var finished = false

    init(maxConcurrent: Int = LiveUploadQueue.defaultConcurrency, upload: @escaping @Sendable (LiveUploadItem) async throws -> Void) {
        self.maxConcurrent = max(1, maxConcurrent)
        self.upload = upload
    }

    func enqueue(_ item: LiveUploadItem) {
        guard !finished else { return }
        pending.append(item)
        schedule()
    }

    func finishAndWait() async {
        finished = true
        while active > 0 || !pending.isEmpty {
            schedule()
            try? await Task.sleep(for: .milliseconds(50))
        }
    }

    func backlogCount() -> Int { pending.count + active }

    private func schedule() {
        while active < maxConcurrent, !pending.isEmpty {
            let item = pending.removeFirst()
            active += 1
            Task {
                for attempt in 0..<3 {
                    do {
                        try await upload(item)
                        break
                    } catch {
                        if attempt < 2 {
                            try? await Task.sleep(for: .seconds(1))
                        }
                        // Final BEGIN reconciliation remains authoritative. A file that
                        // still fails after bounded retries stays safely on the iPhone.
                    }
                }
                await completed()
            }
        }
    }

    private func completed() {
        active = max(0, active - 1)
        schedule()
    }
}

final class LiveUploadService {
    private let fileManager = FileManager.default
    private let session: URLSession
    private let token: String
    private let baseURL: URL
    private lazy var queue = LiveUploadQueue { [weak self] item in
        guard let self else { return }
        try await self.upload(item)
    }

    init(serverURLString: String, authToken: String, session: URLSession = .shared) throws {
        guard let url = URL(string: serverURLString.trimmingCharacters(in: .whitespacesAndNewlines)),
              let scheme = url.scheme, ["http", "https"].contains(scheme), url.host != nil else {
            throw TransferError.invalidServerURL
        }
        self.baseURL = url
        self.token = authToken
        self.session = session
    }

    func start(sessionID: String) async throws {
        var request = request(path: ["api", "v1", "live", "sessions", sessionID, "start"], method: "POST", contentType: "application/json")
        request.httpBody = Data("{}".utf8)
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw TransferError(stage: .health, message: "Live receiver start failed", responseBody: String(data: data.prefix(8192), encoding: .utf8))
        }
    }

    func enqueueFrame(sessionID: String, frameDirectory: URL, frameIndex: Int) async {
        let names = ["rgb.jpg", "depth.f32", "confidence.u8", "frame.json"]
        for name in names {
            let url = frameDirectory.appendingPathComponent(name)
            guard let attributes = try? fileManager.attributesOfItem(atPath: url.path),
                  let size = attributes[.size] as? NSNumber,
                  let data = try? Data(contentsOf: url) else { continue }
            let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            await queue.enqueue(LiveUploadItem(
                sessionID: sessionID,
                relativePath: String(format: "frames/%06d/%@", frameIndex, name),
                fileURL: url,
                size: size.int64Value,
                sha256: digest
            ))
        }
    }

    func finishAndWait() async {
        await queue.finishAndWait()
    }

    func status(sessionID: String) async throws -> LiveReceiverStatus {
        let request = request(path: ["api", "v1", "live", "sessions", sessionID, "status"], method: "GET", contentType: "application/json")
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw TransferError(stage: .health, message: "Live status request failed", responseBody: String(data: data.prefix(8192), encoding: .utf8))
        }
        return try JSONDecoder().decode(LiveReceiverStatus.self, from: data)
    }

    private func upload(_ item: LiveUploadItem) async throws {
        var request = request(path: ["api", "v1", "live", "sessions", item.sessionID, "files", item.relativePath], method: "PUT", contentType: "application/octet-stream")
        request.setValue(String(item.size), forHTTPHeaderField: "Content-Length")
        request.setValue(item.sha256, forHTTPHeaderField: "X-File-SHA256")
        let (data, response) = try await session.upload(for: request, fromFile: item.fileURL)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw TransferError(stage: .upload, message: "Live upload failed", filePath: item.relativePath, responseBody: String(data: data.prefix(8192), encoding: .utf8))
        }
    }

    private func request(path: [String], method: String, contentType: String) -> URLRequest {
        let url = path.reduce(baseURL) { current, component in
            component.split(separator: "/").reduce(current) { $0.appendingPathComponent(String($1)) }
        }
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.setValue("1", forHTTPHeaderField: "X-Protocol-Version")
        request.setValue(contentType, forHTTPHeaderField: "Content-Type")
        if !token.isEmpty { request.setValue("Bearer (token)", forHTTPHeaderField: "Authorization") }
        return request
    }
}
