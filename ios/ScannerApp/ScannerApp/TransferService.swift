import CryptoKit
import Foundation

enum TransferError: LocalizedError {
    case invalidServerURL
    case sessionNotFound
    case invalidSession
    case protocolMismatch
    case serverRejected(String)
    case invalidAcknowledgement

    var errorDescription: String? {
        switch self {
        case .invalidServerURL: return "Enter a valid Windows server URL, for example http://192.168.1.50:8765"
        case .sessionNotFound: return "Completed scan session was not found"
        case .invalidSession: return "Session metadata is not completed"
        case .protocolMismatch: return "Windows receiver protocol version is unsupported"
        case .serverRejected(let message): return message
        case .invalidAcknowledgement: return "Windows receiver did not return a verified acknowledgement"
        }
    }
}

struct TransferSettings {
    static let userDefaultsKey = "scanner.windowsServerURL"

    static var serverURL: String {
        get { UserDefaults.standard.string(forKey: userDefaultsKey) ?? "" }
        set { UserDefaults.standard.set(newValue.trimmingCharacters(in: .whitespacesAndNewlines), forKey: userDefaultsKey) }
    }
}

private struct TransferFile: Codable {
    let path: String
    let size: Int64
    let sha256: String
}

private struct TransferManifest: Codable {
    let protocolVersion: Int
    let sessionID: String
    let files: [TransferFile]

    enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol_version"
        case sessionID = "session_id"
        case files
    }
}

private struct BeginResponse: Codable {
    let protocolVersion: Int
    let status: String
    let sessionID: String
    let missing: [String]?

    enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol_version"
        case status
        case sessionID = "session_id"
        case missing
    }
}

private struct VerifiedResponse: Codable {
    let protocolVersion: Int
    let status: String
    let sessionID: String
    let fileCount: Int?

    enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol_version"
        case status
        case sessionID = "session_id"
        case fileCount = "file_count"
    }
}

struct TransferResult {
    let sessionID: String
    let fileCount: Int
}

final class SessionTransferService {
    static let protocolVersion = 1

    private let fileManager = FileManager.default
    private let captureService: FrameCaptureService
    private let session: URLSession

    init(captureService: FrameCaptureService, session: URLSession = .shared) {
        self.captureService = captureService
        self.session = session
    }

    func transfer(sessionID: String, serverURLString: String) async throws -> TransferResult {
        guard let serverURL = URL(string: serverURLString.trimmingCharacters(in: .whitespacesAndNewlines)),
              let scheme = serverURL.scheme,
              ["http", "https"].contains(scheme),
              serverURL.host != nil
        else { throw TransferError.invalidServerURL }

        let sessionDirectory = try captureService.sessionDirectory(for: sessionID)
        guard fileManager.fileExists(atPath: sessionDirectory.path) else { throw TransferError.sessionNotFound }
        let manifest = try makeManifest(sessionID: sessionID, directory: sessionDirectory)
        let encoder = JSONEncoder()
        let begin = try await post(
            endpoint(serverURL, components: ["v1", "sessions", "begin"]),
            body: try encoder.encode(manifest),
            decode: BeginResponse.self
        )
        guard begin.protocolVersion == Self.protocolVersion, begin.sessionID == sessionID else {
            throw TransferError.protocolMismatch
        }
        if begin.status == "verified" {
            try captureService.deleteSession(sessionID: sessionID)
            return TransferResult(sessionID: sessionID, fileCount: manifest.files.count)
        }
        guard begin.status == "ready" else { throw TransferError.invalidAcknowledgement }

        let filesByPath = Dictionary(uniqueKeysWithValues: manifest.files.map { ($0.path, $0) })
        for relativePath in begin.missing ?? [] {
            guard let expected = filesByPath[relativePath] else { throw TransferError.protocolMismatch }
            let localURL = sessionDirectory.appendingPathComponent(relativePath)
            guard fileManager.fileExists(atPath: localURL.path) else { throw TransferError.sessionNotFound }
            try await upload(
                endpoint(serverURL, components: ["v1", "sessions", sessionID, "files", relativePath]),
                fileURL: localURL,
                expected: expected
            )
        }

        let verified = try await post(
            endpoint(serverURL, components: ["v1", "sessions", sessionID, "finalize"]),
            body: try encoder.encode(manifest),
            decode: VerifiedResponse.self
        )
        guard verified.protocolVersion == Self.protocolVersion,
              verified.status == "verified",
              verified.sessionID == sessionID
        else { throw TransferError.invalidAcknowledgement }

        try captureService.deleteSession(sessionID: sessionID)
        return TransferResult(sessionID: sessionID, fileCount: verified.fileCount ?? manifest.files.count)
    }

    private func makeManifest(sessionID: String, directory: URL) throws -> TransferManifest {
        let sessionData = try Data(contentsOf: directory.appendingPathComponent("session.json"))
        guard let metadata = try? JSONDecoder().decode(SessionMetadata.self, from: sessionData),
              metadata.status == "completed",
              metadata.sessionID == sessionID
        else { throw TransferError.invalidSession }

        guard let enumerator = fileManager.enumerator(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else { throw TransferError.sessionNotFound }
        var files: [TransferFile] = []
        for case let fileURL as URL in enumerator {
            guard (try? fileURL.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true else { continue }
            let relative = fileURL.path
                .replacingOccurrences(of: directory.path + "/", with: "")
                .replacingOccurrences(of: "\\", with: "/")
            let data = try Data(contentsOf: fileURL, options: [.mappedIfSafe])
            let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            files.append(TransferFile(path: relative, size: Int64(data.count), sha256: digest))
        }
        return TransferManifest(protocolVersion: Self.protocolVersion, sessionID: sessionID, files: files.sorted { $0.path < $1.path })
    }

    private func endpoint(_ base: URL, components: [String]) -> URL {
        components.reduce(base) { $0.appendingPathComponent($1) }
    }

    private func baseRequest(_ url: URL, method: String, contentType: String) -> URLRequest {
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.setValue(String(Self.protocolVersion), forHTTPHeaderField: "X-Protocol-Version")
        request.setValue(contentType, forHTTPHeaderField: "Content-Type")
        return request
    }

    private func post<T: Decodable>(_ url: URL, body: Data, decode: T.Type) async throws -> T {
        var request = baseRequest(url, method: "POST", contentType: "application/json")
        request.httpBody = body
        let (data, response) = try await session.data(for: request)
        try validateHTTP(response, data: data)
        return try JSONDecoder().decode(T.self, from: data)
    }

    private func upload(_ url: URL, fileURL: URL, expected: TransferFile) async throws {
        var request = baseRequest(url, method: "PUT", contentType: "application/octet-stream")
        request.setValue(String(expected.size), forHTTPHeaderField: "Content-Length")
        let (data, response) = try await session.upload(for: request, fromFile: fileURL)
        try validateHTTP(response, data: data)
    }

    private func validateHTTP(_ response: URLResponse, data: Data) throws {
        guard let http = response as? HTTPURLResponse else { throw TransferError.serverRejected("Invalid HTTP response") }
        guard (200..<300).contains(http.statusCode) else {
            let message = (try? JSONDecoder().decode([String: String].self, from: data)["error"]) ?? "Server rejected transfer"
            throw TransferError.serverRejected(message)
        }
    }
}
