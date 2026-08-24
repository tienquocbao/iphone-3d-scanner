import CryptoKit
import Foundation
import Security

enum TransferStage: String {
    case health = "HEALTH"
    case manifest = "MANIFEST"
    case begin = "BEGIN"
    case upload = "PUT"
    case finalize = "FINALIZE"
    case ackValidation = "ACK VALIDATION"
}

struct TransferError: LocalizedError {
    let stage: TransferStage
    let message: String
    let filePath: String?
    let statusCode: Int?
    let responseBody: String?
    let underlyingDescription: String?

    init(
        stage: TransferStage,
        message: String,
        filePath: String? = nil,
        statusCode: Int? = nil,
        responseBody: String? = nil,
        underlying: Error? = nil
    ) {
        self.stage = stage
        self.message = message
        self.filePath = filePath
        self.statusCode = statusCode
        self.responseBody = responseBody
        self.underlyingDescription = underlying?.localizedDescription
    }

    var errorDescription: String? {
        var lines = ["Stage: \(stage.rawValue)"]
        if let filePath { lines.append("File: \(filePath)") }
        if let statusCode { lines.append("HTTP: \(statusCode)") }
        lines.append(message)
        if let responseBody, !responseBody.isEmpty { lines.append(responseBody) }
        if let underlyingDescription { lines.append(underlyingDescription) }
        return lines.joined(separator: "\n")
    }
}

extension TransferError {
    static var invalidServerURL: TransferError { TransferError(stage: .health, message: "Invalid receiver URL") }
    static var sessionNotFound: TransferError { TransferError(stage: .manifest, message: "Completed scan session was not found") }
    static var invalidSession: TransferError { TransferError(stage: .manifest, message: "Session metadata is not completed") }
    static var protocolMismatch: TransferError { TransferError(stage: .ackValidation, message: "Protocol or session ID mismatch") }
    static var invalidAcknowledgement: TransferError { TransferError(stage: .ackValidation, message: "Invalid VERIFIED acknowledgement") }
    static func serverRejected(_ message: String, stage: TransferStage = .health, filePath: String? = nil, statusCode: Int? = nil, responseBody: String? = nil, underlying: Error? = nil) -> TransferError {
        TransferError(stage: stage, message: message, filePath: filePath, statusCode: statusCode, responseBody: responseBody, underlying: underlying)
    }
}

struct TransferSettings {
    static let userDefaultsKey = "scanner.windowsServerURL"

    static var serverURL: String {
        get { UserDefaults.standard.string(forKey: userDefaultsKey) ?? "" }
        set { UserDefaults.standard.set(newValue.trimmingCharacters(in: .whitespacesAndNewlines), forKey: userDefaultsKey) }
    }
}

enum KeychainStore {
    private static let service = "com.local.iphone3dscanner.transfer"
    private static let account = "receiver-bearer-token"

    static func load() -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        var result: AnyObject?
        guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
              let data = result as? Data,
              let value = String(data: data, encoding: .utf8),
              !value.isEmpty else { return nil }
        return value
    }

    static func save(_ value: String) throws {
        let data = Data(value.utf8)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account
        ]
        let attributes: [String: Any] = [kSecValueData as String: data]
        let status = SecItemUpdate(query as CFDictionary, attributes as CFDictionary)
        if status == errSecItemNotFound {
            var item = query
            item[kSecValueData as String] = data
            guard SecItemAdd(item as CFDictionary, nil) == errSecSuccess else {
                throw TransferError.serverRejected("Could not save receiver token in Keychain")
            }
        } else if status != errSecSuccess {
            throw TransferError.serverRejected("Could not update receiver token in Keychain")
        }
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
    let manifestSHA256: String?
    let fileCount: Int?
    let totalBytes: Int64?

    enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol_version"
        case status
        case sessionID = "session_id"
        case missing
        case manifestSHA256 = "manifest_sha256"
        case fileCount = "file_count"
        case totalBytes = "total_bytes"
    }
}

private struct VerifiedResponse: Codable {
    let protocolVersion: Int
    let status: String
    let sessionID: String
    let fileCount: Int?
    let manifestSHA256: String?
    let totalBytes: Int64?

    enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol_version"
        case status
        case sessionID = "session_id"
        case fileCount = "file_count"
        case manifestSHA256 = "manifest_sha256"
        case totalBytes = "total_bytes"
    }
}

private struct HealthResponse: Codable {
    let protocolVersion: Int
    let status: String

    enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol_version"
        case status
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

    func transfer(sessionID: String, serverURLString: String, authToken: String) async throws -> TransferResult {
        guard let serverURL = URL(string: serverURLString.trimmingCharacters(in: .whitespacesAndNewlines)),
              let scheme = serverURL.scheme,
              ["http", "https"].contains(scheme),
              serverURL.host != nil
        else { throw TransferError.invalidServerURL }

        let sessionDirectory = try captureService.sessionDirectory(for: sessionID)
        guard fileManager.fileExists(atPath: sessionDirectory.path) else { throw TransferError.sessionNotFound }
        let manifest = try makeManifest(sessionID: sessionID, directory: sessionDirectory)
        let encoder = JSONEncoder()
        let beginBody = try encoder.encode(manifest)
        print("TRANSFER manifest files=\(manifest.files.count) payload_bytes=\(beginBody.count)")
        let begin = try await post(
            endpoint(serverURL, components: ["v1", "sessions", "begin"]),
            body: beginBody,
            token: authToken,
            stage: .begin,
            decode: BeginResponse.self
        )
        guard begin.protocolVersion == Self.protocolVersion, begin.sessionID == sessionID else {
            throw TransferError(stage: .begin, message: "Protocol or session ID mismatch in BEGIN response")
        }
        if begin.status == "verified" {
            guard begin.manifestSHA256 == manifestHash(manifest),
                  begin.fileCount == manifest.files.count,
                  begin.totalBytes == manifestTotalBytes(manifest)
            else { throw TransferError(stage: .ackValidation, message: "Verified BEGIN response does not match manifest") }
            try captureService.deleteSession(sessionID: sessionID)
            return TransferResult(sessionID: sessionID, fileCount: manifest.files.count)
        }
        guard begin.status == "ready" else { throw TransferError(stage: .begin, message: "Unexpected BEGIN status: \(begin.status)") }

        let filesByPath = Dictionary(uniqueKeysWithValues: manifest.files.map { ($0.path, $0) })
        for relativePath in begin.missing ?? [] {
            guard let expected = filesByPath[relativePath] else { throw TransferError(stage: .upload, message: "Receiver requested a file not present in manifest", filePath: relativePath) }
            let localURL = sessionDirectory.appendingPathComponent(relativePath)
            guard fileManager.fileExists(atPath: localURL.path) else { throw TransferError(stage: .upload, message: "Local file was not found", filePath: relativePath) }
            try await upload(
                endpoint(serverURL, components: ["v1", "sessions", sessionID, "files", relativePath]),
                fileURL: localURL,
                expected: expected,
                token: authToken,
                stage: .upload
            )
        }

        let verified = try await post(
            endpoint(serverURL, components: ["v1", "sessions", sessionID, "finalize"]),
            body: try encoder.encode(manifest),
            token: authToken,
            stage: .finalize,
            decode: VerifiedResponse.self
        )
        guard verified.protocolVersion == Self.protocolVersion,
              verified.status == "verified",
              verified.sessionID == sessionID,
              verified.manifestSHA256 == manifestHash(manifest),
              verified.fileCount == manifest.files.count,
              verified.totalBytes == manifestTotalBytes(manifest)
        else { throw TransferError(stage: .ackValidation, message: "Missing or invalid VERIFIED acknowledgement") }

        try captureService.deleteSession(sessionID: sessionID)
        return TransferResult(sessionID: sessionID, fileCount: verified.fileCount ?? manifest.files.count)
    }

    func testConnection(serverURLString: String, authToken: String) async throws {
        guard let serverURL = URL(string: serverURLString.trimmingCharacters(in: .whitespacesAndNewlines)),
              let scheme = serverURL.scheme,
              ["http", "https"].contains(scheme),
              serverURL.host != nil else { throw TransferError.invalidServerURL }
        var request = baseRequest(endpoint(serverURL, components: ["api", "v1", "health"]), method: "GET", contentType: "application/json", token: authToken)
        request.httpBody = nil
        let (data, response) = try await session.data(for: request)
        try validateHTTP(response, data: data, stage: .health)
        guard let object = try? JSONDecoder().decode(HealthResponse.self, from: data),
              object.protocolVersion == Self.protocolVersion,
              object.status == "ok" else { throw TransferError.invalidAcknowledgement }
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
            let data: Data
            do {
                data = try Data(contentsOf: fileURL, options: [.mappedIfSafe])
            } catch {
                throw TransferError(stage: .manifest, message: "Cannot read file", filePath: relative, underlying: error)
            }
            let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            files.append(TransferFile(path: relative, size: Int64(data.count), sha256: digest))
        }
        return TransferManifest(protocolVersion: Self.protocolVersion, sessionID: sessionID, files: files.sorted { $0.path < $1.path })
    }

    private func endpoint(_ base: URL, components: [String]) -> URL {
        components.reduce(base) { current, component in
            component.split(separator: "/", omittingEmptySubsequences: true)
                .reduce(current) { $0.appendingPathComponent(String($1)) }
        }
    }

    private func baseRequest(_ url: URL, method: String, contentType: String, token: String) -> URLRequest {
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.setValue(String(Self.protocolVersion), forHTTPHeaderField: "X-Protocol-Version")
        request.setValue(contentType, forHTTPHeaderField: "Content-Type")
        if !token.isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        return request
    }

    private func post<T: Decodable>(_ url: URL, body: Data, token: String, stage: TransferStage, decode: T.Type) async throws -> T {
        var request = baseRequest(url, method: "POST", contentType: "application/json", token: token)
        request.httpBody = body
        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: request)
        } catch {
            throw TransferError(stage: stage, message: "URLSession request failed", underlying: error)
        }
        try validateHTTP(response, data: data, stage: stage)
        do {
            return try JSONDecoder().decode(T.self, from: data)
        } catch {
            throw TransferError(stage: stage, message: "Cannot decode server response", responseBody: boundedBody(data), underlying: error)
        }
    }

    private func upload(_ url: URL, fileURL: URL, expected: TransferFile, token: String, stage: TransferStage) async throws {
        var request = baseRequest(url, method: "PUT", contentType: "application/octet-stream", token: token)
        request.setValue(String(expected.size), forHTTPHeaderField: "Content-Length")
        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.upload(for: request, fromFile: fileURL)
        } catch {
            throw TransferError(stage: stage, message: "URLSession upload failed", filePath: expected.path, underlying: error)
        }
        try validateHTTP(response, data: data, stage: stage, filePath: expected.path)
    }

    private func validateHTTP(_ response: URLResponse, data: Data, stage: TransferStage, filePath: String? = nil) throws {
        guard let http = response as? HTTPURLResponse else { throw TransferError.serverRejected("Invalid HTTP response") }
        guard (200..<300).contains(http.statusCode) else {
            throw TransferError(stage: stage, message: "Server rejected request", filePath: filePath, statusCode: http.statusCode, responseBody: boundedBody(data))
        }
    }

    private func boundedBody(_ data: Data) -> String {
        String(data: data.prefix(8192), encoding: .utf8) ?? "<non-UTF8 response>"
    }

    private func manifestHash(_ manifest: TransferManifest) -> String {
        let data = (try? JSONEncoder().encode(manifest)) ?? Data()
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private func manifestTotalBytes(_ manifest: TransferManifest) -> Int64 {
        manifest.files.reduce(Int64(0)) { total, file in total + file.size }
    }
}
