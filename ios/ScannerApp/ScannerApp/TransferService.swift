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

struct TransferPolicy {
    let maxConcurrentUploads: Int

    static let `default` = TransferPolicy(maxConcurrentUploads: 4)
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

struct TransferFile: Codable {
    let path: String
    let size: Int64
    let sha256: String
}

enum TransferPath {
    static func canonicalRelativePath(sessionRoot: URL, fileURL: URL) throws -> String {
        let root = sessionRoot.standardizedFileURL
        let file = fileURL.standardizedFileURL
        guard root.isFileURL, file.isFileURL else {
            throw TransferError(stage: .manifest, message: "Session paths must be file URLs")
        }

        let rootComponents = root.pathComponents
        let fileComponents = file.pathComponents
        guard fileComponents.count > rootComponents.count,
              Array(fileComponents.prefix(rootComponents.count)) == rootComponents else {
            throw TransferError(stage: .manifest, message: "File is outside the scan session", filePath: file.path)
        }

        let relativeComponents = Array(fileComponents.dropFirst(rootComponents.count))
        guard !relativeComponents.isEmpty else {
            throw TransferError(stage: .manifest, message: "Session file path is empty")
        }
        for component in relativeComponents {
            guard !component.isEmpty, component != ".", component != "..",
                  !component.contains("/"), !component.contains("\\") else {
                throw TransferError(stage: .manifest, message: "Invalid session file path", filePath: component)
            }
        }
        return relativeComponents.joined(separator: "/")
    }

    static func localURL(sessionRoot: URL, relativePath: String) -> URL {
        relativePath.split(separator: "/", omittingEmptySubsequences: false)
            .reduce(sessionRoot) { $0.appendingPathComponent(String($1), isDirectory: false) }
    }
}

struct TransferManifest: Codable {
    let protocolVersion: Int
    let sessionID: String
    let files: [TransferFile]

    enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol_version"
        case sessionID = "session_id"
        case files
    }
}

func canonicalManifestBytes(_ manifest: TransferManifest) -> Data {
    var lines = [String(manifest.protocolVersion), manifest.sessionID]
    for file in manifest.files.sorted(by: { $0.path < $1.path }) {
        lines.append(file.path)
        lines.append(String(file.size))
        lines.append(file.sha256)
    }
    return Data((lines.joined(separator: "\n") + "\n").utf8)
}

func manifestSHA256(_ manifest: TransferManifest) -> String {
    SHA256.hash(data: canonicalManifestBytes(manifest)).map { String(format: "%02x", $0) }.joined()
}

private struct BeginResponse: Codable {
    let protocolVersion: Int
    let status: String
    let sessionID: String
    let missing: [String]?
    let manifestSHA256: String?
    let verifiedFileCount: Int?
    let verifiedTotalBytes: Int64?

    enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol_version"
        case status
        case sessionID = "session_id"
        case missing
        case manifestSHA256 = "manifest_sha256"
        case verifiedFileCount = "verified_file_count"
        case verifiedTotalBytes = "verified_total_bytes"
    }
}

struct VerifiedResponse: Codable {
    let protocolVersion: Int
    let status: String
    let sessionID: String
    let verifiedFileCount: Int
    let manifestSHA256: String
    let verifiedTotalBytes: Int64

    enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol_version"
        case status
        case sessionID = "session_id"
        case verifiedFileCount = "verified_file_count"
        case manifestSHA256 = "manifest_sha256"
        case verifiedTotalBytes = "verified_total_bytes"
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
    private let policy: TransferPolicy

    init(captureService: FrameCaptureService, session: URLSession = .shared, policy: TransferPolicy = .default) {
        self.captureService = captureService
        self.session = session
        self.policy = policy
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
        let beginBody: Data
        do {
            beginBody = try encoder.encode(manifest)
        } catch {
            throw TransferError(stage: .begin, message: "Cannot encode manifest", underlying: error)
        }
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
        let localManifestHash = manifestSHA256(manifest)
        if begin.status == "verified" {
            guard begin.manifestSHA256 == localManifestHash,
                  begin.verifiedFileCount == manifest.files.count,
                  begin.verifiedTotalBytes == manifestTotalBytes(manifest)
            else { throw TransferError(stage: .ackValidation, message: "Verified BEGIN response does not match manifest") }
            try captureService.deleteSession(sessionID: sessionID)
            return TransferResult(sessionID: sessionID, fileCount: manifest.files.count)
        }
        guard begin.status == "ready" else { throw TransferError(stage: .begin, message: "Unexpected BEGIN status: \(begin.status)") }
        guard begin.manifestSHA256 == localManifestHash else {
            throw TransferError(stage: .begin, message: "Receiver manifest hash differs from local canonical manifest")
        }

        let filesByPath = Dictionary(uniqueKeysWithValues: manifest.files.map { ($0.path, $0) })
        var missingFiles: [TransferFile] = []
        for relativePath in begin.missing ?? [] {
            guard let expected = filesByPath[relativePath] else { throw TransferError(stage: .upload, message: "Receiver requested a file not present in manifest", filePath: relativePath) }
            missingFiles.append(expected)
        }
        let cursor = TransferUploadCursor(files: missingFiles)
        try await withThrowingTaskGroup(of: Void.self) { group in
            let workerCount = min(policy.maxConcurrentUploads, max(1, missingFiles.count))
            for _ in 0..<workerCount {
                group.addTask {
                    while let expected = await cursor.next() {
                        let localURL = TransferPath.localURL(sessionRoot: sessionDirectory, relativePath: expected.path)
                        guard self.fileManager.fileExists(atPath: localURL.path) else {
                            throw TransferError(stage: .upload, message: "Local file was not found", filePath: expected.path)
                        }
                        try await self.upload(
                            self.endpoint(serverURL, components: ["v1", "sessions", sessionID, "files", expected.path]),
                            fileURL: localURL,
                            expected: expected,
                            token: authToken,
                            stage: .upload
                        )
                    }
                }
            }
            try await group.waitForAll()
        }

        let verified = try await post(
            endpoint(serverURL, components: ["v1", "sessions", sessionID, "finalize"]),
            body: try encodeFinalizeManifest(manifest, encoder: encoder),
            token: authToken,
            stage: .finalize,
            decode: VerifiedResponse.self
        )
        guard Self.isValidVerifiedAcknowledgement(
            verified,
            sessionID: sessionID,
            manifestHash: localManifestHash,
            fileCount: manifest.files.count,
            totalBytes: manifestTotalBytes(manifest)
        )
        else { throw TransferError(stage: .ackValidation, message: "Missing or invalid VERIFIED acknowledgement") }

        try captureService.deleteSession(sessionID: sessionID)
        return TransferResult(sessionID: sessionID, fileCount: verified.verifiedFileCount)
    }

    func testConnection(serverURLString: String, authToken: String) async throws {
        guard let serverURL = URL(string: serverURLString.trimmingCharacters(in: .whitespacesAndNewlines)),
              let scheme = serverURL.scheme,
              ["http", "https"].contains(scheme),
              serverURL.host != nil else { throw TransferError.invalidServerURL }
        var request = baseRequest(endpoint(serverURL, components: ["api", "v1", "health"]), method: "GET", contentType: "application/json", token: authToken)
        request.httpBody = nil
        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: request)
        } catch {
            throw TransferError(stage: .health, message: "URLSession health request failed", underlying: error)
        }
        try validateHTTP(response, data: data, stage: .health)
        guard let object = try? JSONDecoder().decode(HealthResponse.self, from: data),
              object.protocolVersion == Self.protocolVersion,
              object.status == "ok" else { throw TransferError.invalidAcknowledgement }
    }

    private func makeManifest(sessionID: String, directory: URL) throws -> TransferManifest {
        let sessionData: Data
        do {
            sessionData = try Data(contentsOf: directory.appendingPathComponent("session.json"))
        } catch {
            throw TransferError(stage: .manifest, message: "Cannot read session.json", filePath: "session.json", underlying: error)
        }
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
            let relative = try TransferPath.canonicalRelativePath(sessionRoot: directory, fileURL: fileURL)
            let components = relative.split(separator: "/", omittingEmptySubsequences: false).map(String.init)
            let isSessionMetadata = relative == "session.json"
            let isFrameFile = components.count == 3
                && components[0] == "frames"
                && components[1].count == 6
                && components[1].allSatisfy { $0.isNumber }
                && ["rgb.jpg", "depth.f32", "confidence.u8", "frame.json"].contains(components[2])
            guard isSessionMetadata || isFrameFile else {
                throw TransferError(stage: .manifest, message: "Unexpected file in completed session", filePath: relative)
            }
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
        guard let http = response as? HTTPURLResponse else { throw TransferError.serverRejected("Invalid HTTP response", stage: stage, filePath: filePath) }
        guard (200..<300).contains(http.statusCode) else {
            throw TransferError(stage: stage, message: "Server rejected request", filePath: filePath, statusCode: http.statusCode, responseBody: boundedBody(data))
        }
    }

    private func boundedBody(_ data: Data) -> String {
        String(data: data.prefix(8192), encoding: .utf8) ?? "<non-UTF8 response>"
    }

    private func manifestTotalBytes(_ manifest: TransferManifest) -> Int64 {
        manifest.files.reduce(Int64(0)) { total, file in total + file.size }
    }

    static func isValidVerifiedAcknowledgement(
        _ response: VerifiedResponse,
        sessionID: String,
        manifestHash: String,
        fileCount: Int,
        totalBytes: Int64
    ) -> Bool {
        response.protocolVersion == protocolVersion
            && response.status == "verified"
            && response.sessionID == sessionID
            && response.manifestSHA256 == manifestHash
            && response.verifiedFileCount == fileCount
            && response.verifiedTotalBytes == totalBytes
    }

    private func encodeFinalizeManifest(_ manifest: TransferManifest, encoder: JSONEncoder) throws -> Data {
        do {
            return try encoder.encode(manifest)
        } catch {
            throw TransferError(stage: .finalize, message: "Cannot encode finalize manifest", underlying: error)
        }
    }
}

private actor TransferUploadCursor {
    private let files: [TransferFile]
    private var index = 0

    init(files: [TransferFile]) {
        self.files = files
    }

    func next() -> TransferFile? {
        guard index < files.count else { return nil }
        defer { index += 1 }
        return files[index]
    }
}
