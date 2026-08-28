import CryptoKit
import Foundation
import Security

enum TransferStage: String { case health = "HEALTH", plan = "PLAN", start = "START", batch = "BATCH", finalize = "FINALIZE", acknowledgement = "ACK" }

struct TransferError: LocalizedError {
    let stage: TransferStage; let message: String; let batchIndex: Int?; let statusCode: Int?; let responseBody: String?
    init(stage: TransferStage, message: String, batchIndex: Int? = nil, statusCode: Int? = nil, responseBody: String? = nil) { self.stage = stage; self.message = message; self.batchIndex = batchIndex; self.statusCode = statusCode; self.responseBody = responseBody }
    var errorDescription: String? { var lines=["Stage: \(stage.rawValue)"]; if let batchIndex { lines.append("Batch: \(batchIndex)") }; if let statusCode { lines.append("HTTP: \(statusCode)") }; lines.append(message); if let responseBody, !responseBody.isEmpty { lines.append(responseBody) }; return lines.joined(separator:"\n") }
}

struct TransferSettings { static let userDefaultsKey = "scanner.windowsServerURL"; static var serverURL: String { get { UserDefaults.standard.string(forKey: userDefaultsKey) ?? "" } set { UserDefaults.standard.set(newValue.trimmingCharacters(in: .whitespacesAndNewlines), forKey: userDefaultsKey) } } }

enum KeychainStore {
    private static let service = "com.local.iphone3dscanner.transfer"; private static let account = "receiver-bearer-token"
    static func load() -> String? { let q: [String: Any] = [kSecClass as String:kSecClassGenericPassword, kSecAttrService as String:service, kSecAttrAccount as String:account, kSecReturnData as String:true, kSecMatchLimit as String:kSecMatchLimitOne]; var result: AnyObject?; guard SecItemCopyMatching(q as CFDictionary, &result) == errSecSuccess, let data = result as? Data, let value = String(data:data, encoding:.utf8), !value.isEmpty else { return nil }; return value }
    static func save(_ value: String) throws { let data = Data(value.utf8); let q: [String: Any] = [kSecClass as String:kSecClassGenericPassword, kSecAttrService as String:service, kSecAttrAccount as String:account]; let status = SecItemUpdate(q as CFDictionary, [kSecValueData as String:data] as CFDictionary); if status == errSecItemNotFound { var add = q; add[kSecValueData as String] = data; guard SecItemAdd(add as CFDictionary, nil) == errSecSuccess else { throw TransferError(stage:.health, message:"Could not save receiver token") } } else if status != errSecSuccess { throw TransferError(stage:.health, message:"Could not update receiver token") } }
}

struct BatchTransferPolicy { let targetBodyBytes: Int64; static let `default` = BatchTransferPolicy(targetBodyBytes: 16 * 1024 * 1024) }
struct TransferProgress { let sentBytes: Int64; let totalBytes: Int64; let batchIndex: Int; let batchCount: Int }
struct TransferResult { let sessionID: String; let fileCount: Int }

private struct StartRequest: Codable { let protocolVersion:Int; let sessionID:String; let frameCount:Int; let batchCount:Int; let totalBytes:Int64; enum CodingKeys:String,CodingKey { case protocolVersion="protocol_version", sessionID="session_id", frameCount="frame_count", batchCount="batch_count", totalBytes="total_bytes" } }
private struct UploadStatus: Codable { let protocolVersion:Int; let sessionID:String; let receivedBatches:[Int]; let receivedBytes:Int64; enum CodingKeys:String,CodingKey { case protocolVersion="protocol_version", sessionID="session_id", receivedBatches="received_batches", receivedBytes="received_bytes" } }
private struct FinalizeRequest: Codable { let protocolVersion:Int; let sessionID:String; enum CodingKeys:String,CodingKey { case protocolVersion="protocol_version", sessionID="session_id" } }
struct FinalizeAck: Codable { let protocolVersion:Int; let status:String; let sessionID:String; enum CodingKeys:String,CodingKey { case protocolVersion="protocol_version", status, sessionID="session_id" } }
private struct HealthResponse: Codable { let protocolVersion:Int; let status:String; enum CodingKeys:String,CodingKey { case protocolVersion="protocol_version", status } }
private struct BatchPart: Codable { let path:String; let size:Int64; let sha256:String }
private struct BatchDescriptor { let index:Int; let parts:[BatchPart]; var sourceBytes:Int64 { parts.reduce(0) { $0 + $1.size } } }

final class SessionTransferService {
    static let protocolVersion = 2
    static let batchMagic = Data("IPHONE3D-BATCH-V2\n".utf8)
    private let fileManager = FileManager.default; private let captureService: FrameCaptureService; private let session: URLSession; private let policy: BatchTransferPolicy
    init(captureService: FrameCaptureService, session: URLSession = .shared, policy: BatchTransferPolicy = .default) { self.captureService=captureService; self.session=session; self.policy=policy }

    func transfer(sessionID:String, serverURLString:String, authToken:String, progress:@escaping (TransferProgress)->Void) async throws -> TransferResult {
        let base = try receiverURL(serverURLString); let root = try captureService.sessionDirectory(for:sessionID); let metadata = try completedMetadata(sessionID:sessionID, directory:root); let plan = try makePlan(root:root, frameCount:metadata.frameCount); let total = plan.reduce(0) { $0 + $1.sourceBytes }
        let start = StartRequest(protocolVersion:Self.protocolVersion, sessionID:sessionID, frameCount:metadata.frameCount, batchCount:plan.count, totalBytes:total)
        let status = try await post(endpoint(base,["api","v2","sessions",sessionID,"start"]), body:try JSONEncoder().encode(start), token:authToken, stage:.start, type:UploadStatus.self)
        guard status.protocolVersion == Self.protocolVersion, status.sessionID == sessionID else { throw TransferError(stage:.start,message:"Protocol or session ID mismatch") }
        var received = Set(status.receivedBatches); var sent = plan.filter { received.contains($0.index) }.reduce(Int64(0)) { $0 + $1.sourceBytes }
        for descriptor in plan where !received.contains(descriptor.index) {
            let batch = try buildBatch(descriptor, root:root); defer { try? fileManager.removeItem(at:batch.url) }
            try await upload(endpoint(base,["api","v2","sessions",sessionID,"batches",String(descriptor.index)]), batch:batch, token:authToken, index:descriptor.index)
            received.insert(descriptor.index); sent += descriptor.sourceBytes; progress(TransferProgress(sentBytes:min(sent,total),totalBytes:total,batchIndex:descriptor.index+1,batchCount:plan.count))
        }
        let ack = try await post(endpoint(base,["api","v2","sessions",sessionID,"finalize"]), body:try JSONEncoder().encode(FinalizeRequest(protocolVersion:Self.protocolVersion,sessionID:sessionID)), token:authToken, stage:.finalize, type:FinalizeAck.self)
        guard Self.canDeleteLocalSession(localSessionID:sessionID, ack:ack) else { throw TransferError(stage:.acknowledgement,message:"Missing or invalid VERIFIED acknowledgement") }
        try captureService.deleteSession(sessionID:sessionID); return TransferResult(sessionID:sessionID,fileCount:plan.reduce(0) { $0 + $1.parts.count })
    }

    func testConnection(serverURLString:String, authToken:String) async throws { let base=try receiverURL(serverURLString); var req=request(endpoint(base,["api","v2","health"]),method:"GET",token:authToken,contentType:"application/json"); req.httpBody=nil; let(data,response)=try await session.data(for:req); try validate(response,data:data,stage:.health); let health=try JSONDecoder().decode(HealthResponse.self,from:data); guard health.protocolVersion == Self.protocolVersion, health.status == "ok" else { throw TransferError(stage:.health,message:"Unexpected receiver health response") } }
    static func canDeleteLocalSession(localSessionID:String, ack:FinalizeAck)->Bool { ack.protocolVersion == protocolVersion && ack.status == "verified" && ack.sessionID == localSessionID }

    private func completedMetadata(sessionID:String,directory:URL)throws->SessionMetadata { guard let data=try? Data(contentsOf:directory.appendingPathComponent("session.json")), let metadata=try? JSONDecoder().decode(SessionMetadata.self,from:data), metadata.status == "completed", metadata.sessionID == sessionID else { throw TransferError(stage:.plan,message:"Completed session metadata was not found") }; return metadata }
    private func makePlan(root:URL,frameCount:Int)throws->[BatchDescriptor] { var groups:[[BatchPart]]=[[try part(root,"session.json")]]; for index in 0..<frameCount { let prefix=String(format:"frames/%06d",index); groups.append(try ["rgb.jpg","depth.f32","confidence.u8","frame.json"].map { try part(root,"\(prefix)/\($0)") }) }; var result:[BatchDescriptor]=[]; var current:[BatchPart]=[]; var size:Int64=0; for group in groups { let groupSize=group.reduce(0) { $0+$1.size }; if !current.isEmpty && size+groupSize > policy.targetBodyBytes { result.append(BatchDescriptor(index:result.count,parts:current)); current=[]; size=0 }; current += group; size += groupSize }; if !current.isEmpty { result.append(BatchDescriptor(index:result.count,parts:current)) }; return result }
    private func part(_ root:URL,_ path:String)throws->BatchPart { let url=localURL(root,path); return BatchPart(path:path,size:try fileSize(url),sha256:try sha256(url)) }
    private func buildBatch(_ descriptor:BatchDescriptor,root:URL)throws->(url:URL,bytes:Int64,sha256:String) { let dir=fileManager.temporaryDirectory.appendingPathComponent("iphone3d-v2-batches",isDirectory:true); try fileManager.createDirectory(at:dir,withIntermediateDirectories:true); let url=dir.appendingPathComponent("batch-\(UUID().uuidString).bin"); guard fileManager.createFile(atPath:url.path,contents:nil) else { throw TransferError(stage:.batch,message:"Cannot create temporary batch",batchIndex:descriptor.index) }; let output=try FileHandle(forWritingTo:url); defer { try? output.close() }; var hasher=SHA256(); func write(_ data:Data)throws { try output.write(contentsOf:data); hasher.update(data:data) }; try write(Self.batchMagic); let encoder=JSONEncoder(); encoder.outputFormatting=[.sortedKeys]; for part in descriptor.parts { try write(try encoder.encode(part)); try write(Data("\n".utf8)); let input=try FileHandle(forReadingFrom:localURL(root,part.path)); defer { try? input.close() }; var copied:Int64=0; while let chunk=try input.read(upToCount:1024*1024),!chunk.isEmpty { try write(chunk); copied += Int64(chunk.count) }; guard copied == part.size else { throw TransferError(stage:.batch,message:"Source file changed while batching",batchIndex:descriptor.index) }; try write(Data("\n".utf8)) }; return (url,try fileSize(url),hasher.finalize().map { String(format:"%02x",$0) }.joined()) }
    private func upload(_ url:URL,batch:(url:URL,bytes:Int64,sha256:String),token:String,index:Int) async throws { var req=request(url,method:"PUT",token:token,contentType:"application/vnd.iphone3d.batch-v2"); req.setValue(String(batch.bytes),forHTTPHeaderField:"Content-Length"); req.setValue(batch.sha256,forHTTPHeaderField:"X-Batch-SHA256"); let(data,response)=try await session.upload(for:req,fromFile:batch.url); try validate(response,data:data,stage:.batch,batchIndex:index) }
    private func post<T:Decodable>(_ url:URL,body:Data,token:String,stage:TransferStage,type:T.Type)async throws->T { var req=request(url,method:"POST",token:token,contentType:"application/json"); req.httpBody=body; let(data,response)=try await session.data(for:req); try validate(response,data:data,stage:stage); do { return try JSONDecoder().decode(T.self,from:data) } catch { throw TransferError(stage:stage,message:"Cannot decode receiver response",responseBody:bodyText(data)) } }
    private func request(_ url:URL,method:String,token:String,contentType:String)->URLRequest { var req=URLRequest(url:url); req.httpMethod=method; req.setValue(String(Self.protocolVersion),forHTTPHeaderField:"X-Protocol-Version"); req.setValue(contentType,forHTTPHeaderField:"Content-Type"); if !token.isEmpty { req.setValue("Bearer \(token)",forHTTPHeaderField:"Authorization") }; return req }
    private func receiverURL(_ value:String)throws->URL { guard let url=URL(string:value.trimmingCharacters(in:.whitespacesAndNewlines)),["http","https"].contains(url.scheme ?? ""),url.host != nil else { throw TransferError(stage:.health,message:"Invalid receiver URL") }; return url }
    private func endpoint(_ base:URL,_ parts:[String])->URL { parts.reduce(base) { $0.appendingPathComponent($1) } }
    private func localURL(_ root:URL,_ path:String)->URL { path.split(separator:"/").reduce(root) { $0.appendingPathComponent(String($1)) } }
    private func fileSize(_ url:URL)throws->Int64 { guard let size=try url.resourceValues(forKeys:[.fileSizeKey]).fileSize else { throw TransferError(stage:.plan,message:"Cannot read file size") }; return Int64(size) }
    private func sha256(_ url:URL)throws->String { let input=try FileHandle(forReadingFrom:url); defer { try? input.close() }; var hasher=SHA256(); while let chunk=try input.read(upToCount:1024*1024),!chunk.isEmpty { hasher.update(data:chunk) }; return hasher.finalize().map { String(format:"%02x",$0) }.joined() }
    private func validate(_ response:URLResponse,data:Data,stage:TransferStage,batchIndex:Int?=nil)throws { guard let http=response as? HTTPURLResponse else { throw TransferError(stage:stage,message:"Invalid HTTP response",batchIndex:batchIndex) }; guard (200..<300).contains(http.statusCode) else { throw TransferError(stage:stage,message:"Receiver rejected request",batchIndex:batchIndex,statusCode:http.statusCode,responseBody:bodyText(data)) } }
    private func bodyText(_ data:Data)->String { String(data:data.prefix(8192),encoding:.utf8) ?? "<non-UTF8 response>" }
}
