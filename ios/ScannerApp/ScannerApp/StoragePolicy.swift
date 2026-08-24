import Foundation

struct StoragePolicy {
    let minimumFreeBytesToStart: Int64
    let minimumFreeBytesDuringRecording: Int64
    let maximumSessionBytes: Int64

    static let `default` = StoragePolicy(
        minimumFreeBytesToStart: 2 * 1024 * 1024 * 1024,
        minimumFreeBytesDuringRecording: 1 * 1024 * 1024 * 1024,
        maximumSessionBytes: 2 * 1024 * 1024 * 1024
    )

    func availableBytes(at url: URL) -> Int64? {
        let keys: Set<URLResourceKey> = [.volumeAvailableCapacityForImportantUsageKey, .volumeAvailableCapacityKey]
        guard let values = try? url.resourceValues(forKeys: keys) else { return nil }
        if let importantCapacity = values.volumeAvailableCapacityForImportantUsage {
            return importantCapacity
        }
        return values.volumeAvailableCapacity.map(Int64.init)
    }

    func canStart(at url: URL) -> Bool {
        guard let available = availableBytes(at: url) else { return true }
        return available >= minimumFreeBytesToStart
    }

    func canContinue(at url: URL, sessionBytes: Int64) -> Bool {
        guard sessionBytes < maximumSessionBytes else { return false }
        guard let available = availableBytes(at: url) else { return true }
        return available >= minimumFreeBytesDuringRecording
    }
}
