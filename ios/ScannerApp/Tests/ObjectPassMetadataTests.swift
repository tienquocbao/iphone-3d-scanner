import XCTest

@testable import ScannerApp

final class ObjectPassMetadataTests: XCTestCase {
    func testContiguousGlobalFramePassesRoundTrip() throws {
        let passes = [
            ScanPassMetadata(id: 0, startFrame: 0, endFrame: 74),
            ScanPassMetadata(id: 1, startFrame: 75, endFrame: 141)
        ]
        let data = try JSONEncoder().encode(passes)
        XCTAssertEqual(try JSONDecoder().decode([ScanPassMetadata].self, from: data), passes)
        XCTAssertEqual(passes[0].endFrame + 1, passes[1].startFrame)
    }
}
