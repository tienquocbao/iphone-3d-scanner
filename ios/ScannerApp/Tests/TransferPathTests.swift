import XCTest

final class TransferPathTests: XCTestCase {
    func testCanonicalRelativePathUsesProtocolSeparators() throws {
        let root = URL(fileURLWithPath: "/tmp/session_test", isDirectory: true)
        XCTAssertEqual(
            try TransferPath.canonicalRelativePath(
                sessionRoot: root,
                fileURL: root.appendingPathComponent("frames/000000/rgb.jpg")
            ),
            "frames/000000/rgb.jpg"
        )
        XCTAssertEqual(
            try TransferPath.canonicalRelativePath(
                sessionRoot: root,
                fileURL: root.appendingPathComponent("session.json")
            ),
            "session.json"
        )
    }

    func testCanonicalRelativePathRejectsOutsideFile() {
        let root = URL(fileURLWithPath: "/tmp/session_test", isDirectory: true)
        XCTAssertThrowsError(
            try TransferPath.canonicalRelativePath(
                sessionRoot: root,
                fileURL: URL(fileURLWithPath: "/tmp/session_test_backup/file.bin")
            )
        )
    }

    func testCanonicalRelativePathRejectsRootItself() {
        let root = URL(fileURLWithPath: "/tmp/session_test", isDirectory: true)
        XCTAssertThrowsError(try TransferPath.canonicalRelativePath(sessionRoot: root, fileURL: root))
    }

    func testCanonicalRelativePathIsStableForStandardizedFileURLs() throws {
        let root = URL(fileURLWithPath: "/tmp/session_test", isDirectory: true)
        XCTAssertEqual(
            try TransferPath.canonicalRelativePath(
                sessionRoot: root,
                fileURL: root.appendingPathComponent("frames/000128/depth.f32")
            ),
            "frames/000128/depth.f32"
        )
    }
}
