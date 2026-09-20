import Foundation

actor ModelStore {
    static let shared = ModelStore()

    private let fm = FileManager.default

    private var root: URL {
        fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("iFaceFusion/Models", isDirectory: true)
    }

    func localURL(for asset: ModelAsset) -> URL {
        root.appendingPathComponent(asset.file)
    }

    private func localHashURL(for asset: ModelAsset) -> URL {
        localURL(for: asset)
            .deletingPathExtension()
            .appendingPathExtension("hash")
    }

    func isCached(_ asset: ModelAsset) -> Bool {
        fm.fileExists(atPath: localURL(for: asset).path)
    }

    func ensure(_ asset: ModelAsset) async throws -> URL {
        try fm.createDirectory(at: root, withIntermediateDirectories: true)

        let destination = localURL(for: asset)
        let hashDestination = localHashURL(for: asset)

        if fm.fileExists(atPath: destination.path) {
            if fm.fileExists(atPath: hashDestination.path) {
                let expected = try String(contentsOf: hashDestination, encoding: .utf8)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                    .lowercased()
                if try crc32Hex(of: destination) == expected {
                    return destination
                }
                try? fm.removeItem(at: destination)
                try? fm.removeItem(at: hashDestination)
            } else {
                // Preserve offline usability for a model cached by an older app build.
                return destination
            }
        }

        let (hashData, hashResponse) = try await URLSession.shared.data(from: asset.hashRemoteURL)
        guard
            let hashHTTP = hashResponse as? HTTPURLResponse,
            (200..<300).contains(hashHTTP.statusCode),
            let expectedHash = String(data: hashData, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased(),
            expectedHash.count == 8
        else {
            throw ModelError.download(asset.file + " hash")
        }

        let (temporary, response) = try await URLSession.shared.download(from: asset.remoteURL)
        guard
            let http = response as? HTTPURLResponse,
            (200..<300).contains(http.statusCode)
        else {
            throw ModelError.download(asset.file)
        }

        if fm.fileExists(atPath: destination.path) {
            try fm.removeItem(at: destination)
        }
        try fm.moveItem(at: temporary, to: destination)

        let actualHash = try crc32Hex(of: destination)
        guard actualHash == expectedHash else {
            try? fm.removeItem(at: destination)
            throw ModelError.integrity(asset.file)
        }

        try Data(expectedHash.utf8).write(to: hashDestination, options: .atomic)
        return destination
    }

    func removeAll() throws {
        if fm.fileExists(atPath: root.path) {
            try fm.removeItem(at: root)
        }
    }

    private func crc32Hex(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }

        var crc: UInt32 = 0xFFFF_FFFF
        while let data = try handle.read(upToCount: 1024 * 1024), !data.isEmpty {
            for byte in data {
                crc ^= UInt32(byte)
                for _ in 0..<8 {
                    if (crc & 1) != 0 {
                        crc = (crc >> 1) ^ 0xEDB8_8320
                    } else {
                        crc >>= 1
                    }
                }
            }
        }
        crc ^= 0xFFFF_FFFF
        return String(format: "%08x", crc)
    }
}

enum ModelError: LocalizedError {
    case download(String)
    case integrity(String)

    var errorDescription: String? {
        switch self {
        case .download(let file):
            return "Could not download \(file)."
        case .integrity(let file):
            return "Downloaded model \(file) failed its FaceFusion CRC32 integrity check."
        }
    }
}
