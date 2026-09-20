import Foundation

actor ModelStore {
    static let shared = ModelStore()
    private let fm = FileManager.default
    private var root: URL { fm.urls(for:.applicationSupportDirectory,in:.userDomainMask)[0].appendingPathComponent("iFaceFusion/Models",isDirectory:true) }
    func localURL(for asset: ModelAsset) -> URL { root.appendingPathComponent(asset.file) }
    func isCached(_ asset: ModelAsset) -> Bool { fm.fileExists(atPath: localURL(for:asset).path) }
    func ensure(_ asset: ModelAsset) async throws -> URL {
        try fm.createDirectory(at:root,withIntermediateDirectories:true)
        let dst=localURL(for:asset); if fm.fileExists(atPath:dst.path) { return dst }
        let (tmp,response)=try await URLSession.shared.download(from:asset.remoteURL)
        guard let http=response as? HTTPURLResponse,(200..<300).contains(http.statusCode) else { throw ModelError.download(asset.file) }
        try fm.moveItem(at:tmp,to:dst); return dst
    }
    func removeAll() throws { if fm.fileExists(atPath:root.path) { try fm.removeItem(at:root) } }
}
enum ModelError: LocalizedError {
    case download(String)
    var errorDescription:String? { switch self { case .download(let f): "Could not download \(f)." } }
}