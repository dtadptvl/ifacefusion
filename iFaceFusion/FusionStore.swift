import SwiftUI
import Photos
import PhotosUI

@MainActor
final class FusionStore: ObservableObject {
    @Published var source: UIImage?
    @Published var target: UIImage?
    @Published var result: UIImage?
    @Published var selected: Set<ProcessorID> = [.faceSwap]
    @Published var settings = ProcessorSettings()
    @Published var isProcessing = false
    @Published var status = ""
    @Published var errorMessage: String?
    @Published var showAdvanced = false

    private let pipeline = ProcessingPipeline()

    var canProcess: Bool {
        target != nil &&
        !selected.isEmpty &&
        (!selected.contains(.faceSwap) || source != nil) &&
        !isProcessing
    }

    func load(_ item: PhotosPickerItem?, asSource: Bool) async {
        guard let item else { return }
        do {
            guard
                let data = try await item.loadTransferable(type: Data.self),
                let image = UIImage(data: data)
            else {
                throw LoadError.invalid
            }
            _ = try await FaceGeometryDetector.detect(in: image)
            if asSource {
                source = image
            } else {
                target = image
                result = nil
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func process() {
        guard let target else { return }
        isProcessing = true
        result = nil
        status = "Preparing…"
        errorMessage = nil

        let ordered = ProcessorID.allCases.filter { selected.contains($0) }
        let currentSource = source
        let currentSettings = settings

        Task {
            do {
                let output = try await pipeline.process(
                    source: currentSource,
                    target: target,
                    processors: ordered,
                    settings: currentSettings
                ) { step in
                    Task { @MainActor in
                        self.status = step
                    }
                }
                result = output
                status = "Done"
            } catch {
                errorMessage = error.localizedDescription
                status = ""
            }
            isProcessing = false
        }
    }

    func save() {
        guard let result else { return }
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            guard status == .authorized || status == .limited else { return }
            UIImageWriteToSavedPhotosAlbum(result, nil, nil, nil)
        }
    }
}

enum LoadError: LocalizedError {
    case invalid
    var errorDescription: String? {
        "The selected item is not a readable image."
    }
}