import SwiftUI
import Photos
import PhotosUI

@MainActor
final class FusionStore:ObservableObject {
    @Published var source:UIImage?,target:UIImage?,result:UIImage?
    @Published var selected:Set<ProcessorID>=[.faceSwap]
    @Published var settings=ProcessorSettings()
    @Published var isProcessing=false
    @Published var status="",errorMessage:String?
    @Published var showAdvanced=false
    private let pipeline=ProcessingPipeline()

    var canProcess:Bool { target != nil && !selected.isEmpty && (!selected.contains(.faceSwap) || source != nil) && !isProcessing }

    func load(_ item:PhotosPickerItem?,asSource:Bool) async {
        guard let item else{return}
        do {
            guard let data=try await item.loadTransferable(type:Data.self),let image=UIImage(data:data) else{throw LoadError.invalid}
            _=try await FaceGeometryDetector.detect(in:image)
            if asSource { source=image } else { target=image; result=nil }
        } catch { errorMessage=error.localizedDescription }
    }

    func process() {
        guard let target else{return}
        isProcessing=true; result=nil; status="Preparing…"; errorMessage=nil
        let ordered=ProcessorID.allCases.filter{selected.contains($0)},source=self.source,settings=self.settings
        Task {
            do {
                let out=try await pipeline.process(source:source,target:target,processors:ordered,settings:settings){ step in
                    Task { @MainActor in self.status=step }
                }
                result=out; status="Done"
            } catch { errorMessage=error.localizedDescription; status="" }
            isProcessing=false
        }
    }

    func save() {
        guard let result else{return}
        PHPhotoLibrary.requestAuthorization(for:.addOnly){ status in
            guard status == .authorized || status == .limited else{return}
            UIImageWriteToSavedPhotosAlbum(result,nil,nil,nil)
        }
    }
}
enum LoadError:LocalizedError { case invalid; var errorDescription:String?{"The selected item is not a readable image."} }
