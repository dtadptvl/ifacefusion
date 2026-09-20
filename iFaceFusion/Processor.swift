import Foundation

enum ProcessorID: String, CaseIterable, Identifiable, Codable {
    case faceSwap, deepSwap, faceEnhance, age, expressionRestore, faceEdit
    case backgroundRemove, colourise, frameEnhance
    var id: String { rawValue }
    var title: String {
        switch self {
        case .faceSwap: "Face Swap"; case .deepSwap: "Deep Swap"; case .faceEnhance: "Face Enhance"
        case .age: "Age"; case .expressionRestore: "Expression Restore"; case .faceEdit: "Face Edit"
        case .backgroundRemove: "Background Remove"; case .colourise: "Colourise"; case .frameEnhance: "Frame Enhance"
        }
    }
    var subtitle: String {
        switch self {
        case .faceSwap: "Transfer Source identity"; case .deepSwap: "DFL identity model"; case .faceEnhance: "Restore facial detail"
        case .age: "Adjust apparent age"; case .expressionRestore: "Restore Target expression"; case .faceEdit: "Adjust expression and pose"
        case .backgroundRemove: "Portrait matting"; case .colourise: "Colourise monochrome image"; case .frameEnhance: "Upscale the full image"
        }
    }
    var systemImage: String {
        switch self {
        case .faceSwap: "person.2.crop.square.stack"; case .deepSwap: "person.crop.square.filled.and.at.rectangle"; case .faceEnhance: "wand.and.stars"
        case .age: "clock.arrow.trianglehead.counterclockwise.rotate.90"; case .expressionRestore: "face.smiling"; case .faceEdit: "slider.horizontal.3"
        case .backgroundRemove: "person.crop.rectangle.badge.minus"; case .colourise: "paintpalette"; case .frameEnhance: "arrow.up.left.and.arrow.down.right"
        }
    }
    var needsSource: Bool { self == .faceSwap }
    var changesResolution: Bool { self == .frameEnhance }
}

struct ProcessorSettings {
    var faceSwapWeight = 0.50, faceEnhanceBlend = 0.80, ageDirection = 0.0
    var expressionFactor = 0.80, backgroundOpacity = 0.0, colourBlend = 1.0
    var frameEnhanceBlend = 0.80, faceEditSmile = 0.0, faceEditEyeOpen = 0.0
    var frameScale = 2
}