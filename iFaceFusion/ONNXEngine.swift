import Foundation
import UIKit
import OnnxRuntimeBindings

actor ONNXEngine {
    static let shared = ONNXEngine()

    private let env: ORTEnv
    private var sessions: [String: ORTSession] = [:]

    init() {
        env = try! ORTEnv(loggingLevel: .warning)
    }

    private func session(path: String) throws -> ORTSession {
        if let cached = sessions[path] {
            return cached
        }

        let options = try ORTSessionOptions()
        try? options.setIntraOpNumThreads(2)
        let coreMLOptions = ORTCoreMLExecutionProviderOptions()
        coreMLOptions.enableOnSubgraphs = true
        try? options.appendCoreMLExecutionProvider(with: coreMLOptions)

        let session = try ORTSession(env: env, modelPath: path, sessionOptions: options)
        sessions[path] = session
        return session
    }

    func runImage(
        modelURL: URL,
        image: UIImage,
        inputSize: CGSize,
        normalization: ClosedRange<Float> = 0...1,
        outputRange: ClosedRange<Float> = 0...1
    ) throws -> UIImage {
        let session = try session(path: modelURL.path)
        guard
            let inputName = try session.inputNames().first,
            let outputName = try session.outputNames().first
        else {
            throw InferenceError.io
        }

        let floats = try TensorImage.chw(image, size: inputSize, range: normalization)
        let input = try makeTensor(
            floats,
            shape: [1, 3, NSNumber(value: Int(inputSize.height)), NSNumber(value: Int(inputSize.width))]
        )
        let outputs = try session.run(
            withInputs: [inputName: input],
            outputNames: Set([outputName]),
            runOptions: nil
        )
        guard let output = outputs[outputName] else {
            throw InferenceError.io
        }
        let info = try output.tensorTypeAndShapeInfo()
        let bytes = try output.tensorData()
        return try TensorImage.image(data: bytes, shape: info.shape, range: outputRange)
    }

    func embedding(modelURL: URL, image: UIImage, inputSize: CGSize) throws -> [Float] {
        let session = try session(path: modelURL.path)
        guard
            let inputName = try session.inputNames().first,
            let outputName = try session.outputNames().first
        else {
            throw InferenceError.io
        }

        let floats = try TensorImage.chw(image, size: inputSize, range: -1...1)
        let input = try makeTensor(
            floats,
            shape: [1, 3, NSNumber(value: Int(inputSize.height)), NSNumber(value: Int(inputSize.width))]
        )
        let outputs = try session.run(
            withInputs: [inputName: input],
            outputNames: Set([outputName]),
            runOptions: nil
        )
        guard let output = outputs[outputName] else {
            throw InferenceError.io
        }
        let data = try output.tensorData()
        return floatsFromData(data)
    }

    func faceSwap(
        modelURL: URL,
        sourceEmbedding: [Float],
        targetImage: UIImage,
        inputSize: CGSize
    ) throws -> UIImage {
        let session = try session(path: modelURL.path)
        let inputNames = try session.inputNames()
        guard
            let sourceName = inputNames.first(where: { $0.lowercased().contains("source") }),
            let targetName = inputNames.first(where: { $0.lowercased().contains("target") }),
            let outputName = try session.outputNames().first
        else {
            throw InferenceError.unsupported("The selected face swap model has an unexpected input layout.")
        }

        let targetFloats = try TensorImage.chw(targetImage, size: inputSize, range: -1...1)
        let source = try makeTensor(sourceEmbedding, shape: [1, NSNumber(value: sourceEmbedding.count)])
        let target = try makeTensor(
            targetFloats,
            shape: [1, 3, NSNumber(value: Int(inputSize.height)), NSNumber(value: Int(inputSize.width))]
        )

        let outputs = try session.run(
            withInputs: [sourceName: source, targetName: target],
            outputNames: Set([outputName]),
            runOptions: nil
        )
        guard let output = outputs[outputName] else {
            throw InferenceError.io
        }
        let info = try output.tensorTypeAndShapeInfo()
        let data = try output.tensorData()
        return try TensorImage.image(data: data, shape: info.shape, range: -1...1)
    }

    private func makeTensor(_ floats: [Float], shape: [NSNumber]) throws -> ORTValue {
        let data = floats.withUnsafeBufferPointer { buffer -> NSMutableData in
            NSMutableData(
                bytes: buffer.baseAddress,
                length: buffer.count * MemoryLayout<Float>.size
            )
        }
        return try ORTValue(tensorData: data, elementType: .float, shape: shape)
    }

    private func floatsFromData(_ data: NSMutableData) -> [Float] {
        let count = data.length / MemoryLayout<Float>.size
        let pointer = data.bytes.bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }
}

enum InferenceError: LocalizedError {
    case io
    case unsupported(String)

    var errorDescription: String? {
        switch self {
        case .io:
            return "The model returned an unreadable tensor."
        case .unsupported(let message):
            return message
        }
    }
}

enum TensorImage {
    static func chw(
        _ image: UIImage,
        size: CGSize,
        range: ClosedRange<Float>
    ) throws -> [Float] {
        guard let cgImage = image.normalizedCGImage else {
            throw InferenceError.io
        }

        let width = Int(size.width)
        let height = Int(size.height)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var rgba = [UInt8](repeating: 0, count: width * height * 4)

        guard let context = CGContext(
            data: &rgba,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw InferenceError.io
        }

        context.interpolationQuality = .high
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        var output = [Float](repeating: 0, count: width * height * 3)
        for index in 0..<(width * height) {
            for channel in 0..<3 {
                let value = Float(rgba[index * 4 + channel]) / 255
                output[channel * width * height + index] =
                    range.lowerBound + value * (range.upperBound - range.lowerBound)
            }
        }
        return output
    }

    static func image(
        data: NSMutableData,
        shape: [NSNumber],
        range: ClosedRange<Float>
    ) throws -> UIImage {
        guard shape.count >= 4 else {
            throw InferenceError.io
        }

        let height = shape[shape.count - 2].intValue
        let width = shape[shape.count - 1].intValue
        let count = data.length / MemoryLayout<Float>.size
        guard count >= 3 * width * height else {
            throw InferenceError.io
        }

        let pointer = data.bytes.bindMemory(to: Float.self, capacity: count)
        let denominator = max(range.upperBound - range.lowerBound, 0.000001)
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        for index in 0..<(width * height) {
            for channel in 0..<3 {
                let normalized = (pointer[channel * width * height + index] - range.lowerBound) / denominator
                let value = Int(normalized * 255)
                rgba[index * 4 + channel] = UInt8(max(0, min(255, value)))
            }
        }

        guard
            let provider = CGDataProvider(data: Data(rgba) as CFData),
            let cgImage = CGImage(
                width: width,
                height: height,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: width * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                provider: provider,
                decode: nil,
                shouldInterpolate: true,
                intent: .defaultIntent
            )
        else {
            throw InferenceError.io
        }

        return UIImage(cgImage: cgImage)
    }
}