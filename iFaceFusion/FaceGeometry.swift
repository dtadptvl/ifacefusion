import UIKit
import Vision

struct FaceGeometry {
    let boundingBox: CGRect
    let landmarks: [CGPoint]
    let leftEyeOpenRatio: CGFloat
    let rightEyeOpenRatio: CGFloat
    let lipOpenRatio: CGFloat
}
enum FaceGeometryError: LocalizedError {
    case noFace, multipleFaces, missingLandmarks
    var errorDescription: String? {
        switch self {
        case .noFace: "No face was found. Choose a clear portrait with exactly one face."
        case .multipleFaces: "More than one face was found. iFaceFusion requires exactly one face per image."
        case .missingLandmarks: "The face was found, but its key landmarks could not be resolved."
        }
    }
}
enum FaceGeometryDetector {
    static func detect(in image: UIImage) async throws -> FaceGeometry {
        guard let cg = image.normalizedCGImage else { throw FaceGeometryError.noFace }
        return try await Task.detached(priority: .userInitiated) {
            let request = VNDetectFaceLandmarksRequest()
            let handler = VNImageRequestHandler(cgImage: cg, orientation: .up)
            try handler.perform([request])
            let faces = request.results ?? []
            guard faces.count == 1 else {
                throw faces.isEmpty ? FaceGeometryError.noFace : FaceGeometryError.multipleFaces
            }
            guard let face = faces.first, let landmarks = face.landmarks,
                  let leftEye = landmarks.leftEye?.normalizedPoints,
                  let rightEye = landmarks.rightEye?.normalizedPoints,
                  let nose = landmarks.nose?.normalizedPoints,
                  let lips = landmarks.outerLips?.normalizedPoints else {
                throw FaceGeometryError.missingLandmarks
            }

            let width = CGFloat(cg.width)
            let height = CGFloat(cg.height)
            func average(_ points: [CGPoint]) -> CGPoint {
                let sum = points.reduce(.zero) { CGPoint(x: $0.x + $1.x, y: $0.y + $1.y) }
                return CGPoint(x: sum.x / CGFloat(points.count), y: sum.y / CGFloat(points.count))
            }
            func pixel(_ point: CGPoint) -> CGPoint {
                CGPoint(
                    x: (face.boundingBox.minX + point.x * face.boundingBox.width) * width,
                    y: (1 - (face.boundingBox.minY + point.y * face.boundingBox.height)) * height
                )
            }

            let lipPixels = lips.map(pixel)
            let centerX = lipPixels.map(\.x).reduce(0, +) / CGFloat(lipPixels.count)
            let leftMouth = lipPixels.filter { $0.x <= centerX }.min(by: { $0.y < $1.y }) ?? lipPixels[0]
            let rightMouth = lipPixels.filter { $0.x > centerX }.min(by: { $0.y < $1.y }) ?? lipPixels[lipPixels.count / 2]
            let noseTip = pixel(nose.max(by: { $0.y < $1.y }) ?? nose[nose.count / 2])
            let box = CGRect(
                x: face.boundingBox.minX * width,
                y: (1 - face.boundingBox.maxY) * height,
                width: face.boundingBox.width * width,
                height: face.boundingBox.height * height
            )

            func openness(_ points: [CGPoint]) -> CGFloat {
                let converted = points.map(pixel)
                guard let minX = converted.map(\.x).min(),
                      let maxX = converted.map(\.x).max(),
                      let minY = converted.map(\.y).min(),
                      let maxY = converted.map(\.y).max() else { return 0 }
                return (maxY - minY) / max(maxX - minX, 0.000001)
            }

            return FaceGeometry(
                boundingBox: box,
                landmarks: [
                    pixel(average(leftEye)),
                    pixel(average(rightEye)),
                    noseTip,
                    leftMouth,
                    rightMouth
                ],
                leftEyeOpenRatio: openness(leftEye),
                rightEyeOpenRatio: openness(rightEye),
                lipOpenRatio: openness(lips)
            )
        }.value
    }
}

extension UIImage {
    var normalizedCGImage: CGImage? {
        if imageOrientation == .up, let cgImage { return cgImage }

        let pixelSize = CGSize(
            width: max(1, round(size.width * scale)),
            height: max(1, round(size.height * scale))
        )
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = false
        return UIGraphicsImageRenderer(size: pixelSize, format: format).image { _ in
            draw(in: CGRect(origin: .zero, size: pixelSize))
        }.cgImage
    }
}