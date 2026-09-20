import UIKit
import Vision

struct FaceGeometry { let boundingBox:CGRect; let landmarks:[CGPoint] }
enum FaceGeometryError: LocalizedError {
    case noFace,multipleFaces,missingLandmarks
    var errorDescription:String? {
        switch self {
        case .noFace:"No face was found. Choose a clear portrait with exactly one face."
        case .multipleFaces:"More than one face was found. iFaceFusion requires exactly one face per image."
        case .missingLandmarks:"The face was found, but its key landmarks could not be resolved."
        }
    }
}
enum FaceGeometryDetector {
    static func detect(in image:UIImage) async throws -> FaceGeometry {
        guard let cg=image.normalizedCGImage else { throw FaceGeometryError.noFace }
        return try await Task.detached(priority:.userInitiated) {
            let req=VNDetectFaceLandmarksRequest(); let handler=VNImageRequestHandler(cgImage:cg,orientation:.up)
            try handler.perform([req]); let faces=req.results ?? []
            guard faces.count==1 else { throw faces.isEmpty ? FaceGeometryError.noFace : FaceGeometryError.multipleFaces }
            guard let face=faces.first,let lm=face.landmarks,
                  let le=lm.leftEye?.normalizedPoints,let re=lm.rightEye?.normalizedPoints,
                  let nose=lm.nose?.normalizedPoints,let lips=lm.outerLips?.normalizedPoints else { throw FaceGeometryError.missingLandmarks }
            let w=CGFloat(cg.width),h=CGFloat(cg.height)
            func avg(_ ps:[CGPoint])->CGPoint { let s=ps.reduce(.zero){CGPoint(x:$0.x+$1.x,y:$0.y+$1.y)}; return CGPoint(x:s.x/CGFloat(ps.count),y:s.y/CGFloat(ps.count)) }
            func px(_ p:CGPoint)->CGPoint { CGPoint(x:(face.boundingBox.minX+p.x*face.boundingBox.width)*w,y:(1-(face.boundingBox.minY+p.y*face.boundingBox.height))*h) }
            let lip=lips.map(px), cx=lip.map(\.x).reduce(0,+)/CGFloat(lip.count)
            let l=lip.filter{$0.x<=cx}.min(by:{$0.y<$1.y}) ?? lip[0]
            let r=lip.filter{$0.x>cx}.min(by:{$0.y<$1.y}) ?? lip[lip.count/2]
            let n=px(nose.max(by:{$0.y<$1.y}) ?? nose[nose.count/2])
            let box=CGRect(x:face.boundingBox.minX*w,y:(1-face.boundingBox.maxY)*h,width:face.boundingBox.width*w,height:face.boundingBox.height*h)
            return FaceGeometry(boundingBox:box,landmarks:[px(avg(le)),px(avg(re)),n,l,r])
        }.value
    }
}
extension UIImage {
    var normalizedCGImage:CGImage? {
        if imageOrientation == .up,let cgImage { return cgImage }
        return UIGraphicsImageRenderer(size:size).image{_ in draw(in:CGRect(origin:.zero,size:size))}.cgImage
    }
}